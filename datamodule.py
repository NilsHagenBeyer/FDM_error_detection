import pytorch_lightning as pl
import pandas as pd
import cv2
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from augmentation import Augmentations
from albumentations.core.composition import Compose
import albumentations as A
from albumentations.pytorch import ToTensorV2


class PrintingErrorDataset(Dataset):
    def __init__(self, path, image_ids, labels, transform = A.Compose([
                                                            A.Resize(256, 256),
                                                            A.Normalize(),
                                                            ToTensorV2(),
                                                        ])):
        self.image_ids = image_ids
        self.labels = labels
        self.path = path
        self.transform = transform

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, item):
        image_ids = str(self.image_ids[item])
        label = self.labels[item]
        image = cv2.imread(self.path+image_ids)
        # apply augmentations
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        transformed = self.transform(image=image)
        # img = cv2.resize(img_file,(self.image_size,self.image_size))
        image = transformed["image"]

        return image, torch.tensor(label, dtype=torch.long)


class PrintingErrorDatamodule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.batch_size = args.batch_size
        self.folds = args.folds
        self.kfold_data_loaders = None
        self.image_path =args.im_path
        self.test_path = args.test
        self.train_path = args.train
        self.val_path = args.val
        self.workers = args.workers
        
        # specify augmentation strategy
        if args.aug=="vertical_flip":
            self.transform = Augmentations.vertical_flip()
        elif args.aug=="horizontal_flip":
            self.transform = Augmentations.horizontal_flip()
        elif args.aug=="image_aug":
            self.transform = Augmentations.image_aug()
        elif args.aug=="geometric_aug":
            self.transform = Augmentations.geometric_aug()
        elif args.aug=="weak_aug":
            self.transform = Augmentations.weak_aug()
        elif args.aug=="strong_aug":
            self.transform = Augmentations.strong_aug()
        else:
            self.transform = Augmentations.no_aug()

    # is called by trainer or has to be called manually to load training data
    def setup(self, stage=None):        
        # check if validation set is specified
        # use validation set if specified
        if self.val_path != "":           
            # load train data
            print("==============================================================================")
            df_train = pd.read_csv(self.train_path, delimiter=";")
            xtrain = df_train["image"].values
            ytrain = df_train["class"].values
            print("Load train data: %s instances" % (len(df_train)))
            print("from: %s" % (self.train_path))
            
            # load validation data
            df_val = pd.read_csv(self.val_path, delimiter=";")
            xval = df_val["image"].values
            yval = df_val["class"].values
            print("Load validation data: %s instances" % (len(df_val)))
            print("from: %s" % (self.val_path))
            print("==============================================================================")
        
        # make random split if no validation set is specified
        else:
            if self.train_path != "": 
                print("Make random split of train data")
                dfx = pd.read_csv(self.train_path, delimiter=";")
                # split val and train
                xtrain, xval, ytrain, yval = train_test_split(dfx["image"].values,
                                                            dfx["class"].values,
                                                            test_size=0.2 , stratify=dfx["class"])
            else :
                print("No train data path specified: only testing")

        # Create train and validation dataset
        self.train_dataset = PrintingErrorDataset(self.image_path, xtrain, ytrain, transform=self.transform)
        self.validation_dataset = PrintingErrorDataset(self.image_path, xval, yval)
        
        # make test data if specified
        if self.test_path != "":
            df_test = pd.read_csv(self.test_path, delimiter=";")
            xtest = df_test["image"].values
            ytest = df_test["class"].values
            # Create test dataset
            self.test_dataset = PrintingErrorDataset(self.image_path, xtest, ytest)


        
        # if kfold is specified make kfold splits (old implementation, handle with care)
        # to use, specify splits and call get_kfold_dataloaders() to get a list of dataloaders for the splits
        if self.folds:
            self.kfold_data_loaders = self.get_splits(dfx)

    def get_splits(self, dataframe):
        print("MAKE KFOLD SPLIT")
        print("Image data path: %s" % (self.image_path))
        skf = StratifiedKFold(n_splits=self.folds,
                              random_state=None, shuffle=False)
        data = []
        # iterate over splits and read train and validation data, create dataloader for each split
        for train_index, val_index in skf.split(dataframe["image"].values, dataframe["class"].values):
            # read data
            xtrain, xval = dataframe["image"].values[train_index], dataframe["image"].values[val_index]
            ytrain, yval = dataframe["class"].values[train_index], dataframe["class"].values[val_index]
            # create dataloaders
            fold_train = DataLoader(PrintingErrorDataset(
                self.image_path, xtrain, ytrain), batch_size=self.batch_size, num_workers=self.workers)
            fold_val = DataLoader(PrintingErrorDataset(
                self.image_path, xval, yval), batch_size=self.batch_size, num_workers=4)
            data.append((fold_train, fold_val))
        return data

    
    def train_dataloader(self):
        train_loader = DataLoader(self.train_dataset,
                                  batch_size=self.batch_size,
                                  shuffle=True, num_workers=self.workers)
        return train_loader

    def val_dataloader(self):
        valid_loader = DataLoader(self.validation_dataset,
                                  batch_size=self.batch_size,
                                  shuffle=False, num_workers=4)
        return valid_loader

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size,
                          shuffle=False, num_workers=4)
    
    # call to get a list of dataloaders for each split, folds parameter 
    def get_kfold_dataloaders(self):
        return self.kfold_data_loaders

    def make_dataloader(self, path, image_path=None):
        if image_path is None:
            image_path = self.image_path
        df_test = pd.read_csv(path, delimiter=";")
        xtest = df_test["image"].values
        ytest = df_test["class"].values
        test_dataset = PrintingErrorDataset(image_path, xtest, ytest)

        return DataLoader(test_dataset, batch_size=self.batch_size,
                          shuffle=False, num_workers=16)