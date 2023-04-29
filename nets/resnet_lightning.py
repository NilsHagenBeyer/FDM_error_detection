import os
import torch.nn as nn
import torchvision.models as models
import pytorch_lightning as pl

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))

#DATASET_NAME = "PRINTING_ERRORS"
#PATH = "%s/PRINTING_ERRORS/train_images/" %(SCRIPT_DIR)
#TRAIN_CSV = "%s/%s/train_hand_select.csv" %(SCRIPT_DIR, DATASET_NAME)
#TEST_CSV = "%s/%s/test_hand_select.csv" %(SCRIPT_DIR, DATASET_NAME)

# Here we define a new class to turn the ResNet model that we want to use as a feature extractor
# into a pytorch-lightning module so that we can take advantage of lightning's Trainer object.
# We aim to make it a little more general by allowing users to define the number of prediction classes.
class ResNetClassifier(pl.LightningModule):
    resnets = {
        18: models.resnet18,
        34: models.resnet34,
        50: models.resnet50,
        101: models.resnet101,
        152: models.resnet152,
    }

    def __init__(
        self,
        num_classes = 3,
        version = 18,
        #Für Transfer beides True, sonst beides False
        use_transfer=False,
        tune_fc_only=False,
    ):
        super(ResNetClassifier, self).__init__()

        self.num_classes = num_classes
        # Using a pretrained ResNet backbone
        self.resnet_model = self.resnets[version](pretrained=use_transfer)

        # Replace old FC layer with Identity so we can train our own
        linear_size = list(self.resnet_model.children())[-1].in_features
        # replace final layer for fine tuning
        self.resnet_model.fc = nn.Linear(linear_size, num_classes)



        if tune_fc_only:  # option to only tune the fully-connected layers
            for child in list(self.resnet_model.children())[:-1]:
                for param in child.parameters():
                    param.requires_grad = False

    def forward(self, X):
        return self.resnet_model(X)

#     def configure_optimizers(self):
#         return self.optimizer(self.parameters(), lr=self.lr)

#     def _step(self, batch):
#         x, y = batch
#         preds = self(x)

#         if self.num_classes == 1:
#             preds = preds.flatten()
#             y = y.float()

#         loss = self.loss_fn(preds, y)
#         acc = self.acc(preds, y)
#         return loss, acc


#     def training_step(self, batch, batch_idx):
#         loss, acc = self._step(batch)
#         # perform logging
#         self.log(
#             "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
#         )
#         self.log(
#             "train_acc", acc, on_step=True, on_epoch=True, prog_bar=True, logger=True
#         )
#         return loss

#     def validation_step(self, batch, batch_idx):
#         loss, acc = self._step(batch)
#         # perform logging
#         self.log("val_loss", loss, on_epoch=True, prog_bar=False, logger=True)
#         self.log("val_acc", acc, on_epoch=True, prog_bar=True, logger=True)

#     def test_dataloader(self):
#         return self._dataloader(self.test_path)

#     def test_step(self, batch, batch_idx):
#         loss, acc = self._step(batch)
#         # perform logging
#         self.log("test_loss", loss, on_step=True, prog_bar=True, logger=True)
#         self.log("test_acc", acc, on_step=True, prog_bar=True, logger=True)

# class PrintErrorDataset(Dataset):
#     def __init__(self,path,image_ids,labels,image_size):
#         self.image_ids = image_ids
#         self.labels = labels
#         self.path = path
#         self.image_size = image_size

#     def __len__(self):
#         return len(self.image_ids)
    
#     def __getitem__(self,item):
#       image_ids = str(self.image_ids[item])
#       labels = self.labels[item]
#       img_file = cv2.imread(self.path+image_ids)
#       img = cv2.resize(img_file,(self.image_size,self.image_size))
#       img = img.astype(np.float64)

#       return {
#             "x":torch.tensor(img,dtype=torch.float),
#             "y":torch.tensor(labels,dtype=torch.long),
#         }

# class PrintErrorDatamodule(pl.LightningDataModule):
#     def __init__(self,batch_size=64, folds=None):
#       super().__init__()
#       self.batch_size = batch_size
#       self.folds = folds
#       self.kfold_data_loaders = None
    
#     def setup(self,stage=None):
#       dfx = pd.read_csv(TRAIN_CSV, delimiter=";")
#       df_test = pd.read_csv(TEST_CSV, delimiter=";")

      
      
#       xtrain, xval, ytrain, yval = train_test_split(dfx["image"].values,
#                                                       dfx["class"].values,
#                                                       test_size = 0.1)
      
#       xtest = df_test["image"].values
#       ytest = df_test["class"].values

      
#       self.train_dataset = PrintErrorDataset(PATH,xtrain,ytrain,IMG_SIZE)
#       self.validation_dataset = PrintErrorDataset(PATH,xval,yval,IMG_SIZE)
#       self.test_dataset = PrintErrorDataset(PATH,xtest,ytest,IMG_SIZE)
#       if self.folds:
#         self.kfold_data_loaders = self.get_splits(dfx)


#     def get_splits(self, dataframe):
#       skf = StratifiedKFold(n_splits=self.folds, random_state=None, shuffle=False)
#       data = []
#       for train_index, val_index in skf.split(dataframe["image"].values, dataframe["class"].values):          
#           xtrain, xval = dataframe["image"].values[train_index], dataframe["image"].values[val_index]
#           ytrain, yval = dataframe["class"].values[train_index], dataframe["class"].values[val_index]
#           fold_test =  DataLoader(PrintErrorDataset(PATH,xtrain,ytrain,IMG_SIZE), batch_size=self.batch_size, num_workers=24)
#           fold_val = DataLoader(PrintErrorDataset(PATH,xval,yval,IMG_SIZE), batch_size=self.batch_size, num_workers=24)
#           data.append((fold_test, fold_val))      
#       return data


#     def train_dataloader(self):
#       train_loader = DataLoader(self.train_dataset,
#                             batch_size=self.batch_size,
#                             shuffle=True, num_workers=20)
#       return train_loader
#     def val_dataloader(self):
#       valid_loader = DataLoader(self.validation_dataset,
#                             batch_size=self.batch_size,
#                             shuffle=False, num_workers=4)       
#       return valid_loader
    
#     def test_dataloader(self):
#        return DataLoader(self.test_dataset, batch_size=self.batch_size,
#                             shuffle=False, num_workers=24)

#     def get_kfold_dataloaders(self):
#       return self.kfold_data_loaders


# if __name__ == "__main__":
#     parser = ArgumentParser()
#     # Required arguments
#     parser.add_argument(
#         "model",
#         help="""Choose one of the predefined ResNet models provided by torchvision. e.g. 50""",
#         type=int,
#     )
#     parser.add_argument(
#         "num_classes", help="""Number of classes to be learned.""", type=int
#     )
#     parser.add_argument("num_epochs", help="""Number of Epochs to Run.""", type=int)
#     # Optional arguments
#     parser.add_argument(
#         "-amp",
#         "--mixed_precision",
#         help="""Use mixed precision during training. Defaults to False.""",
#         action="store_true",
#     )
#     parser.add_argument(
#         "-o",
#         "--optimizer",
#         help="""PyTorch optimizer to use. Defaults to adam.""",
#         default="adam",
#     )
#     parser.add_argument(
#         "-lr",
#         "--learning_rate",
#         help="Adjust learning rate of optimizer.",
#         type=float,
#         default=1e-3,
#     )
#     parser.add_argument(
#         "-b",
#         "--batch_size",
#         help="""Manually determine batch size. Defaults to 16.""",
#         type=int,
#         default=16,
#     )
#     parser.add_argument(
#         "-tr",
#         "--transfer",
#         help="""Determine whether to use pretrained model or train from scratch. Defaults to True.""",
#         action="store_true",
#     )
#     parser.add_argument(
#         "-to",
#         "--tune_fc_only",
#         help="Tune only the final, fully connected layers.",
#         action="store_true",
#     )
#     parser.add_argument(
#         "-s", "--save_path", help="""Path to save model trained model checkpoint."""
#     )
#     parser.add_argument(
#         "-g", "--gpus", help="""Enables GPU acceleration.""", type=int, default=None
#     )
#     args = parser.parse_args()
    
    
#     # # Instantiate Model
#     '''model = ResNetClassifier(
#         num_classes=args.num_classes,
#         resnet_version=args.model,
#         test_path=args.test_set,
#         optimizer=args.optimizer,
#         lr=args.learning_rate,
#         batch_size=args.batch_size,
#         transfer=args.transfer,
#         tune_fc_only=args.tune_fc_only,
#     )'''


#     model = ResNetClassifier(
#         num_classes=3,
#         resnet_version=50,
#         optimizer=args.optimizer,
#         lr=args.learning_rate,
#         batch_size=args.batch_size,
#         transfer=args.transfer,
#         tune_fc_only=args.tune_fc_only,
#     )

#     save_path = args.save_path if args.save_path is not None else "./models"
#     checkpoint_callback = pl.callbacks.ModelCheckpoint(
#         dirpath=save_path,
#         filename="resnet-model-{epoch}-{val_loss:.2f}-{val_acc:0.2f}",
#         monitor="val_loss",
#         save_top_k=3,
#         mode="min",
#         save_last=True,
#     )

#     stopping_callback = pl.callbacks.EarlyStopping(monitor="val_loss", patience=5, mode="min")

#     dx = PrintErrorDatamodule(model)

#     # Instantiate lightning trainer and train model
#     trainer_args = {
#         "accelerator": "gpu",
#         "max_epochs": args.num_epochs,
#         "callbacks": [checkpoint_callback],
#         "precision": 16 if args.mixed_precision else 32,
#         "logger":TensorBoardLogger("./tblogs",name="ResNet50", version="test"),
#     }
#     trainer = pl.Trainer(**trainer_args)

#     trainer.fit(model=model, datamodule=dx)

#     if args.test_set:
#         trainer.test(model)
#     # Save trained model weights
#     torch.save(trainer.model.resnet_model.state_dict(), save_path + "/trained_model.pt")