from argparse import ArgumentParser
import pytorch_lightning as pl
import pandas as pd
import os
from torch import nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
import torchmetrics
# from torch.utils.tensorboard import SummaryWriter
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import Callback
from nets.resnet_lightning import ResNetClassifier
from nets.base_cnn import BaseCNN
from albumentations.core.composition import Compose
import albumentations as A
from albumentations.pytorch import ToTensorV2
from augmentation import Augmentations
from torchmetrics.classification import MulticlassConfusionMatrix
from torchmetrics.classification import MulticlassRecall
from torchmetrics.classification import MulticlassPrecision
import csv
import utils.sorting_utility as util


SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))

DATASET_NAME = "PRINTING_ERRORS"
PATH = "%s/PRINTING_ERRORS/train_images256/" % (SCRIPT_DIR)
TRAIN_CSV = "%s/%s/train_hand_select.csv" % (SCRIPT_DIR, DATASET_NAME)
TEST_CSV = "%s/%s/test_hand_select.csv" % (SCRIPT_DIR, DATASET_NAME)

net_zoo = {
    "baseCnn": BaseCNN,
    "ResNet": ResNetClassifier,
}


class ErrorDetectionModel(pl.LightningModule):
    def __init__(self, args):
        super(ErrorDetectionModel, self).__init__()
        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=3)
        self.net = net_zoo[args.net](
            use_transfer=args.use_transfer, 
            tune_fc_only=args.tune_fc_only, 
            version=args.net_version)
        self.args = args
        self.lr = args.lr
        self.use_transfer = args.use_transfer
        self.image_size = args.image_size
        self.classes = args.classes

    def forward(self, x):
        return self.net(x)

    def loss_fn(self, out, target):
        return nn.CrossEntropyLoss()(out.view(-1, self.classes), target)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer

    def training_step(self, batch, batch_idx):
        img, label = batch 
        out = self(img)
        loss = self.loss_fn(out, label)
        accu = self.accuracy(out, label)
        self.log('train_loss', loss)
        self.log("train_acc", accu, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        img, label = batch
        out = self(img)
        loss = self.loss_fn(out, label)
        accu = self.accuracy(out, label)
        self.log('val_loss', loss)
        self.log('val_acc', accu, prog_bar=True)
        return loss, accu
    
    '''
    def test_step_old(self, batch, batch_idx, dataloader_idx):
        loss, acc = self._shared_eval_step(batch, batch_idx)
        
        if dataloader_idx:
            metrics = {"test_acc_%s" %dataloader_idx: acc, "test_loss": loss}
        else:
            metrics = {"test_acc": acc, "test_loss": loss}       
        self.log_dict(metrics)
        return metrics
    '''

    def test_step(self, batch, batch_idx, *args):
        if len(args) == 0:
            dataloader_idx = 1
        else:
            dataloader_idx = args[0]
        img, label = batch
        out = self(img)
        accu = self.accuracy(out, label)
        # log test accuracy
        metrics = {"test_acc_%s" %dataloader_idx: accu}
        self.log_dict(metrics, prog_bar=True)
        return (out,label)

    def test_epoch_end(self, all_outputs) -> None:
        
        # check if multiple test dataloaders were applied, nest outputs into list otherwise
        if not type(all_outputs[0]) == list:
            all_outputs = [all_outputs,  ]
        else:
            print("Multiple test dataloaders found")

        # iterate over the outputs of the test predictions in case of multiple test dataloaders
        for i, outputs in  enumerate(all_outputs):            
            # extract contents
            out = torch.cat([x[0] for x in outputs])
            label = torch.cat([x[1] for x in outputs])
            logits = torch.argmax(out,dim=1)
            
            # Cretate metics 
            # CONFUSION MATRIX
            conf_matrix = MulticlassConfusionMatrix(num_classes=3).to("cuda")
            matrix = conf_matrix(logits, label)
            # save confusion matrix as csv
            # generate name and path
            path = self.logger.save_dir
            name = self.logger.name
            version = self.logger.version
            current_logdir = os.path.join(path, name, version)
            csv_name = "%s/conf_matrix_%s_%s_tl(%s).csv" % (current_logdir, name, version, i)
            os.path.join(path, name, version, csv_name)
            # convert to dataframe and save as csv
            matrix = matrix.to("cpu").numpy()
            matrix_df = pd.DataFrame(matrix)
            print(matrix_df)
            print(csv_name)
            util.dump_csv(matrix_df, csv_name)


            # RECALL
            recall = MulticlassRecall(num_classes=3).to("cuda")
            recall = recall(logits, label)
            self.log('recall_tl(%s)' %i, recall)
            
            # PRECISION
            precision = MulticlassPrecision(num_classes=3).to("cuda")
            precision = precision(logits, label)
            self.log('precision_tl(%s)' %i, precision)
        return