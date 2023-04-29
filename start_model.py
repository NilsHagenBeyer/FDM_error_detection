from glob import glob
import os
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from argparse import ArgumentParser
from nets.resnet_lightning import ResNetClassifier
from nets.base_cnn import BaseCNN
from nets.convnext_lightning import ConvNextClassifier
from error_detection_model import ErrorDetectionModel
from datamodule import PrintingErrorDatamodule, PrintingErrorDataset
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import utils.sorting_utility as util

# directory of this script
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))

# select model from the following implementations
net_zoo = {
    "baseCnn": BaseCNN,
    "ResNet": ResNetClassifier,
}

# Set up argument parser
parser = ArgumentParser()
# Model selection
parser.add_argument("--net", type=str,
                    choices=list(net_zoo.keys()), default="ResNet")
parser.add_argument("--net_version", type=int, default=18, help="Choose ResNet version: 18, 34, 50, 101, 152")
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--epochs", type=int, default=200)
parser.add_argument("--image_size", type=int, default=256)
parser.add_argument("--classes", type=int, default=3, help="Number of classes in the dataset")
# Transfer Learning
parser.add_argument("--use_transfer", default=True,
                    help="whether to use Transfer Learning")
parser.add_argument("--tune_fc_only", type=bool, default=True)
# Logging
parser.add_argument("--exp_name", type=str, default="test")
parser.add_argument("--exp_version", type=str, default="v0")
# Dataloader
parser.add_argument("--aug", type=str, default="weak")
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--folds", type=int, default=None, help="Split given dataset into k folds") # handle with care !!! (tested but not veryfied, use manually split data instead)
parser.add_argument("--workers", type=int, default=16)
# Dataset paths
parser.add_argument("--train", type=str, default="")
parser.add_argument("--test", type=str, default="")
parser.add_argument("--val", type=str, default="")
parser.add_argument("--im_path", type=str, default="")
parser.add_argument("--test_ckpt", type=str, default="")
args = parser.parse_args()


# set up model checkpoint callback
checkpoint_callback = pl.callbacks.ModelCheckpoint(
    monitor='val_acc',
    dirpath='./models/',
    filename='%s_%s_models-{epoch:02d}-{val_acc:.2f}' % (args.exp_name, args.exp_version),
    save_top_k=1,
    mode='max')

# set up early stopping callback
early_stop_callback = EarlyStopping(monitor="val_acc", min_delta=0.00, patience=5, verbose=False, mode="max")

# set up trainer
trainer = pl.Trainer(
    logger=TensorBoardLogger("./tblogs2", name=args.exp_name, version=args.exp_version),
    accelerator="gpu",
    precision=16,
    max_epochs=args.epochs,
    callbacks=[checkpoint_callback, early_stop_callback],
)


testset_paths = []
# check if multiple testsets are used
# add testsets to testset_paths if args.test is a directory
if os.path.isdir(args.test):
        print("Using multiple testsets from: %s" % (args.test, ))
        # get all dataset file from directory
        testset_paths = util.get_all_file_paths(args.test, ext=".csv")
        # initialize testset with first file in directory (nor used but needed for datamodule)
        args.test = testset_paths[0]
# leave args.test if it is a file
elif os.path.isfile(args.test):
    pass


mod = ErrorDetectionModel(args)
dx = PrintingErrorDatamodule(args)
# check if model should be trained or not (see if checkpoint is given)
# train model
if not args.test_ckpt:
    trainer.fit(mod, datamodule=dx)
    best_model_path = trainer.checkpoint_callback.best_model_path
    print(best_model_path)

    # test model
    # check if multiple testsets are used
    if not testset_paths:
        trainer.test(model=mod, datamodule=dx)
    else:
        test_loaders = []
        for i, testset in enumerate(testset_paths):
            testset_name = os.path.basename(testset)
            print("Test dataloader %i: %s" % (i, testset_name))
            test_loaders.append(dx.make_dataloader(testset))

        trainer.test(dataloaders=test_loaders)

# only test model
else:
    trainer.test(model=mod, datamodule=dx, ckpt_path=args.test_ckpt)