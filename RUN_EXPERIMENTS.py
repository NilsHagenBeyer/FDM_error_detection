import nets.resnet_lightning as resnet_lightning
import os
import subprocess
import utils.sorting_utility as util

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
# set some default pahts
DATASET_NAME = "PRINTING_ERRORS"
IMAGE_PATH = "%s/%s/images/all_images256/" % (SCRIPT_DIR, DATASET_NAME)
KFOLD_DATA = "%s/%s/kfold_data/Geometry_Split_k-fold_train_val_splits" % (SCRIPT_DIR, DATASET_NAME)  # <---- configure here
LESS_LAYERS_DIR = "%s/%s/images/kfold_data/layer_filtered" % (SCRIPT_DIR, DATASET_NAME)

################# start model helper functions ################

# test a testset from a model checkpoint
def test_model(test_data, test_ckpt, image_dir):
    # load test and train set
    # parse arguments
    args = ["python3", 
            "start_model.py",
            "--net_version", "18",
            "--use_transfer", "True",
            "--tune_fc_only", "True",
            "--aug", "no",
            "--exp_name", "onlyTest",
            "--exp_version", "onlyTest_v0",
            "--im_path", image_dir,
            "--test_ckpt", test_ckpt,
            "--test", test_data,
            ]
    # run model with parameters
    subprocess.call(args)

# train model with: train test kfold split, train val stratified random split
def __run_kfold_randomsplit__(kfold_data_dir, 
        net = "ResNet",
        folds = ["fold0", "fold1", "fold2", "fold3", "fold4"],
        augs = ["no_aug"],
        net_versions = [18],
        transfer_learning = [(True, True),],
        tag = ""
    ):

    csv_files = util.get_all_file_paths(kfold_data_dir, ".csv")

    # iterate over all combinations
    for net_version in net_versions:
        for use_transfer, tune_fc_only in transfer_learning:
            for aug in augs:
                for fold in folds:
                    # load test and train set
                    #train_path = "%s/train_%s.csv" % (kfold_data_dir, fold)
                    #test_path = "%s/test_%s.csv" % (kfold_data_dir, fold)

                    test_path = [csv for csv in csv_files if fold in util.get_filename_from_path(csv) and "test" in util.get_filename_from_path(csv)][0]
                    train_path = [csv for csv in csv_files if fold in util.get_filename_from_path(csv) and "train" in util.get_filename_from_path(csv)][0]
                    
                    print(train_path)
                    print(test_path)

                    # generate experiment names
                    trans = "transfer" if use_transfer else "no-transfer"
                    experiment_name = "resnet_%s_%s_%s-%s" % (net_version, trans, aug, tag)
                    experiment_version = fold
                    # parse arguments
                    args = ["python3", 
                            "start_model.py",
                            "--net", net,
                            "--net_version", str(net_version),
                            "--use_transfer", str(use_transfer),
                            "--tune_fc_only", str(tune_fc_only),
                            "--aug", aug,
                            "--exp_name", experiment_name,
                            "--exp_version", experiment_version,
                            "--train", train_path,
                            "--test", test_path,
                            "--im_path", IMAGE_PATH
                            ]
                    # run model with parameters
                    subprocess.call(args)

# train model with: train, val kfold split
def __run_kfold__(kfold_data_dir, 
        net = "ResNet",
        folds = ["fold0", "fold1", "fold2", "fold3", "fold4"],
        augs = ["no_aug"],
        net_versions = [18],
        transfer_learning = [(True, True),],
        tag = "",
        test_dir = "",
        workers=24
    ):

    csv_files = util.get_all_file_paths(kfold_data_dir, ".csv")

    # iterate over all combinations
    for net_version in net_versions:
        for use_transfer, tune_fc_only in transfer_learning:
            for aug in augs:
                for fold in folds:

                    # load train and validation set paths
                    val_path = [csv for csv in csv_files if fold in util.get_filename_from_path(csv) and "val" in util.get_filename_from_path(csv)][0]
                    train_path = [csv for csv in csv_files if fold in util.get_filename_from_path(csv) and "train" in util.get_filename_from_path(csv)][0]

                    # generate experiment names
                    trans = "transfer" if use_transfer else "no-transfer"
                    experiment_name = "resnet_%s_%s_%s-%s" % (net_version, trans, aug, tag)
                    experiment_version = fold
                    # parse arguments
                    args = ["python3", 
                            "start_model.py",
                            "--net", net,
                            "--net_version", str(net_version),
                            "--use_transfer", str(use_transfer),
                            "--tune_fc_only", str(tune_fc_only),
                            "--aug", aug,
                            "--exp_name", experiment_name,
                            "--exp_version", experiment_version,
                            "--train", train_path,
                            "--test", test_dir,
                            "--val", val_path,
                            "--im_path", IMAGE_PATH,
                            "--workers", str(workers),
                            ]
                    # run model with parameters
                    subprocess.call(args)

# train model with: double train test val kfold split (25 folds)
def __run_double_kfold__(kfold_data_dir, 
        net = "ResNet",
        test_folds = ["fold0", "fold1", "fold2", "fold3", "fold4"],
        train_folds = ["fold0", "fold1", "fold2", "fold3", "fold4"],
        augs = ["no_aug"],
        net_versions = [18],
        transfer_learning = [(True, True),],
        tag = "double_kfold",
        workers=16
    ):

    csv_files = util.get_all_file_paths(kfold_data_dir, ".csv")

    # iterate over all combinations
    for net_version in net_versions:
        for use_transfer, tune_fc_only in transfer_learning:
            for aug in augs:
                for test_fold in test_folds:
                    for train_fold in train_folds:
                        # load test and train set
                        #train_path = "%s/train_%s.csv" % (kfold_data_dir, fold)
                        #test_path = "%s/test_%s.csv" % (kfold_data_dir, fold)

                        #val_path = [csv for csv in csv_files if fold in util.get_filename_from_path(csv) and "val" in util.get_filename_from_path(csv)][0]
                        #train_path = [csv for csv in csv_files if fold in util.get_filename_from_path(csv) and "train" in util.get_filename_from_path(csv)][0]

                        train_path = "%s/train_%s_%s.csv" % (kfold_data_dir, test_fold, train_fold)
                        val_path = "%s/val_%s_%s.csv" % (kfold_data_dir, test_fold, train_fold)
                        test_path = "%s/test_%s.csv" % (kfold_data_dir, test_fold)


                        # generate experiment names
                        trans = "transfer" if use_transfer else "no-transfer"
                        experiment_name = "resnet_%s_%s_%s-%s" % (net_version, trans, aug, tag)
                        experiment_version = "%s_%s" % (test_fold, train_fold)
                        # parse arguments
                        args = ["python3", 
                                "start_model.py",
                                "--net", net,
                                "--net_version", str(net_version),
                                "--use_transfer", str(use_transfer),
                                "--tune_fc_only", str(tune_fc_only),
                                "--aug", aug,
                                "--exp_name", experiment_name,
                                "--exp_version", experiment_version,
                                "--train", train_path,
                                "--test", test_path,
                                "--val", val_path,
                                "--im_path", IMAGE_PATH,
                                "--workers", str(workers),
                                ]
                        # run model with parameters
                        subprocess.call(args)

def __run__(
        net = "ResNet",
        augs = ["no_aug"],
        net_versions = [18],
        transfer_learning = [(True, True),],
        tag = "",
        experiment_version = "",
        train_path = "",
        test_dir = "",
        val_path = ""
        ):


    # iterate over all combinations
    for net_version in net_versions:
        for use_transfer, tune_fc_only in transfer_learning:
            for aug in augs:
                # generate experiment names
                trans = "transfer" if use_transfer else "no-transfer"
                experiment_name = "resnet_%s_%s_%s-%s" % (net_version, trans, aug, tag)
                # parse arguments
                args = ["python3", 
                        "start_model.py",
                        "--net", net,
                        "--net_version", str(net_version),
                        "--use_transfer", str(use_transfer),
                        "--tune_fc_only", str(tune_fc_only),
                        "--aug", aug,
                        "--exp_name", experiment_name,
                        "--exp_version", experiment_version,
                        "--train", train_path,
                        "--test", test_dir,
                        "--val", val_path,
                        "--im_path", IMAGE_PATH,
                        ]

                # run model with parameters
                subprocess.call(args)


################## EXPERIMENTS ###########################

# first overall experiment to get an overview of the possibilites
def first_searchspace_run(kfold_data_dir):
     # Experimental Setup
    kwargs ={
    "folds" : ["fold0", "fold1", "fold2", "fold3", "fold4"],
    "augs" : ["weak", "strong", "no"],
    "net_versions" : [18, 50],
    "transfer_learning" : [(True, True), (False, False)]
    }
    
    __run_kfold_randomsplit__(kfold_data_dir, **kwargs)

# compare selection for the ResNet, ResNet18 seems to be sufficient
def compare_resnets_run(kfold_data_dir):
    folds = ["fold0", "fold1", "fold2", "fold3", "fold4"]
    net_versions = [18, 34, 50]
    transfer_learning = [(True, True),]
    for net_version in net_versions:
        for use_transfer, tune_fc_only in transfer_learning:
            for fold in folds:
                # load test and train set
                    train_path = "%s/train_%s.csv" % (kfold_data_dir, fold)
                    test_path = "%s/test_%s.csv" % (kfold_data_dir, fold)
                    # generate experiment names
                    trans = "transfer" if use_transfer else "no-transfer"
                    experimenet_name = "NETCOMPARE_resnet_%s_%s" % (net_version, trans)
                    experiment_version = fold
                    # parse arguments
                    args = ["python3", 
                            "start_model.py",
                            "--net_version", str(net_version),
                            "--use_transfer", str(use_transfer),
                            "--tune_fc_only", str(tune_fc_only),
                            "--aug", "no",
                            "--exp_name", experimenet_name,
                            "--exp_version", experiment_version,
                            "--train", train_path,
                            "--test", test_path,
                            "--im_path", IMAGE_PATH
                            ]
                    # run model with parameters
                    subprocess.call(args)

# test augmentations train/test kfold geometry split, train/val kfold split
def compare_aug_run_double_kfold(kfold_data_dir):
    # Experimental Setup
    augmentations = ["no_aug", "vertical_flip", "horizontal_flip" ,"image_aug", "geometric_aug", "heavy_aug"]
    #augmentations = ["geometric_aug", "heavy_aug"]
    kwargs ={
        "augs" : augmentations,
        "net_versions" : [18],
        "transfer_learning" : [(False, False)],
        "tag" : "_double_kfold_normalized"
        }

    __run_double_kfold__(kfold_data_dir, **kwargs)

# test augmentations train/test kfold geometry split, train/val randomsplit
def compare_aug_run_multi_random_split(kfold_data_dir):
    # Experimental Setup
    for i in range(5):
        kwargs ={
            "augs" : ["no_aug", "vertical_flip", "horizontal_flip" ,"image_aug", "geometric_aug", "heavy_aug"],
            "net_versions" : [18],
            "transfer_learning" : [(False, False)],
            "tag" : "_randsplit%s" %(i, ),
            }

        __run_kfold_randomsplit__(kfold_data_dir, **kwargs)

# test layer filtered train geometr splits
def less_layers_run(experiment_dir):
    exp_paths = util.get_all_folder_paths(experiment_dir)
    
    # iterate over filtered layers
    for kfold_data_dir in exp_paths:
        print("Using k-fold data from: %s" %kfold_data_dir)
        # Train every filtered geometry split 3 times with random train/val split
        for i in range(3):
            print("====================================== RUNNING INTERATION %s ======================================" %i)
            kwargs ={
                "folds" : ["fold0", "fold1", "fold2", "fold3", "fold4"],
                "augs" : ["no_aug",],
                "net_versions" : [18],
                "transfer_learning" : [(False, False)],
                "tag" : "%s_%s" %(kfold_data_dir.split("/")[-1], i),
                }

            __run_kfold_randomsplit__(kfold_data_dir, **kwargs)

# train full dataset and test three testsets
def test_testsets():
    kfold_data_dir = "/mnt/c/Users/Nils/Desktop/DATASET/PRINTING_ERRORS/kfold_data/train_val_split"
    test_dir = "/mnt/c/Users/Nils/Desktop/DATASET/PRINTING_ERRORS/test_datasets"
    
    # Experimental Setup
    kwargs ={
        "folds" : ["fold0", "fold1", "fold2", "fold3", "fold4"],
        "augs" : ["no_aug"],
        "net_versions" : [18],
        "transfer_learning" : [(False, False)],
        "test_dir": test_dir, 
        "tag" : "testset_tests",
        "workers" : 16
        }
    
    __run_kfold__(kfold_data_dir, **kwargs)

# test
def test_testsets_2():
    train = "/mnt/c/Users/Nils/Desktop/DATASET/PRINTING_ERRORS/general_data/black_bed_train.csv"
    test = "/mnt/c/Users/Nils/Desktop/DATASET/PRINTING_ERRORS/test_data"
    __run__(train_path=train, test_dir=test, tag="neuer_testsetrun")


if __name__ == "__main__":
    compare_aug_run_double_kfold(KFOLD_DATA)