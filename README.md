# FDM_error_detection

Is an experimental setup to train machine learing models to detect printng errors in FDM 3D printed parts.
The repository consists of a model, wich can be configured to fit different network architectures (ResNet, base_cnn)

An implementation of differend experiments is prepared in the *RUN_EXPERIMENTS* scripts

## Training the model

First extract the PRINTING_ERRORS.zip dataset in the /PRINTING_ERRORS directory

To train the model use the *start_model* script:

    # Example for training an ResNet18 model:
    python3 start_model.py --exp_name Test --exp_version 1 --aug vertical_flip --train PRINTING_ERRORS/general_data/black_bed_train.csv --test PRINTING_ERRORS/general_data/black_bed_test10%.csv --im_path PRINTING_ERRORS/images/all_images256



The parameter for training the model are as followed:

    #Parameter
    --net           Choice of network architecture: Coose from "ResNet" and "baseCNN"
    --net_version   Choose version for the ResNet: 18, 34, 50, 101, 152
    --lr            Learning rate
    --epochs        Maximum number of training epochs
    --image_size    Image resolution of a square image
    --classes       Number of classes in the dataset
    --exp_name      Namer for the experiment
    --exp_version   Verson of the experiment
    --aug           Augmentation strategy: Choose from "no_aug", "vertical_flip", "horizontal_flip", "image_aug", "geometric_aug" and "strong_aug"
    --batch_size    Training batch size
    --train         Path to the train dataset labels in csv-format
    --val           Optional: Path to the validation labels in csv-format, will be split randomly from train at 20% if not specified
    --test          Path to test labels or directory of test label files in csv-format
    --im_path       Directory of the image data
    --test_ckpt     Filepath of a checkpoint if used for testing
    

The implementation uses the tensorboard logger that will save the log files in the /tblogs folder.
Model checkpoints are stored in the /models folder.

## Experiments

The experiments that were carried out on the model are scripted in the *RUN_EXPERIMENTS* script.
The model is trained with different augmentation strategies, and data splits.

The processed results of the experiments are stored in the /Results folder.

  
