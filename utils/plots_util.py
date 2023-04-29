import os
import matplotlib.pyplot as plt
import tensorboard as tb
import numpy as np
import pandas as pd
from tbparse import SummaryReader
import sorting_utility as util
import statistics
import yaml
import copy
import operator

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))

DATASET_NAME = "PRINTING_ERRORS"
PATH = "%s/PRINTING_ERRORS/train_images256/" % (SCRIPT_DIR)
TRAIN_CSV = "%s/%s/kfold_data/train_fold0.csv" % (SCRIPT_DIR, DATASET_NAME)
TEST_CSV = "%s/%s/kfold_data/test_fold0.csv" % (SCRIPT_DIR, DATASET_NAME)
MODELS_DIR = "%s/models" % (SCRIPT_DIR)
TBLOGS_DIR = "%s/tblogs" % (SCRIPT_DIR)
DROPFILE = "%s/dropfile.csv" % (SCRIPT_DIR)
#SAVE_LOG_DIR = "/mnt/c/Users/Nils/Desktop/DATASET/log_saves/new_aug_run"
BOXPLOT_SAVE_DIR = "%s/Bilder/Boxplots/new_aug_run" % (SCRIPT_DIR)

#reader = SummaryReader("/home/ws/urdrf/DATASET/tblogs/resnet_18_no-transfer_no-aug/fold2", pivot=True)
#df = reader.scalars
#print(df)
#util.dump_csv(df, DROPFILE)


def mean_and_std(data):
    mean = statistics.mean(data)
    std = statistics.stdev(data)
    return mean, std

def read_tblog(path):
    # load log data from tblogs
    reader = SummaryReader(path, pivot=True)
    df = reader.scalars
    return df

def read_tblog_into_extracted_values_old(path):
    extracted_values = {"test_acc": [], "recall": [], "precision": []}
    # load log data from tblogs
    reader = SummaryReader(path, pivot=True)
    df = reader.scalars
    print(df)
    for key in extracted_values.keys():
        if key in df.columns:
            # select column
            log_value = df.loc[~df[key].isna()][key].iloc[0]
            # append to results
            extracted_values[key].append(float(log_value))
        else:
            print("Key %s not found in tblog" % (key))
    return extracted_values

# This function extracts the log values defined in the log_values dict from the tblogs and bundles all folds
def extract_kfold_tblogs(log_dir, save=False):
    folds = ["fold0", "fold1", "fold2", "fold3", "fold4"]
    folders = util.get_all_folder_paths(log_dir)
    results = {}
    extracted_values = {"test_acc": [], "recall": [], "precision": []}

    for folder in folders:
        # generate folder name as tag
        folder_name = folder.split("/")[-1]
        # initialize results dict
        results[folder_name] = copy.deepcopy(extracted_values)
        # iterate over folds
        for fold in folds:
            path = "%s/%s" % (folder, fold)
            # load log data from tblogs
            reader = SummaryReader(path, pivot=True)
            df = reader.scalars
            for key in results[folder_name].keys():
                if key in df.columns:
                    # select column
                    log_value = df.loc[~df[key].isna()][key].iloc[0]
                    # append to results
                    results[folder_name][key].append(float(log_value))
                else:
                    print("Key %s not found in tblog" % (key))

    # save results to yaml
    if save:
        util.dump_yaml(results, "%s/results.yaml" % (SAVE_LOG_DIR))
    return(results)

def candlestick(values, labels, x=None, y=None, Title="LALA", y_label="DADA", savename="candlestick"):
    
    # Create a figure and axes object
    fig, ax = plt.subplots()
    positions = range(len(values))

    # Create the boxplot
    bp = ax.boxplot(values, positions=range(len(values)), widths=0.5,
                    patch_artist=True,
                    showcaps=True, whiskerprops={'linestyle': '--'},
                    boxprops={'facecolor': 'red', 'alpha': 0.7},
                    medianprops={'color': 'blue'})
    # Add the whiskers
    #for i in range(len(mean_values)):
    #    ax.plot([i, i], [lower_limits[i], upper_limits[i]], color='black')

    if x and y:
        ax_line_lin = ax.twinx()
        ax_line_lin.plot(x, y, label='Metric 1')
        ax_line_lin.set_ylabel('Mean Sample Size')

    # Set the x-axis labels
    ax.set_xticks(range(len(values)))
    #ax.set_xticklabels(labels, rotation=20)
    ax.set_xticklabels(labels)

    # Add a title and y-axis label
    ax.set_title(Title)
    ax.set_ylabel(y_label,)  
    # save the plot
    if savename:
        plt.savefig("%s/%s.svg" % (BOXPLOT_SAVE_DIR, savename), format="svg")

    # Show the plot
    plt.show()

def calc_matrix_mean(dataframes):
    # concatenate the input dataframes along the rows
    df_concat = pd.concat(dataframes)
    # group the concatenated dataframe by column names and calculate the mean
    df_mean = df_concat.groupby(level=0).mean()
    return df_mean

def generate_conf_matrix_over_folds(folds_dir, savename="conf_matrix"):
    files = util.get_all_file_paths_deep(folds_dir, ".csv")
    folder_name = folds_dir.split("/")[-1]
    image_path = "%s/%s_%s.png" % (folds_dir, savename, folder_name)
    matrices = []
    for file in files:
        matrix = util.open_csv_file(file)
        #print(matrix)
        #plot_confusion_matrix(matrix)
        matrices.append(matrix)

    mean_matrix = calc_matrix_mean(matrices)
    print(mean_matrix)
    plot_confusion_matrix(mean_matrix, image_path=image_path, save=True)

def generate_conf_matix_over_experiment(exp_dir, savename="conf_matrix"):
    folders = util.get_all_folder_paths(exp_dir)
    for folder in folders:
        generate_conf_matrix_over_folds(folder, savename=savename)

def plot_confusion_matrix(file, image_path=None, save=False, show=True):
    print("Plotting confusion matrix...")
    # check if file is csv path or datafame
    # in case of dataframe a image save path is required
    if type(file)==str:
        image_path = file.replace(".csv", ".png")
        df = util.open_csv_file(file)
    elif image_path and type(file)==pd.DataFrame:
        df = file
        df = df.astype(int)

    # calculate in percent
    df = df.div(df.sum(axis=1), axis=0)
    df = df.round(2)
    

    print("Extracting confusion matrix values...")
    # Extract confusion matrix values
    classes = df.index.values
    cm_values = df.values

    # Create figure and axis objects
    fig, ax = plt.subplots()

    print("Plotting confusion matrix...")
    # Plot confusion matrix
    im = ax.imshow(cm_values, interpolation='nearest', cmap=plt.cm.Blues)
    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))
    #ax.set_xticklabels(classes, rotation=45)
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')
    ax.set_title('Confusion Matrix')

    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)

    print("Adding text to confusion matrix...")
    # Display values in each cell
    for i in range(len(classes)):
        for j in range(len(classes)):
            text = ax.text(j, i, cm_values[i, j],
                           ha='center', va='center', color='black')
    print("Saving confusion matrix to %s" %image_path)

    if save:
        plt.savefig(image_path)
    if show:
        plt.show()

# use this to extract the values from the tb log files into csv tables
def create_table(input_dir, output_dir, columns, translate_dic = None, level=2,):
    #level 1 if input_dir is for one run
    #level 2 if input_dir conatins multiple runs
    if level == 1:
        dirs = [input_dir, ]
    if level == 2:
        dirs = util.get_all_folder_paths(input_dir)
    
    # add column for folder name at the beginning
    columns.insert(0, "folder_name")
    for dir in dirs:
        print("Scanning: %s" %dir)
        # create dict with columns as keys
        result_dict = {column: None for column in columns}
        # create dataframe with columns
        df = pd.DataFrame(columns=columns)
        # get all folders in input dir
        folders = util.get_all_folder_paths(dir)
        for folder in folders:
            folder_name = folder.split("/")[-1]
            results = read_tblog(folder)
            # add row to dataframe with folder name in first column
            for key in result_dict.keys():
                if key == "folder_name":
                    result_dict[key] = folder_name
                else:
                    # read value from results dataframe, key is column name (use the highest value)
                    # in the tb logs the test case is only one value, so the max is the only value
                    result_dict[key] = results[key].max()

            #df = df.append(result_dict, ignore_index=True)
            df = pd.concat([df, pd.DataFrame(result_dict, index=[0])], ignore_index=True, axis=0, join='outer')

        print(df)
        
        # change column names according to translate_dic
        if translate_dic:
            df.rename(columns=translate_dic, inplace=True)

        # save dataframe as csv
        if output_dir:
            # generate filenames and pahts
            experiment_name = input_dir.split("/")[-1]
            out_folder = "%s/%s" % (output_dir, experiment_name)
            filename = dir.split("/")[-1]            
            out_path = "%s/%s.csv" % (out_folder, filename)
            # create output folder if not exists
            if not os.path.exists(out_folder):
                os.makedirs(out_folder)
            print("Saving to: %s" % out_path)
            # save dataframe as csv
            util.dump_csv(df, out_path)

def export_tables_for_experiments():
    DIR= "/mnt/c/Users/Nils/Desktop/DATASET/log_saves/random_split_aug_run"
    RESULTS_DIR = "/mnt/c/Users/Nils/Desktop/DATASET/Results"
    '''
    column_keys = [ "precision_tl(0)",
                    "precision_tl(1)",
                    "precision_tl(2)",
                    "recall_tl(0)",
                    "recall_tl(1)",
                    "recall_tl(2)",
                    "test_acc_1/dataloader_idx_0",
                    "test_acc_1/dataloader_idx_1",
                    "test_acc_1/dataloader_idx_2",]
    '''
    column_keys = [ "precision_tl(0)",
                    "recall_tl(0)",
                    "test_acc_1"]    
    # open yaml to dict
    translate_keys = util.open_yaml("/mnt/c/Users/Nils/Desktop/DATASET/translate_column_names.yaml")

    create_table(DIR, RESULTS_DIR, column_keys, translate_keys )

def merge_data_tables(dir, file_identifiers):
    files = util.get_all_file_paths(dir, ".csv")
    #iteration = ["geometric_aug", "heavy_aug", "no_aug", "image_aug", "vertical_flip", "horizontal_flip"]
    
    for ident in file_identifiers:
        df = pd.DataFrame()
        # add column named "randomsplit" with value i
        iterfiles = [file for file in files if ident in file]
        for i, file in enumerate(iterfiles):
            new_df = util.open_csv_file(file)
            # add column to new_df
            new_df["randomsplit"] = i

            df = pd.concat([df, new_df], ignore_index=True, axis=0, join='outer')

        #print(df)
        merge_name = "%s.csv" % ident
        merge_path = "%s/%s" % (dir, merge_name)
        util.dump_csv(df, merge_path)

def make_aug_boxplots():
    files = ["/mnt/c/Users/Nils/Desktop/DATASET/Results/new_double_kfold_aug_run_normalized/resnet_18_no-transfer_no_aug-_double_kfold_normalized.csv",
             "/mnt/c/Users/Nils/Desktop/DATASET/Results/new_double_kfold_aug_run_normalized/resnet_18_no-transfer_vertical_flip-_double_kfold_normalized.csv", 
             "/mnt/c/Users/Nils/Desktop/DATASET/Results/new_double_kfold_aug_run_normalized/resnet_18_no-transfer_horizontal_flip-_double_kfold_normalized.csv",
             "/mnt/c/Users/Nils/Desktop/DATASET/Results/new_double_kfold_aug_run_normalized/resnet_18_no-transfer_image_aug-_double_kfold_normalized.csv",
             "/mnt/c/Users/Nils/Desktop/DATASET/Results/new_double_kfold_aug_run_normalized/resnet_18_no-transfer_geometric_aug-_double_kfold_normalized.csv",
             "/mnt/c/Users/Nils/Desktop/DATASET/Results/new_double_kfold_aug_run_normalized/resnet_18_no-transfer_heavy_aug-_double_kfold_normalized.csv"]


    values = []
    labels = ["no aug", "vertical\nfilp", "horizontal\nflip", "image\naug", "geometric\naug", "heavy\naug"]
    for file in files:
        df = util.open_csv_file(file)
        values.append(df["recall_tl(0)"])
        #labels.append(file.split("/")[-1].split(".")[0])
    #x = [0, 1, 2, 3, 4]
    #y = [4468, 2233, 1489, 1108, 893]
    candlestick(values, labels,  Title="", y_label="Recall", savename="recall_new_aug")

def candle_lineplot(x_, mean_, std_, savename=None):
    # Generate some sample data
    fig, ax = plt.subplots()
    colours = ["red", "blue", "green", "yellow", "orange"]
    splits = ["Split 1", "Split 2", "Split 3", "Split 4", "Split 5"]
    for index in range(len(x_)):
        x= x_
        mean = mean_[index]
        std = std_[index]
        factor = 0.5
        colour = colours[index]
        # Calculate the upper and lower values of the candlesticks
        upper = [mean[i] + std[i]*factor for i in range(len(mean))]
        lower = [mean[i] - std[i]*factor for i in range(len(mean))]

        # Create the plot
        print(upper)
        print(lower)
        print(x)

        
        ax.plot(x, mean, '-', label=splits[index], color=colour)
        ax.vlines(x, lower, upper, color=colour, alpha=0.5)
        ax.set_xticklabels(x)

        # Add labels and legend
        #ax.set_xlabel('X')
        ax.set_ylabel("Test Accuracy")
        #ax.set_title('Line Plot with Candlestick Markers')
        ax.legend()

    #plt.show()
    if savename:
        # save as svg
        file_path = "%s/%s" % ("/mnt/c/Users/Nils/Desktop/DATASET/Bilder", savename)
        plt.savefig(file_path, format="svg")
    plt.show()

def make_keep_plot():

    files = ["/mnt/c/Users/Nils/Desktop/DATASET/Results/less_layer_run_multirun(no_transfer, no_aug)/keep_1.csv",
             "/mnt/c/Users/Nils/Desktop/DATASET/Results/less_layer_run_multirun(no_transfer, no_aug)/keep_2.csv", 
             "/mnt/c/Users/Nils/Desktop/DATASET/Results/less_layer_run_multirun(no_transfer, no_aug)/keep_3.csv",
             "/mnt/c/Users/Nils/Desktop/DATASET/Results/less_layer_run_multirun(no_transfer, no_aug)/keep_4.csv",
             "/mnt/c/Users/Nils/Desktop/DATASET/Results/less_layer_run_multirun(no_transfer, no_aug)/keep_5.csv",]
    


    values = []
    labels = ["n=1", "n=2", "n=3", "n=4", "n=5", ]
    folds = ["fold0", "fold1", "fold2", "fold3", "fold4",]
    means = []
    stds = []
    sup_means = []
    sup_stds = []
    for file in files:
        df = util.open_csv_file(file)
        
        for fold in folds:
            # get data for fold
            fold_data = df[df["folder_name"] == fold]
            
            #mean_fold_data = fold_data.mean()
            mean, std = mean_and_std(fold_data["test_acc"])
            means.append(mean)
            stds.append(std)
        # values.append(mean_fold_data["test_acc"])

        sup_means.append(means)
        sup_stds.append(stds)
        means = []
        stds = []


    sup_means_t = list(zip(*sup_means))
    sup_stds_t = list(zip(*sup_stds))

    x = ["n=1", "n=2", "n=3", "n=4", "n=5"]
    candle_lineplot(x, sup_means_t, sup_stds_t, savename="keep_test_fold_performance.svg")

def scatterplot(x, y, savename=None):
    symbols = ["o", "^", "s", "x", "d"]
    folds = ["Split 1", "Split 2", "Split 3", "Split 4", "Split 5"]
    colors = ["red", "blue", "green", "brown", "orange"]

    for index in range(len(x)):
        symbol = symbols[index]
        fold = folds[index]
        y_values = y[index]
        color = colors[index]
        
        print(x)
        print(y_values)
        
        
        plt.scatter(folds, y_values, marker=symbol, color=color, label=fold)


    plt.legend()
    #plt.show()

    if savename:
        # save as svg
        file_path = "%s/%s" % ("/mnt/c/Users/Nils/Desktop/DATASET/Bilder", savename)
        plt.savefig(file_path, format="svg")

    plt.show()
        

if __name__ == "__main__":

    SAVE_LOG_DIR = "/mnt/c/Users/Nils/Desktop/DATASET/log_saves/new_aug_run"
    RESULTS_DIR = "/mnt/c/Users/Nils/Desktop/DATASET/Results/double_kfold_aug_run"

    DIR = "/mnt/c/Users/Nils/Desktop/DATASET/tblogs"
    OUT = "/mnt/c/Users/Nils/Desktop/DATASET/Results/new_double_kfold_aug_run_normalized"
    columns = ["test_acc_1", "precision_tl(0)", "recall_tl(0)"]

    make_aug_boxplots()

    #create_table(DIR, OUT, columns)

    #print(read_tblog("/mnt/c/Users/Nils/Desktop/DATASET/tblogs/resnet_18_no-transfer_geometric_aug-_double_kfold_normalized/fold0_fold0"))