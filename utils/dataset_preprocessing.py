import pandas as pd
import os
import sorting_utility as util
import sklearn.model_selection as skl
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score


def concat_dataframes_classes(CSV_PATHS):
    df = pd.DataFrame()
    for file in CSV_PATHS:
        new_df = pd.read_csv(file, delimiter=";")
        df = pd.concat([df, new_df])        
    df.to_csv("%s/%s.csv" %(SCRIPT_DIR, "train_all_metadata"), index=False, sep=";")

def split_dataframe_with_class_balance(df, percent, column_name):
    y = df[column_name]
    sss = skl.StratifiedShuffleSplit(n_splits=1, test_size=percent, random_state=0)
    for train_index, test_index in sss.split(df, y):
        train_df, test_df = df.iloc[train_index], df.iloc[test_index]
    return train_df, test_df

# DROPS SPAGHETTI CLASS
def make_class_to_number(df):
    #df = df.loc[:, ['image', 'class']]
    #drop all rows with class = "SPAGHETTI"
    df = df[df["class"] != "SPAGHETTI"]
    df.loc[df["class"] == "GOOD", "class"] = 0
    df.loc[df["class"] == "MIN_IMPERFECTION", "class"] = 0
    df.loc[df["class"] == "UNDEREXTRUSION", "class"] = 1
    df.loc[df["class"] == "UNDEREXTRUSION_subtle", "class"] = 1
    df.loc[df["class"] == "STRINGING", "class"] = 2
    df.loc[df["class"] == "STRINING_subtle", "class"] = 2
    return df

# check for image class type

def class_to_number_csv(train_runs=None):
    if not train_runs:
        train_runs = ["train_hand_select", "train_hand_select_nofirst", "train_hand_select_noreoccurence", "train_hand_select_layerfiltered_3",  ]

    for train_set in train_runs:
        df = pd.read_csv("%s/%s.csv" %(DATASET_DIR, train_set), delimiter=";")
        df = df[df["class"] != "SPAGHETTI"]
        df.loc[df["class"] == "GOOD", "class"] = 0
        df.loc[df["class"] == "MIN_IMPERFECTION", "class"] = 0
        df.loc[df["class"] == "UNDEREXTRUSION", "class"] = 1
        df.loc[df["class"] == "UNDEREXTRUSION_subtle", "class"] = 1
        df.loc[df["class"] == "STRINGING", "class"] = 2
        df.loc[df["class"] == "STRINING_subtle", "class"] = 2
        df.to_csv("%s/%s.csv" %(DATASET_DIR, train_set), index=False, sep=";")

def subtract_df(df1, df2):
    df = df1[~df1.image.isin(df2.image)]   
    
    return df

#select all rows where layer is greater than 8
def select_layer(df, layer):
    df = df[df["layer"] > layer]
    return df

def select_recording(source_df, recordings=None):
    if not recordings:
        recordings = ["Recording_06122022_19_52_40", "Recording_11012023_10_32_58", "Recording_22122022_12_19_15", "Recording_06012023_12_17_19", "Recording_10012023_19_39_45"]
    
    new_df = pd.DataFrame()
    for rev in recordings:
        new_df = pd.concat([new_df, df.loc[df['recording'] == rev]])

# remove the duplicate layers in every recording
def remove_duplicate_layers(df):
    recording_grouped = df.groupby("recording")
    for recording, recording_df in recording_grouped:
        layer_grouped = recording_df.groupby("layer")
        for layer, layer_df in layer_grouped:
            if len(layer_df) > 1:
                df = df.drop(layer_df.index[1:])
    return df

def filter_nth_entries(df, n):
    """
    Returns a DataFrame with only the nth entries of a specified column.
    
    Parameters:
        - df: A pandas DataFrame.
        - n: An integer specifying the nth entry to filter.
        - column: A string specifying the name of the column to filter.
    
    Returns:
        - A pandas DataFrame with only the nth entries of the specified column.
    """
    # Get the length of the DataFrame
    length = len(df)
    
    # Create a list of boolean values indicating which entries to keep
    keep = [True if i % n == (n - 1) else False for i in range(length)]
    
    # Return the filtered DataFrame
    return df.loc[keep]

def select_rows_by_column_values(df, column, values):
    return df[df[column].isin(values)]

## to check for reoccurence
def get_common_values(df1, df2, col):
    return set(df1[col]).intersection(df2[col])

def generate_less_layers_experiment(kfold_data_dir, save_dir ):
    # read csv files in directory
    data = util.get_all_file_paths(kfold_data_dir, "csv")
    # iterate over csv files
    
    for data_fold in data:
        # load fold
        df = util.open_csv_file(data_fold)
        # generate name and filepath
        new_filename = util.get_filename_from_path(data_fold).replace("_", "_no_duplicates_")
        no_duplicates_dir = "%s/no_duplicates" %save_dir
        no_duplicates_path = "%s/no_duplicates/%s" %(save_dir, new_filename)
        if not os.path.exists(no_duplicates_dir):
            os.makedirs(no_duplicates_dir)

        # filter duplicate layers
        df_no_duplicates = remove_duplicate_layers(df)        
        print("Generating: %s" %no_duplicates_path)
        print(len(df_no_duplicates))
        # save dataframe
        util.dump_csv(df_no_duplicates, no_duplicates_path)
        
        keep = [1, 2, 3, 4, 5]
        for n in keep:
            # filter, keep every nth entry
            df_keep_n = filter_nth_entries(df_no_duplicates, n, "layer")
            # generate name and filepath
            new_filename = util.get_filename_from_path(data_fold).replace("_", "_keep_%s_" %n)
            keep_n_dir = "%s/keep_%s" %(save_dir, n)
            keep_n_path = "%s/keep_%s/%s" %(save_dir, n, new_filename)
            if not os.path.exists(keep_n_dir):
                os.makedirs(keep_n_dir)
            print("Generating: %s" %keep_n_path)
            print(len(df_keep_n))
            # dave dataframe
            util.dump_csv(df_keep_n, keep_n_path)

def make_k_splits( csv_path, output_dir, splits=5, train_name="train", val_name="val"):
        dataframe = util.open_csv_file(csv_path)        
        
        print("MAKE KFOLD SPLIT")
        skf = StratifiedKFold(n_splits=splits,
                              random_state=None, shuffle=False)
        
        for i,  (train_index, val_index) in enumerate(skf.split(dataframe["image"].values, dataframe["class"].values)):
            #xtrain, xval = dataframe["image"].values[train_index], dataframe["image"].values[val_index]
            #ytrain, yval = dataframe["class"].values[train_index], dataframe["class"].values[val_index]

            #get dataframe split by index
            train_df = dataframe.iloc[train_index]
            val_df = dataframe.iloc[val_index]

            
            
            train_path = "%s/%s_fold%i.csv" %(output_dir, train_name, i)
            val_path = "%s/%s_fold%i.csv" %(output_dir, val_name, i)

            print("Generating: %s" %train_path)

            util.dump_csv(train_df, train_path)
            util.dump_csv(val_df, val_path)


# split train test k-fold split further into k-fold train and val split each
def make_train_val_double_split(csv_dir, output_dir, splits=5):
    csv_files = util.get_all_file_paths(csv_dir, "csv")
    train_files = [x for x in csv_files if "train" in os.path.basename(x)]

    for train_file in train_files:

        train_name = os.path.basename(train_file).replace(".csv", "")
        val_name = train_name.replace("train", "val")

        make_k_splits(train_file, output_dir, splits=splits, train_name=train_name, val_name=val_name)



if __name__ == "__main__":
    SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
    
    
    