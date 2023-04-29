import pandas as pd
import os 
import shutil
import yaml
import csv
import cv2
import numpy as np
import yaml
# get the directory of this script
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
#RAW_DATA_PATH = "%s/%s" % (SCRIPT_DIR, "RAW_DATA/EXTENSION_RAW_2")
#COLLECT_CSV_PATH = "%s/%s" % (SCRIPT_DIR, "RAW_DATA/EXTENSION_SORTED/__csv")
#SORTED_DATA_RAW = "%s/%s" % (SCRIPT_DIR, "RAW_DATA/EXTENSION_SORTED/cropped")
#CSV_LIST_PATH = "%s/%s" % (SCRIPT_DIR, "csv_file_paths.yaml")
#TEST_CSV_PATH = "%s/%s" % (SCRIPT_DIR, "Recording_01.12.2022_15_27_47.csv")

# open csv file into pandas dataframe and show it
def open_csv_file(file_path):
    return pd.read_csv(file_path, delimiter=";")

def dump_csv(df, file_name):
    df.to_csv(file_name, index=False, sep=";")

def dump_yaml(dict, file_name):
    with open(file_name, 'w') as outfile:
        yaml.dump(dict, outfile, default_flow_style=False)

def open_yaml(file_path):
    with open(file_path, "r") as f:
        data = yaml.safe_load(f)
    return data

def open_image(file_path):
    return cv2.imread(file_path)

def dump_plot(plt, file_name):
    plt.savefig(file_name)

def drop_column(df, column_name):
    df.drop(columns=[column_name,], inplace=True)

# get all folder paths in a given directory
def get_all_folder_paths(directory):
    return [os.path.join(directory, name) for name in os.listdir(directory)
            if os.path.isdir(os.path.join(directory, name))]

# get all file paths in a given directory of a given extension
def get_all_file_paths(directory, ext):
    return [os.path.join(directory, name) for name in os.listdir(directory)
            if os.path.isfile(os.path.join(directory, name)) and name.endswith(ext)]

def get_filename_from_path(path):
    return os.path.basename(path)

# find all file (csv) paths in a given directory full depth
def get_all_file_paths_deep(directory, ext):
    csv_paths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(ext):
                csv_paths.append(os.path.join(root, file))
    return csv_paths

# copy all files from a list to a given directory
def copy_files_to_directory(file_paths, directory):
    for file_path in file_paths:
        shutil.copy(file_path, directory)

# save list into a yaml file in a given directory
def save_list_to_yaml_file(file_paths, directory):
    with open("%s/%s" % (directory, "csv_file_paths.yaml"), "w") as file:
        yaml.safe_dump(file_paths, file)

# get all csv file paths in the known folder structure
def extract_csv_file_paths(RAW_DATA_PATH):
    folder_paths = get_all_folder_paths(RAW_DATA_PATH)
    csv_paths = []
    for folder_path in folder_paths:
        file_paths = get_all_file_paths(folder_path, ".csv")
        print(file_paths[0])
        csv_paths.append(file_paths[0])
    return csv_paths

# copy all files from a list to a given directory
def copy_files_to_directory(file_paths, directory):
    for file_path in file_paths:
        shutil.copy(file_path, directory)

# read list from a yaml file in a given directory
def read_list_from_yaml_file(yaml_file):
    with open(yaml_file, "r") as file:
        return yaml.safe_load(file)

# removes header instances, printed multiple times in a csv file
# this was to fix bad recorded metadata, should not be needed further
def remove_repeated_headers(file_path):
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        header = next(reader)
        data = [row for row in reader if row != header]
    with open(file_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)
        for row in data:
            writer.writerow(row)

# collects all csv files from a directory and copies them to a given goal directory
def collect_csv_files(raw_data_path, dest_scv_path):    
    # get all csv file paths from raw data path:
    csv_paths = get_all_file_paths_deep(raw_data_path, ext=".csv")
    # copy all csv files to the destination directory
    copy_files_to_directory(csv_paths, dest_scv_path)

def resize_image(image, width, height):
    # Get the dimensions of the image
    (h, w) = image.shape[:2]
    # Calculate the aspect ratio of the image
    aspect_ratio = w / float(h)
    # Calculate the new dimensions of the image
    new_width, new_height = width, height
    # Resize the image
    image = cv2.resize(image, (new_width, new_height), interpolation = cv2.INTER_LINEAR)
    # Return the resized image
    return image

def crop_image_at_coordinates(image, width, height, x, y):
    # Crop the image
    cropped_image = image[y:y+height, x:x+width]
    # Return the cropped image
    return cropped_image

def copy_folder_structure(source_dir, dest_dir):
    """
    Copies the folder structure from source directory to destination directory without copying any files.
    :param source_dir: The source directory to copy from.
    :param dest_dir: The destination directory to copy to.
    """
    # Check if the source directory exists
    if not os.path.exists(source_dir):
        print(f"The source directory '{source_dir}' does not exist.")
        return

    # Create the destination directory if it doesn't exist
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    # Walk through the directory tree
    for root, dirs, files in os.walk(source_dir):
        # Get the relative path of the current directory
        rel_path = os.path.relpath(root, source_dir)
        dest_path = os.path.join(dest_dir, rel_path)

        # Create the directory in the destination if it doesn't exist
        if not os.path.exists(dest_path):
            os.makedirs(dest_path)

    print("Folder structure copied successfully!")

# copies all images from a given source directory in full depth to a given destination directory and crop it
def crop_image_data_full_depth(source_directory, dest_directory, x, y, crop_size=512):
    # copy folder structure from source directory to destination directory
    copy_folder_structure(source_directory, dest_directory)    
    # get all image file paths from source directory:
    image_paths = get_all_file_paths_deep(source_directory, ext=".png")
    # crop and save all images to the destination directory
    for image_path in image_paths:
        image = cv2.imread(image_path)
        # crop image
        cropped_image = crop_image_at_coordinates(image, crop_size, crop_size, x, y)        
        # generate new image path, by replacing the source directory with the destination directory
        new_image_path = image_path.replace(source_directory, dest_directory)
        print("saved image to: %s" % new_image_path)
        cv2.imshow("cropped", cropped_image)
        cv2.waitKey(1)
        # save image to new path
        if not cv2.imwrite(new_image_path, cropped_image):
            raise Exception("Could not write image")

# crops all images in a source directory and saves them to a destination directory
def crop_image_data(source_directory, dest_directory, x, y, crop_size=512):
    # get all image file paths from source directory:
    image_paths = get_all_file_paths(source_directory, ext=".png")
    # crop and save all images to the destination directory
    for image_path in image_paths:
        image = cv2.imread(image_path)
        # crop image
        cropped_image = crop_image_at_coordinates(image, crop_size, crop_size, x, y)        
        # generate new image path, by replacing the source directory with the destination directory
        new_image_path = image_path.replace(source_directory, dest_directory)
        print("saved image to: %s" % new_image_path)
        cv2.imshow("cropped", cropped_image)
        cv2.waitKey(1)
        # save image to new path
        if not cv2.imwrite(new_image_path, cropped_image):
            raise Exception("Could not write image")

# get the recording name from a given image name
def get_recording_from_image(df: pd.DataFrame, from_column: str, row_value: str, output_column: str) -> str:
    """
    Given a Pandas DataFrame, two column names, and a row value, find the row which matches the row value
    in column_1 and return the corresponding value from column_2.

    Args:
        df (pd.DataFrame): The input DataFrame.
        column_1 (str): The name of the column to match the row value against.
        column_2 (str): The name of the column to retrieve the value from.
        row_value (str): The value to match against the specified column.

    Returns:
        str: The value from column_2 corresponding to the matched row.

    Raises:
        ValueError: If the specified column names are not present in the DataFrame.
    """
    if from_column not in df.columns or output_column not in df.columns:
        raise ValueError("Invalid column names specified.")
    
    row = df[df[from_column] == row_value]
    if row.empty:
        return None
    else:
        return row.iloc[0][output_column]

# change the column value of a given recording in the goal column to a specific value
def change_column_by_recording(df, recording, goal_column, new_value):
    df.loc[df['recording'] == recording, goal_column] = new_value
    return df

# change the column value of a given recording in the goal column to a specific value
def change_column_where_other_column_has_value(df, origin_column, origin_value, goal_column, new_value):
    df.loc[df[origin_column] == origin_value, goal_column] = new_value
    return df

# get all rows from a given recording
def get_all_from_column(df, column, column_value):
    return df[df[column] == column_value]

def get_all_from_column_with_class(df, column, column_value, class_name):
    return df[(df[column] == column_value) & (df['class'] == class_name)]

# get a list of all values from a given column with no reoccurring values
def get_unique_values_from_column(df, column):
    return df[column].unique()

# generate a csv file with an overview of all classes in the dataset
def create_overview_csv(df, output_path=None, all_classes=False, drop=[], differ_rec=False, calc_total=False):    
    # read all unique values from the shape and class column
    geometries = get_unique_values_from_column(df, "shape")    
    classes = get_unique_values_from_column(df, "class")    
    # create output dataframe
    header = ["shape"] + classes.tolist()
    overview = pd.DataFrame(columns=header)

    # iterate over all geometries and sum the images of each class
    for geometry in geometries:
        cls_amount = {}
        for cls in classes:

            # also save the amount of recordings for each class and geometry
            if differ_rec:
                rec_amount = len(get_unique_values_from_column(get_all_from_column_with_class(df, "shape", geometry, cls), "recording"))                
                
                cls_amount[cls] = "[%s] %s" %(rec_amount, (df.loc[df['shape'] == geometry, 'class'] == cls).sum())
            else:
                cls_amount[cls] = (df.loc[df['shape'] == geometry, 'class'] == cls).sum()

            
        '''
        # convert geometry string into int if possible
        if geometry.isdigit():
            cls_amount["shape"] = int(geometry)
        else:
            # change the "unknown" string into a number for sorting
            continue
            cls_amount["shape"] = 99999
        '''

        cls_amount["shape"] = geometry

        # add the geometry information to the overview dataframe
        overview = overview.append(cls_amount, ignore_index=True)

        #pd.concat([overview, pd.DataFrame(cls_amount, index=[0])], ignore_index=True)
        #pd.concat([overview, pd.DataFrame(cls_amount.values(), columns=overview.columns)], ignore_index=False)

    # combine the subtle classes into the main classes if preferred
    if not all_classes:
        overview["GOOD"] += overview["MIN_IMPERFECTION"]
        overview["STRINGING"] += overview["STRINING_subtle"]
        overview["UNDEREXTRUSION"] += overview["UNDEREXTRUSION_subtle"]
        # drop unused columns
        overview.drop(columns=["MIN_IMPERFECTION", "STRINING_subtle", "UNDEREXTRUSION_subtle"], inplace=True)

    # drop unwanted classes
    for cls in drop:
        overview.drop(columns=[cls], inplace=True)

    # sort the dataframe by the shapes
    overview.sort_values(by=["shape"], inplace=True, ascending=True, ignore_index=True)

    # make total sum row
    head = overview.columns.tolist()
    # dont calculate if differ_rec is True, because the total sum is not computable in this case
    if not differ_rec and calc_total:
        for cl in head:
            if cl != "shape":
                overview.loc["total", cl] = overview[cl].sum()

    print(overview)

    if output_path:
        dump_csv(overview, output_path)

    return overview

# sort dataframe indices into bins, such that the sum of column values in one bin equals aproximately the other bins
# it is a version of the first fit decreasing algorithm (FFD)
def bin_sort(df, column, bins=5, prev_binsums=None):
    # Sort the DataFrame by the values in ascending order
    df = df.sort_values(by=column, ascending=False)

    # Calculate the target sum for each bin
    target_sum = df[column].sum() // bins

    # Create an array to store the current sum of each bin
    bin_sums = [0] * bins
    bins ={i: [] for i in range(0, bins)}

    # Start with the first row in the sorted DataFrame and add it to the first bin
    #bins = [[df.index[0]]]

    # For each subsequent row in the sorted DataFrame, add it to the bin with the lowest current sum so far
    for i in range(0, len(df)):
        # Find the bin with the lowest current sum
        min_sum = min(bin_sums)
        min_bin_index = bin_sums.index(min_sum)

        # Add the row index to the bin
        bins[min_bin_index].append(df[column].index[i])
        bin_sums[min_bin_index] += df.iloc[i][column]
    
    print("Bin sums %s" %bin_sums)
    return bins

# split the dataframe stratified classes with no mixing of geometries
# geometries is equal to the overview created witout total (calc_total = False)
def stratified_geometries_kfold(geometries, df, k=5, save_dir=None):    
    ov_string = geometries.loc[geometries["STRINGING"] != 0]
    ov_underex = geometries.loc[geometries["STRINGING"] == 0]
    # indices of the geometries sorted into bins
    bins_string = bin_sort(ov_string, "STRINGING", bins=k)
    
    
    bins_undex = bin_sort(ov_underex, "UNDEREXTRUSION", bins=k)


    # initialize dictionary for split data
    splits = {s: {"train":None, "test":None} for s in range(k)}

    # merge the bins from stinging and underextrusion
    merged_bins = {i: bins_string[i]+ bins_undex[i] for i in range(0, k)}
    print(merged_bins)

    for i in range(k):
        # split into test and train on geometry index level
        train_geom = []
        test_geom = merged_bins[i]
        for j in range(k):
            if j != i:
                train_geom += merged_bins[j]    

        print("train_geom: %s" % train_geom)
        print("test_geom: %s" % test_geom)

        # get the geometries from the indices 
        train_geometries = [geometries.iloc[g]["shape"] for g in train_geom]
        test_geometries = [geometries.iloc[g]["shape"] for g in test_geom]

        # select the datapoints in the dataset wich match the geometries
        train_data = df.loc[df["shape"].isin(train_geometries)]
        test_data = df.loc[df["shape"].isin(test_geometries)]

        # save the splits in the output dict
        splits[i]["train"] = train_data
        splits[i]["test"] = test_data

        # save the splits in ouput files 
        if save_dir:
            dump_csv(train_data, "%s/train_fold%s.csv" % (save_dir, i))
            dump_csv(test_data, "%s/test_fold%s.csv" % (save_dir, i))

    return splits

def create_kfold_overviews(kfold_dir):
    # get all dataset files
    files = get_all_file_paths(kfold_dir, "csv")
    # create new directory
    overview_folder = "%s/%s" %(kfold_dir, "overviews")
    if not os.path.exists(overview_folder):
        os.makedirs(overview_folder)

    for file in files:
        df = open_csv_file(file)
        filename = "%s%s" %(file.split("/")[-1].split(".")[0] ,"_overview.csv")
        filepath = "%s/%s" %(overview_folder, filename)
        overview = create_overview_csv(df, output_path=filepath, all_classes=True, calc_total=True)

# converts the class strings to numbers for training
def class_to_number_csv(paths):
    for path in paths:
        df = pd.read_csv(path, delimiter=";")
        #df = df[df["class"] != "SPAGHETTI"]
        df.loc[df["class"] == "SPAGHETTI", "class"] = 4
        df.loc[df["class"] == "GOOD", "class"] = 0
        df.loc[df["class"] == "MIN_IMPERFECTION", "class"] = 0
        df.loc[df["class"] == "UNDEREXTRUSION", "class"] = 1
        df.loc[df["class"] == "UNDEREXTRUSION_subtle", "class"] = 1
        df.loc[df["class"] == "STRINGING", "class"] = 2
        df.loc[df["class"] == "STRINING_subtle", "class"] = 2
        df.to_csv(path, index=False, sep=";")

if __name__ == "__main__":
    
    new_df = class_to_number_csv(["/mnt/c/Users/Nils/Desktop/DATASET/PRINTING_ERRORS/general_data/all_images copy.csv",])
    