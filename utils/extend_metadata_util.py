import pandas as pd
import os
import sorting_utility as util
import shutil
import crawl_dataset
import cv2
import matplotlib.pyplot as plt
import dataset_preprocessing as processing

# get the directory of this script
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))

'''
RAW_DATA_PATH = "%s/%s" % (SCRIPT_DIR, "DATASET RAW")
###### CHANGE HERE FOR NEW DATASET ###################################
SORTED_DATASET = "%s/%s" % (SCRIPT_DIR, "DATASET_SORTED_512")   ##########   
FULL_DATASET = "%s/%s" % (SCRIPT_DIR, "FULL_DATASET_512")   ##########

######################################################################
SORTED_DATASET_csv = "%s/%s" % (SORTED_DATASET, "__csv")
COLLECT_CSV_PATH = "%s/%s" % (SCRIPT_DIR, "csv")
CSV_LIST_PATH = "%s/%s" % (SCRIPT_DIR, "csv_file_paths.yaml")
TEST_CSV_PATH = "%s/%s" % (SCRIPT_DIR, "Recording_01.12.2022_15_27_47.csv")

UNCLASSIFIED_COMBINED_CSV_PATH = "%s/%s" % (SCRIPT_DIR, "unclassified_combined.csv")
'''

# get the directory of this script
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
#CSV_LIST_PATH = "%s/%s" % (SCRIPT_DIR, "csv_file_paths.yaml")
#TEST_CSV_PATH = "%s/%s" % (SCRIPT_DIR, "Recording_01.12.2022_15_27_47.csv")



# open csv file into pandas dataframe and show it
def open_csv_file(file_path):
    return pd.read_csv(file_path, delimiter=";")

# add column to pandas dataframe and name, and fill it with zeros
def add_column_to_df(df, column_name, fill_value=0):
    df[column_name] = fill_value
    return df

# create recording name from file path
def generate_name_from_path(file_path):
    filename = file_path.split("/")[-1]    
    # remove file extension (remove only last item of list)
    filename = filename.split(".")[:-1]
    # append all items in list to string
    filename = "".join(filename)
    return filename

def get_file_name_from_path(file_path):
    return file_path.split("/")[-1]

def dump_csv(df, file_name):
    df.to_csv(file_name, index=False, sep=";")

def collect_headers(file_paths):
    headers = []
    for file_path in file_paths:
        df = pd.read_csv(file_path, header=0, delimiter=";")
        headers += list(set(df.columns) - set(headers))
    print(headers)

def remove_columns(file_path_list, exclude_keys):
    for file_path in file_path_list:
        df = pd.read_csv(file_path, header=0, delimiter=";")        
        for key in exclude_keys:
            if key in df.columns:
                df.drop(columns=[key,], inplace=True)
        
        df.to_csv(file_path, index=False, sep=";")

def merge_dataframes(df1, df2):
    # Use outer join to combine DataFrames
    combined_df = pd.merge(df1, df2, how='outer')
    return combined_df

def inner_merge_dataframes(df1, df2, column):
    return pd.merge(df1, df2, on=column, how='inner')

def crete_dataframe_from_class(class_name, DATASET_BASEPATH, file_extension=".png"):
    
    class_folder = "%s/%s" % (DATASET_BASEPATH, class_name)
    image_paths = util.get_all_file_paths(class_folder, ext=file_extension)
    image_names = [get_file_name_from_path(path) for path in image_paths]

    df = pd.DataFrame(image_names, columns=["image"])
    df = add_column_to_df(df, "class", fill_value=class_name)
    
    return df

def copy_images_to(csv, source, dest):
    df = open_csv_file(csv)
    images = df["image"].tolist()
    for image in images:
        print(image)
        source_path = "%s/%s" % (source, image)
        dest_path = "%s/%s" % (dest, image)
        shutil.copyfile(source_path, dest_path) 

# combine image:class df with unclassified combined metadata
def combine_class_metadata_classlist(class_list, class_csv_dir, unclassified_combined_csv_path, destination_dir="%s/%s" % (SCRIPT_DIR, "combined_csv")):
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)
    for class_name in class_list:
        source_class_csv = "%s/%s.csv" % (class_csv_dir, class_name)
        destination_csv = "%s/%s.csv" % (destination_dir, class_name)
        class_df = open_csv_file(source_class_csv)
        unclassified_combined_csv = open_csv_file(unclassified_combined_csv_path)
        combined_df = inner_merge_dataframes(class_df, unclassified_combined_csv, "image")
        dump_csv(combined_df, destination_csv)

def combine_class_metadata(class_csv, metadata_csv, output_file):
    class_df = open_csv_file(class_csv)
    unclassified_combined_csv = open_csv_file(metadata_csv)
    combined_df = inner_merge_dataframes(class_df, unclassified_combined_csv, "image")
    dump_csv(combined_df, output_file)

def inner_merge_csv(dir, class_csv, metadata_csv):
    class_df = open_csv_file("%s/%s" % (dir, class_csv))
    metadata_df = open_csv_file("%s/%s" % (dir, metadata_csv))
    combined_df = inner_merge_dataframes(class_df, metadata_df, "image")
    destination_csv = "%s/%s" % (dir, "combined.csv")
    dump_csv(combined_df, destination_csv)

def make_image_class_csv(class_list, DATASET_BASEPATH, dst):
    print("Scanning Image Folder...")
    for class_name in class_list:    
        df = crete_dataframe_from_class(class_name, DATASET_BASEPATH)
        dump_csv(df, "%s/%s.csv" % (dst, class_name))

# collects all csv described in the csv_file_paths.yaml and combines them into one csv, which is then saved in the output file
def combine_all_csv_files(source, output_file):    
    csv_filenames = util.get_all_file_paths(source, extension=".csv")
    df = pd.DataFrame()
    for file in csv_filenames:
        new_df = open_csv_file(file)
        df = pd.concat([df, new_df])
    dump_csv(df, output_file)

def combine_csv_in_dir(source_dir, dest_dir, output_file_name):    
    #csv_file_paths = ["%s/%s" % (dir, filename) for filename in filenames]
    #csv_file_paths = ["%s.csv" %path for path in csv_file_paths if not path.endswith(".csv")]

    csv_file_paths = util.get_all_file_paths(source_dir, ext=".csv")

    output_file = "%s/%s" % (dest_dir, output_file_name)
    df = pd.DataFrame()
    for file in csv_file_paths:
        new_df = open_csv_file(file)
        df = pd.concat([df, new_df])
    print("Saving to %s" % output_file)
    dump_csv(df, output_file)

def split_dataframe_with_class_balance(df, percent, column_name):
    y = df[column_name]
    sss = skl.StratifiedShuffleSplit(n_splits=1, test_size=percent, random_state=0)
    for train_index, test_index in sss.split(df, y):
        train_df, test_df = df.iloc[train_index], df.iloc[test_index]
    return train_df, test_df

#calculates the mean brightness of an image in grayscale
def image_brightness(image_path):
    # Load the image using cv2.imread()
    image = cv2.imread(image_path)
    # Convert the image to grayscale
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Calculate the mean brightness using cv2.mean()
    mean_brightness, *rest = cv2.mean(grayscale_image)
    #print(cv2.mean(grayscale_image))
    return mean_brightness

# itrates over all images in the dataframe and adds the brightness of each image to the dataframe
def add_brightness_to_dataframe(df, image_dir):
    new_df = df.copy(deep=True)
    new_df["brightness"] = 0
    for index, row in new_df.iterrows():
        image_name = row["image"]
        image_path = "%s/%s" % (image_dir, image_name)
        brightness = int(image_brightness(image_path))
        new_df.at[index, "brightness"] = brightness
        print("Writing brightness for %s: %s" % (image_name, brightness))

    return new_df

# adds the mean brightness of each recording to the dataframe, is dependent on the brightness column
def add_mean_brightness_to_recording(df):
    new_df = df.copy(deep=True)
    recordings = util.get_unique_values_from_column(df, "recording")
    new_df["mean_brightness"] = 0
    
    for recording in recordings:
        recording_df = df[df["recording"] == recording]        
        mean_brightness = int(recording_df["brightness"].mean())
        print(mean_brightness)
        new_df.loc[new_df["recording"] == recording, "mean_brightness"] = mean_brightness
            
    return new_df

def calculate_mean_brightness(df):
    mean_brightness = int(df["brightness"].mean())
    print("Mean brightness: %s" % mean_brightness)
    return mean_brightness

def create_histogram(data, bins=64, save=False):
    # Create a figure and axis object
    fig, ax = plt.subplots()

    # Create a histogram using matplotlib's hist function
    ax.hist(data, bins=bins)

    # Set the title and labels for the plot
    #ax.set_title('Brightness Histogram')
    ax.set_xlabel('Brightness')
    ax.set_ylabel('Frequency')


    ax.set_xlim([0, 255])
    if save:
        plt.savefig(save)
    # Show the plot
    plt.show()

def create_mean_brightness_histogram(df):
    print("Creating mean brightness histogram...")
    recordings = util.get_unique_values_from_column(df, "recording")    
    data = []
    for recording in recordings:        
        #print(df.loc[df["recording"] == recording])
        mean_brightness = util.get_unique_values_from_column(df.loc[df["recording"] == recording], "mean_brightness")[0]
        data.append(mean_brightness)
    create_histogram(data)

def drop_where_column_is_in_list(df, column, values=["35", "36", "37", "38", "39", "40", "41", "unknown"]):
    new_df = df.drop(df.loc[df[column].isin(values)].index, inplace=False)

    return new_df

def merge_subclasses(df):
    merged = util.change_column_where_other_column_has_value(df, "class", "MIN_IMPERFECTION", "class", "GOOD")
    merged = util.change_column_where_other_column_has_value(merged, "class", "STRINING_subtle", "class", "STRINGING")
    merged = util.change_column_where_other_column_has_value(merged, "class", "UNDEREXTRUSION_subtle", "class", "UNDEREXTRUSION")
    return merged

def view_images(df, image_dir, show_rec=False):
    images = df["image"].tolist()
    for image_name in images:
        image_path = "%s/%s" % (image_dir, image_name)
        image = cv2.imread(image_path)
        # check if recording should be shown -> this happens in a new window
        if show_rec:        
            recording = df.loc[df["image"] == image_name]["recording"].item()
            print(recording)
        else:
            recording="IMAGE"
        cv2.imshow(recording, image)
        cv2.waitKey(0)

def raw_data_to_dataset_routine():
    RAW_DATA_DIR = "%s/%s" % (SCRIPT_DIR, "RAW_DATA/SILVER_BED_SORTED")
    CSV_DIR = "%s/%s" % (RAW_DATA_DIR, "__csv")
    CLASS_DIR = "%s/%s" % (RAW_DATA_DIR, "class_csv")

    CLASSES = "%s/%s" % (RAW_DATA_DIR, "classes_combined.csv")
    METADATA = "%s/%s" % (RAW_DATA_DIR, "combined.csv")

    UNCROPPED_DIR = "%s/%s" % (RAW_DATA_DIR, "uncropped")
    CROPPED_DIR = "%s/%s" % (RAW_DATA_DIR, "cropped")

    TESTSET = "%s/%s" % (RAW_DATA_DIR, "silver_testset.csv")

    #df = open_csv_file(METADATA)
    #df.loc[df["shape"] == "complex", "shape"] = 10
    #dump_csv(df, METADATA)


    #df = open_csv_file("RAW_DATA/SILVER_BED_RAW/STRINGING_3/Recording_23.03.2023_13_47_38.csv")
    #new_df = processing.filter_nth_entries(df, 15)
    #dump_csv(new_df, "RAW_DATA/SILVER_BED_RAW/STRINGING_3/Recording_23.03.2023_13_47_38.csv")

    #combine_csv_in_dir(CSV_DIR, RAW_DATA_DIR, "combined.csv")
    
    #df = open_csv_file("%s/%s" % (RAW_DATA_DIR, "combined.csv"))
    #util.drop_column(df, "class")
    #dump_csv(df, "%s/%s" % (RAW_DATA_DIR, "combined.csv"))

    #make_image_class_csv(["GOOD", "STRINGING", "UNDEREXTRUSION"] ,RAW_DATA_DIR, CLASS_DIR)

    #combine_csv_in_dir(CLASS_DIR, RAW_DATA_DIR, "classes_combined.csv")

    #combine_class_metadata(CLASSES, METADATA, "%s/%s" % (RAW_DATA_DIR, "merged.csv"))

    #df = open_csv_file("%s/%s" % (RAW_DATA_DIR, "merged.csv"))
    #df_new = processing.make_class_to_number(df)
    #dump_csv(df_new, "%s/%s" % (RAW_DATA_DIR, "silver_testset.csv"))

    #copy_images_to(TESTSET, "RAW_DATA/SILVER_BED_SORTED/Neuer Ordner", UNCROPPED_DIR)

    #util.crop_image_data(UNCROPPED_DIR, CROPPED_DIR, 330, 180)

if __name__ == "__main__":

    file = "/mnt/c/Users/Nils/Desktop/DATASET/PRINTING_ERRORS/test_datasets/silver_test.csv"

    image_dir = "/mnt/c/Users/Nils/Desktop/DATASET/PRINTING_ERRORS/test_images_silver265"

    df = open_csv_file(file)
    view_images(df, image_dir, show_rec=False)