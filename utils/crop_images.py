from IPython.display import display
import pandas as pd
import os
import tabulate
import sorting_utility as util
import cv2
import os
import shutil
import extend_metadata_util as ext_util
import copy

# get the directory of this script
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))

RAW_DATA_PATH = "%s/%s" % (SCRIPT_DIR, "DATASET RAW")
SORTED_DATASET = "%s/%s" % (SCRIPT_DIR, "DATASET_SORTED_512")
SORTED_DATASET_csv = "%s/%s" % (SORTED_DATASET, "__csv")

raw_COLLECT_CSV_PATH = "%s/%s" % (SCRIPT_DIR, "csv")
raw_CSV_LIST_PATH = "%s/%s" % (SCRIPT_DIR, "csv_file_paths.yaml")

TEST_IMAGE_PATH = "%s/%s" % (SCRIPT_DIR, "ELP_12MP_01.12.2022_166992955563.png")
UNCLASSIFIED_COMBINED_CSV_PATH = "%s/%s" % (SCRIPT_DIR, "unclassified_combined.csv")




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

def copy_folder_structure(src, dst):
    # make root directory if not exists
    if not os.path.exists(dst):
        os.makedirs(dst)
    #iterate through all files in root directory
    for item in os.listdir(src):
        # generate sub folder paths
        sub_src = os.path.join(src, item)
        sub_dst = os.path.join(dst, item)
        if os.path.isdir(sub_src):
            # make subdirectory if not exists
            if not os.path.exists(sub_dst):
                os.makedirs(sub_dst)
        #else:
        #   print("NOT A DIRECTORY")


def crop_dataset(src, dst, classes=None, crop_size=512):

    dst_csv_folder = "%s/%s" % (dst, "__csv")
    if not os.path.exists(dst_csv_folder):
        os.makedirs(dst_csv_folder)

    # check if classes are specified, take all classes if not
    if classes is None:
        classes = [c for c in os.listdir(src) if not os.path.basename(c).startswith("__") and os.path.isdir("%s/%s"%(src,c))]

    # iterate through all classes
    for src_class in classes:
        print(src_class)
        src_class_csv_path = "%s/%s/%s%s" % (src, "__csv", src_class, ".csv")
        src_class_path = "%s/%s" % (src, src_class)
        # generate new name for destination class
        dst_class_name = "%s_%s" % (crop_size, src_class)
        dst_folder_path = "%s/%s" % (dst, dst_class_name)
        dst_class_csv_path = "%s/%s/%s%s" % (dst, "__csv", dst_class_name, ".csv")
        
        # make destination class folder if not exists
        if not os.path.exists(dst_folder_path):
            os.makedirs(dst_folder_path)

        # read csv file to pandas dataframe
        src_df = pd.read_csv(src_class_csv_path, sep=";")
        dst_df = src_df.copy(deep=True)
        
        # iterate through all images in column "image"
        for index, row in src_df.iterrows():
            src_image_path = "%s/%s" % (src_class_path, row["image"])
            dst_image_name = row["image"]
            dst_image_path = "%s/%s" % (dst_folder_path, dst_image_name)
            # copy image to destination folder
            # generate image path in destination folder and rename image

            print("%s: %s   -->   %s" %(index, src_image_path, dst_image_path))
            
            ### crop image
            # read image
            src_image = cv2.imread(src_image_path)
            # crop image
            dst_image = crop_image_at_coordinates(src_image, crop_size, crop_size, 300, 400)
            #save image
            cv2.imwrite(dst_image_path, dst_image)

            ## change entry in new dataframe
            dst_df.at[index, "image"] = dst_image_name
            # save new dataframe to csv file
        
        ext_util.dump_csv(dst_df, dst_class_csv_path)

# careful, this may make bad names
def rename_images(basedir):
    folders = util.get_all_folder_paths(basedir)
    for folder_path in folders:
        for filename in os.listdir(folder_path):
            if filename.endswith(".png"):
                new_filename = filename[4:]
                old_file_path = os.path.join(folder_path, filename)
                new_file_path = os.path.join(folder_path, new_filename)
                print("%s   -->   %s" %( old_file_path, new_file_path))
                os.rename(old_file_path, new_file_path)
        
        
 
        









if __name__ == "__main__":
    passs

    #crop_dataset(SORTED_DATASET, "%s/%s" % (SCRIPT_DIR, "DATASET_SORTED_512"), classes=None, crop_size=512)

    
    #rename_images(SORTED_DATASET)






















#copy_folder_structure(SORTED_DATASET, "%s/%s" % (SCRIPT_DIR, "DATASET_SORTED_512"))


