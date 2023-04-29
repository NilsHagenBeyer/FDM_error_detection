import pandas as pd
import os
import sorting_utility as util
import shutil
import cv2
import time
# get the directory of this script
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))

''' This class can be used to crawl through a datasets metadata and alter columns while watching the
images. It is used to extend the metadata of the dataset.
The class is originally written to crawl through the recordings of the dataset and add 
the identifier of the corresponding geometry to the metadata.'''

class Crawler():
    def __init__(self, images, metadata, **kwargs) -> None:
        self.image_dir = images
        self.metadata_dir = metadata
        self.show_images = kwargs.get("show_images", True)
        self.save_metadata = kwargs.get("save", False)
        self.dataframe = util.open_csv_file(self.metadata_dir)

        self.altered_dataframe = self.dataframe.copy(deep=True)
        print("Saving: %s" %self.save_metadata)

    def get_image(self, image_name):
        image_path = "%s/%s" % (self.image_dir, image_name)
        return util.open_image(image_path)

    def crawl(self, crawl_column, change_column, cls=None):
        # get list of unique values in column
        crawl_through = util.get_unique_values_from_column(self.dataframe, crawl_column)

        for item in crawl_through:
            print("Current Item: %s" % item)
            # get all rows with current item
            if cls:
                rows = util.get_all_from_column_with_class(self.dataframe, crawl_column, item, cls)
            else:
                rows = util.get_all_from_column(self.dataframe, crawl_column, item)

            if self.show_images:
                # show all images
                for index, row in rows.iterrows():
                    image = self.get_image(row["image"])
                    try:
                        cv2.imshow("IMAGE", image)
                        cv2.waitKey(0)
                    except:
                        print("Could not show image: %s" % row["image"])
                        pass

            time.sleep(2)
            print("Current Change Values:\n%s" %rows[change_column])
            new_value = input("Change %s to: " %change_column)
            self.altered_dataframe = util.change_column_where_other_column_has_value(self.altered_dataframe, crawl_column, item, change_column, new_value)
            print(util.get_all_from_column(self.altered_dataframe, crawl_column, item)[change_column])

            if self.save_metadata:
                self.save()

            print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")

        



    
    def save(self):
        new_metadata = os.path.splitext(self.metadata_dir)[0] + "_changed.csv"
        util.dump_csv(self.altered_dataframe, new_metadata)



            
if __name__ == "__main__":

    SOURCE_DIR = "%s/%s" % (SCRIPT_DIR, "PRINTING_ERRORS")


    DATASET_CSV = "%s/%s" % (SOURCE_DIR, "all_metadata_2 copy.csv")
    IMAGE_DATA = "%s/%s" % (SOURCE_DIR, "train_images")


    crawler = Crawler(IMAGE_DATA, DATASET_CSV, show_images=True, save=True)
    crawler.crawl("recording", "shape")
    #crawler.save()