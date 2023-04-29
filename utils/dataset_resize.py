import os
import cv2

'''
Resize images to 256x256
and saves then in a new folder
'''

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))

DATASET_NAME = "PRINTING_ERRORS"
PATH = "%s/PRINTING_ERRORS/test_images_silver/" % (SCRIPT_DIR)
TRAIN_CSV = "%s/%s/train_hand_select.csv" % (SCRIPT_DIR, DATASET_NAME)
TEST_CSV = "%s/%s/test_hand_select.csv" % (SCRIPT_DIR, DATASET_NAME)

import cv2
import os
import glob
inputFolder = PATH
new_path = PATH.replace("test_images_silver", "test_images_silver265")
#create new folder
if not os.path.exists(new_path):
    os.makedirs(new_path)

for img in glob.glob(inputFolder + "/*.png"):
    print(img)
    image = cv2.imread(img)
    h = image
    imgResized = cv2.resize(image, (256, 256))
    cv2.imwrite(os.path.join(new_path,img.split(os.sep)[-1]), imgResized)