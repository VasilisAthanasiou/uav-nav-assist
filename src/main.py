import cv2 as cv
import numpy as np
import os
from matplotlib import pyplot as plt
import random

# ---------------------------------------------Image Processing--------------------------------------------------------- #

def process_image(src_img, resize=0.5):

    # # Resize image
    # res_img = cv.resize(src_img, None, fx=resize, fy=resize, interpolation=cv.INTER_CUBIC)

    # Convert to grayscale
    gray_image = cv.cvtColor(src_img, cv.COLOR_BGR2GRAY)

    # # Remove noise
    # gray_image = cv.blur(gray_image, (5, 5), gray_image)
    #
    # # Binarize image
    # (thresh, bin_img) = cv.threshold(gray_image, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    #
    # # Erode binary image
    # kernel = np.ones((2, 2), np.uint8)
    # bin_img = cv.morphologyEx(bin_img, cv.MORPH_OPEN, kernel)
    #
    # # Dilate binary image
    # bin_img = cv.morphologyEx(bin_img, cv.MORPH_CLOSE, kernel)

    return gray_image


# ---------------------------------------------------------------------------------------------------------------------- #

# ------------------------------------------ Matching Algorithm -------------------------------------------------------- #

# Returns the top left pixel location of the sensed template
def find_target(src, temp):

    temp_height, temp_width = temp.shape
    # Apply template matching
    res = cv.matchTemplate(src, temp, cv.TM_CCOEFF)
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)

    return max_loc # Return top left position

# ---------------------------------------------------------------------------------------------------------------------- #

# ------------------------------------------ Statistical Analysis ------------------------------------------------------ #

def evaluate(src, temp, actual_match, n_templates, resize_value=0.5):

    error = float(0)  # Error for the whole set
    error_img = []  # Error for each image Sum(error for every template / number of templates)
    num_of_images = len(src)

    counter = 0  # Keeps track of each iterations

    for img in src:  # For every image
        img_height, img_width, _ = img.shape
        template_height, template_width, _ = temp[counter].shape

        processed_img = process_image(img)

        print("Source {}x{} , Template {}x{}".format(img_height, img_width, template_height, template_width))

        templates_per_image = n_templates  # This is the num of times the src will be scanned

        is_middle = False  # Is used to print when we are in the middle of evaluating
        error_temp_img = 0  # Error for one template. How displaced the sensed frame is from the actual one

        for i in range(templates_per_image):
            # Store both actual and sensed x and y
            actual_x, actual_y, _ = actual_match[i].split(',')
            sensed_x, sensed_y = find_target(processed_img, process_image(temp[counter]))

            # Error for one template is the added absolute differences between x and y divided the number of pixels
            error_temp_img += np.abs(int(sensed_x) - int(actual_x)) + np.abs(int(sensed_y) - int(actual_y))
            counter += 1
            print("For template {} : Sensed X : {} , Actual X : {} , Sensed Y : {} , Actual Y : {} , ERROR : {}".format(i, sensed_x, actual_x, sensed_y, actual_y, np.abs(int(sensed_x) - int(actual_x)) + np.abs(int(sensed_y) - int(actual_y))))

        error_img.append(error_temp_img / templates_per_image)  # Error for a whole image tested with multiple templates
        print("Mean error for image {} : {}px".format(int(counter / templates_per_image), error_temp_img / templates_per_image))

    # Error for all images
    error = sum(error_img)

    print("There is {} pixel mean error.".format(error))

# -------------------------------------------------Main----------------------------------------------------------------- #


# Set image directory
images_directory = '../datasets/sources/source-diverse/blurred-cloudy'
templates_directory = '../datasets/templates/templates-diverse/images'
match_pos_path = '../datasets/templates/templates-diverse/dataset-diverse-loc.txt'

# Append each image path into a list
source_paths = [os.path.join(images_directory, image_path) for image_path in os.listdir(images_directory)]
templates_paths = [os.path.join(templates_directory, template_path) for template_path in os.listdir(templates_directory)]
source_paths.sort()
templates_paths.sort(key=lambda name: int(os.path.splitext(os.path.basename(name))[0]))

# Print all paths to make sure everything is ok
for elem in templates_paths:
    print(elem)

# Read the images
source_images = []
for image_path in source_paths:
    source_images.append(cv.imread(image_path))

# Read the templates
templates = []
for template_path in templates_paths:
    templates.append(cv.imread(template_path))

# Read the txt file with the template's actual position
match_pos_file = open(match_pos_path, 'r')
actual_match_position = match_pos_file.readlines()

# Evaluate the matching method. The method is hardcoded into the evaluation. This should be changed
evaluate(source_images, templates, actual_match_position, 16)


# ---------------------------------------------------------------------------------------------------------------------- #
