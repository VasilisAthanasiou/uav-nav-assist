import cv2 as cv
import numpy as np
import os
from scipy import ndimage

# ---------------------------------------------Image Processing--------------------------------------------------------- #

def process_image(src_img, resize=-1, rot_deg=0):
    '''Applies image processing techniques on an image.

    :param src_img: Image to be processed
    :param resize: Resize value. Takes float values between 0 and 1.
    :param rot_deg: Degrees that the image will be rotated
    :return: Returns the processed image

    '''

    # Resize image
    if resize != -1:
        res_img = cv.resize(src_img, None, fx=resize, fy=resize, interpolation=cv.INTER_CUBIC)

    # Convert to grayscale
    gray_image = cv.cvtColor(src_img, cv.COLOR_BGR2GRAY)

    processed_img = gray_image

    # Rotate the image
    if rot_deg != 0:
        processed_img = ndimage.rotate(gray_image, rot_deg)

    return processed_img

# ---------------------------------------------------------------------------------------------------------------------- #

# ------------------------------------------ Matching Algorithm -------------------------------------------------------- #

# Returns the top left pixel location of the sensed template
def find_target(src, temp):

    '''Uses openCV's matchTemplate to find a desired target inside the source image.

    :param src: Source image
    :param temp: Template image
    :return: Tuple containing the sensed target top left coordinates
    '''

    temp_height, temp_width = temp.shape
    # Apply template matching
    res = cv.matchTemplate(src, temp, cv.TM_CCOEFF)
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)

    return max_loc  # Return top left position

# ---------------------------------------------------------------------------------------------------------------------- #

# ------------------------------------------ Statistical Analysis ------------------------------------------------------ #

def evaluate(src, temp, actual_match, n_templates, resize_value=-1, rotation=0):

    '''Performs the find_target() function for multiple source images on multiple templates, compares the results with
    the locations in the actual_match list, calculates the displacent and finally; writes the results on a text file.

    :param src: Source image
    :param temp: Template image
    :param actual_match: List of strings containing the correct coordinates for a give target
    :param n_templates: Number of template images
    :param resize_value: Resize value for process_image()
    :param rotation: Rotation value for process_image()
    :return: A well structured string containing the results of the experiment
    '''

    error = float(0)  # Error for the whole set
    error_img = []  # Error for each image Sum(error for every template / number of templates)
    num_of_images = len(src)

    counter = 0  # Keeps track of each iterations
    result = ''

    for img in src:
        img_height, img_width, _ = img.shape
        template_height, template_width, _ = temp[counter].shape

        processed_img = process_image(img)

        print("Source {}x{} , Template {}x{}".format(img_height, img_width, template_height, template_width))

        templates_per_image = n_templates  # This is the num of times the src will be scanned

        error_temp_img = 0  # Error for one template. How displaced the sensed frame is from the actual one

        for i in range(templates_per_image):
            # Store both actual and sensed x and y
            actual_x, actual_y, _ = actual_match[i].split(',')
            sensed_x, sensed_y = find_target(processed_img, process_image(temp[counter], rotate=True, rot_deg=rotation))

            # DEBUG
            # cv.imshow('Template', process_image(temp[counter], rotate=True, rot_deg=rotation))
            # while True:
            #     if cv.waitKey(1) == 27:
            #         cv.destroyAllWindows()
            #         break

            # Error for one template is the added absolute differences between x and y divided the number of pixels
            error_temp_img += np.abs(int(sensed_x) - int(actual_x)) + np.abs(int(sensed_y) - int(actual_y))
            counter += 1
            print("For template {} : Sensed X : {} , Actual X : {} , Sensed Y : {} , Actual Y : {} , ERROR : {}".format(i, sensed_x, actual_x, sensed_y, actual_y, np.abs(int(sensed_x) - int(actual_x)) + np.abs(int(sensed_y) - int(actual_y))))


        error_img.append(error_temp_img / templates_per_image)  # Error for a whole image tested with multiple templates
        result += ("Mean error for image {} : {}px\n".format(int(counter / templates_per_image), error_temp_img / templates_per_image))

    # Error for all images
    error = sum(error_img)/num_of_images

    result += ("There is {} pixel mean error.\n\n".format(error))
    return result
# ------------------------------------------------- Main --------------------------------------------------------------- #


# Set image directory
images_directory = '../datasets/sources/source-diverse/source'
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
result_text = evaluate(source_images, templates, actual_match_position, 16, rotation=2)

# Write the experiment results on a text file
file = open("experiment-results.txt", "a")
file.write("-------------- Results using 2deg rotation on source images --------------\n{}".format(result_text))
# ---------------------------------------------------------------------------------------------------------------------- #
