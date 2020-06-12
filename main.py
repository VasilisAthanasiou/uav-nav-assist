import cv2 as cv
import numpy as np
import os
from matplotlib import pyplot as plt
import random

# Set image directory
images_directory = 'datasets/sources/source_diverse/cloudy'
templates_directory = 'datasets/templates/test-temp'
match_pos_path = 'testdatasetdata.txt'

# Append each image path into a list
source_paths = [os.path.join(images_directory, image_path) for image_path in os.listdir(images_directory)]
templates_paths = [os.path.join(templates_directory, template_path) for template_path in os.listdir(templates_directory)]
source_paths.sort()
templates_paths.sort(key=lambda name: int(os.path.splitext(os.path.basename(name))[0]))

for elem in templates_paths:
    print(elem)

# ---------------------------------------------Image Processing--------------------------------------------------------- #


def process_image(src_img, resize):
    # Resize image
    res_img = cv.resize(src_img, None, fx=resize, fy=resize, interpolation=cv.INTER_CUBIC)

    # Convert to grayscale
    gray_image = cv.cvtColor(res_img, cv.COLOR_BGR2GRAY)

    # Remove noise
    gray_image = cv.blur(gray_image, (5, 5), gray_image)

    # Binarize image

    # bin_img = cv.threshold(gray_image, 120, 255, cv.THRESH_BINARY)
    (thresh, bin_img) = cv.threshold(gray_image, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    # Erode binary image
    kernel = np.ones((2, 2), np.uint8)
    bin_img = cv.morphologyEx(bin_img, cv.MORPH_OPEN, kernel)

    # Dilate binary image
    bin_img = cv.morphologyEx(bin_img, cv.MORPH_CLOSE, kernel)

    return bin_img


# ---------------------------------------------------------------------------------------------------------------------- #

# ------------------------------------------ Matching Algorithm -------------------------------------------------------- #

def find_pixel_dx(sat_img, temp_img):  # (satellite / source image, template image)
    # Store the image array shapes
    sat_height, sat_width, _ = sat_img.shape
    temp_height, temp_width, _ = temp_img.shape

    comp_matrix = []  # The source-sensed image comparison matrix is stored here, as well as the first x px of the roi
    max_matrix = (np.zeros(temp_img.shape), 0)  # Will compare with comparison results, to find the greatest match

    processed_template = process_image(temp_img, resize_value)
    cv.imshow('Processed Template', processed_template)
    # Compare every possible roi with the template by performing logical operations and find the greatest match
    for most_left_pixel in range(int(sat_width - temp_width)):  # Search the whole image
        # print("MLP : {} | TEMPWIDTH : {} | SUM : {}".format(most_left_pixel, temp_width, most_left_pixel + temp_width))

        roi = sat_img[0:temp_height, most_left_pixel:most_left_pixel + temp_width]  # Region of interest
        processed_roi = process_image(roi, resize_value)

        and_result = np.logical_and(processed_roi, processed_template)  # Perform AND on each roi and template pixel (1)
        xnor_result = np.invert(np.logical_xor(processed_roi, processed_template))  # Perform XNOR on each roi and template pixel (2)
        comp_matrix.append((np.array(np.logical_or(and_result, xnor_result), dtype=int), most_left_pixel))  # (1) OR (2)

        comp_sum = int(np.sum(comp_matrix[most_left_pixel][0]))  # Sum the comparison matrix
        max_sum = int(np.sum(max_matrix[0]))  # Sum the max matrix

        if comp_sum > max_sum:  # Find max comparison matrix
            max_matrix = np.copy(comp_matrix[most_left_pixel][0]), most_left_pixel  # Store matrix and most left pixel
            cv.imshow('Processed ROI', processed_roi)

    return max_matrix  # Return the result


# ---------------------------------------------------------------------------------------------------------------------- #

# ------------------------------------------ Statistical Analysis ------------------------------------------------------ #

# Read the images
source_images = []
for image_path in source_paths:
    source_images.append(cv.imread(image_path))

templates = []
for template_path in templates_paths:
    templates.append(cv.imread(template_path))

match_pos_file = open(match_pos_path, 'r')
actual_match_position = match_pos_file.readlines()

error = float(0)  # Error for each template, error for each image and error for the whole set
error_img = []

counter = 0

resize_value = 0.5

for img in source_images:
    img_height, img_width, _ = img.shape
    template_height, template_width, _ = templates[counter].shape

    print("Source {}x{} , Template {}x{}".format(img_height, img_width, template_height, template_width))

    templates_per_image = 1  # img_width - template_width

    error_temp_img = 0
    for i in range(templates_per_image):

        matched_matrix_and_location = find_pixel_dx(img, templates[counter])
        # print("Sensed first pixel - actual first pixel : {}".format(
        #     np.abs(matched_matrix_and_location[1] - int(actual_pixel_position[counter]))))
        error_temp_img += float(np.abs(matched_matrix_and_location[1] - int(actual_match_position[counter])) / img_width * resize_value)
        # print("Error for image {} and template {} : {:3f}".format(int(counter / templates_per_image), counter % templates_per_image, error_temp_img))
        counter += 1
    error_img.append(error_temp_img / templates_per_image)
    print("Error for image {} : {}".format(int(counter / templates_per_image), error_temp_img / templates_per_image))
    cv.imshow('Original Image', img)

    #plt.imshow(matched_matrix_and_location[0])
    #plt.show()

    while True:
        if cv.waitKey(0) == 27:
            cv.destroyAllWindows()
            plt.close()
            break

error = (sum(error_img) / len(source_images)) * 100

print("There is {}% error.".format(error))
# ---------------------------------------------------------------------------------------------------------------------- #


# match_matrix_and_location = find_pixel_dx(processedImg, processedTemplate)
#
# print("Numpy array object {}.\n\nStarts at pixel no. {} on x axis.".format(match_matrix_and_location[0],
#                                                                            match_matrix_and_location[1]))

# Close the window when the user presses the ESC key
