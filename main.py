import cv2 as cv
import numpy as np
import os
import random

# Set image directory
images_directory = 'satellite_images'
templates_directory = 'templates-x'  # Try templates x for statistical analysis

# Append each image path into a list
source_paths = [os.path.join(images_directory, image_path) for image_path in os.listdir(images_directory)]
templates_paths = [os.path.join(templates_directory, template_path) for template_path in os.listdir(templates_directory)]
source_paths.sort()
templates_paths.sort()


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


# Read the image
# img = [cv.imread(image) for image in images_paths]
# template = [cv.imread(template) for template in templates_paths]
source_images = []
for image_path in source_paths:
    source_images.append(cv.imread(image_path))

templates = []
for template_path in templates_paths:
    templates.append(cv.imread(template_path))

img_height, img_width, _ = source_images[0].shape

# Display images
sat_index = 3  # Horizontal Test images at 3 and 0 (for x-temp 3)
temp_index = 1  # Horizontal Test images at 3 and 6


# processedImg = process_image(img[sat_index])
# processedTemplate = process_image(template[temp_index])
#
# cv.imshow('Original Image', img[sat_index])
# cv.imshow('Processed Image', processedImg)
#
# cv.imshow('Template', template[temp_index])
# cv.imshow('Processed Template', processedTemplate)


# ------------------------------------------ Matching Algorithm -------------------------------------------------------- #

def find_pixel_dx(sat_img, temp_img):  # (satellite / source image, template image)
    # Store the image array shapes
    sat_height, sat_width = sat_img.shape
    temp_height, temp_width = temp_img.shape

    comp_matrix = []  # The source-sensed image comparison matrix is stored here, as well as the first x px of the roi
    max_matrix = (np.zeros(temp_img.shape), 0)  # Will compare with comparison results, to find the greatest match

    # Compare every possible roi with the template by performing logical operations and find the greatest match
    for most_left_pixel in range(int(sat_width - temp_width)):  # Search the whole image
        # print("MLP : {} | TEMPWIDTH : {} | SUM : {}".format(most_left_pixel, temp_width, most_left_pixel + temp_width))

        roi = sat_img[0:temp_height, most_left_pixel:most_left_pixel + temp_width]  # Region of interest
        and_result = np.logical_and(roi, temp_img)  # Perform AND on each roi and template pixel (1)
        xnor_result = np.invert(np.logical_xor(roi, temp_img))  # Perform XNOR on each roi and template pixel (2)
        comp_matrix.append((np.array(np.logical_or(and_result, xnor_result), dtype=int), most_left_pixel))  # (1) OR (2)

        comp_sum = int(np.sum(comp_matrix[most_left_pixel][0]))  # Sum the comparison matrix
        max_sum = int(np.sum(max_matrix[0]))  # Sum the max matrix

        if comp_sum > max_sum:  # Find max comparison matrix
            max_matrix = np.copy(comp_matrix[most_left_pixel][0]), most_left_pixel  # Store matrix and most left pixel

    return max_matrix  # Return the result


# ---------------------------------------------------------------------------------------------------------------------- #

# ------------------------------------------ Statistical Analysis ------------------------------------------------------ #


pixel_position_file = open('dataset-img-info.txt', 'r')
actual_pixel_position = pixel_position_file.readlines()

error = float(0)

counter = 0
templates_per_image = 600
resize_value = 0.5

for img in source_images:
    processed_image = process_image(img, resize_value)
    for i in range(templates_per_image):
        processed_template = process_image(templates[counter], resize_value)
        matched_matrix_and_location = find_pixel_dx(processed_image, processed_template)
        print("Sensed first pixel - actual first pixel : {}".format(
            np.abs(matched_matrix_and_location[1] - int(actual_pixel_position[counter]))))
        error += float(np.abs(matched_matrix_and_location[1] - int(actual_pixel_position[counter])))
        counter += 1

# print(error)
# print((img_width * resize_value * len(source_images) * templates_per_image))
# error = (error / (img_width * resize_value * len(source_images) * templates_per_image)) * 100

print("There is {}% error.".format(error))
# ---------------------------------------------------------------------------------------------------------------------- #


# match_matrix_and_location = find_pixel_dx(processedImg, processedTemplate)
#
# print("Numpy array object {}.\n\nStarts at pixel no. {} on x axis.".format(match_matrix_and_location[0],
#                                                                            match_matrix_and_location[1]))

# Close the window when the user presses the ESC key
# while True:
#     if cv.waitKey(0) == 27:
#         cv.destroyAllWindows()
#         break
