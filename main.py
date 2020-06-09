import cv2 as cv
import numpy as np
import os
import random

# Set image directory
images_directory = 'satellite_images'
templates_directory = 'templates'

# Append each image path into a list
images_paths = [os.path.join(images_directory, image_path) for image_path in os.listdir(images_directory)]
templates_paths = [os.path.join(templates_directory, template_path) for template_path in os.listdir(templates_directory)]

# ---------------------------------------Image Processing-------------------------------------- #
def process_image(src_img):
    # Resize image
    res_img = cv.resize(src_img, None, fx=0.5, fy=0.5, interpolation=cv.INTER_CUBIC)

    # Convert to grayscale
    gray_image = cv.cvtColor(res_img, cv.COLOR_BGR2GRAY)

    # Remove noise
    #gray_image = cv.fastNlMeansDenoising(gray_image, gray_image, 13, 7, 21) # fastNlMeansDenoising is too slow for rt applications
    gray_image = cv.blur(gray_image, (5, 5), gray_image)

    # Binarize image

    # bin_img = cv.threshold(gray_image, 120, 255, cv.THRESH_BINARY)
    (thresh, bin_img) = cv.threshold(gray_image, 0, 255,  cv.THRESH_BINARY + cv.THRESH_OTSU)

    # Erode binary image
    kernel = np.ones((2, 2), np.uint8)
    bin_img = cv.morphologyEx(bin_img, cv.MORPH_OPEN, kernel)

    # Dilate binary image
    bin_img = cv.morphologyEx(bin_img, cv.MORPH_CLOSE, kernel)

    # Translate image : shift its location

    # Translating the image randomly in the x axis
    # translate_x = random.randrange(-rows, rows + 1)
    # tf_matrix = np.float32([[1, 0, translate_x], [0, 1, 0]])
    # shifted_img = cv.warpAffine(gray_image, tf_matrix, (cols, rows))

    # Try to match the shifted image to the original(gray) one
    # while not (shifted_img == gray_image).all():

    return bin_img


# --------------------------------------------------------------------------------------------- #


# Read the image
img = [cv.imread(image) for image in images_paths]
template = [cv.imread(template) for template in templates_paths]

# Display images
index = 1
processedImg = process_image(img[index])
processedTemplate = process_image(template[5])

cv.imshow('Original Image', img[index])
cv.imshow('Processed Image', processedImg)

cv.imshow('Template', template[5])
cv.imshow('Processed Template', processedTemplate)



#
def find_pixel_dx(sat_img, temp_img, start_x):  # (satellite, template, template random x coordinate inside satellite
    # Store the image array shapes              # image)
    sat_shapes = [x.shape for x in sat_img]
    sat_height, sat_width, _ = zip(*sat_shapes)

    temp_shapes = [x.shape for x in temp_img]
    temp_height, temp_width, _ = zip(*temp_shapes)

    for start_pixel in range(sat_width + 1):
        search_space = sat_img[0:temp_height, start_pixel:start_pixel + temp_width]


# Close the window when the user presses the ESC key
while True:
    if cv.waitKey(0) == 27:
        cv.destroyAllWindows()
        break
