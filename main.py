import cv2 as cv
import numpy as np
import os
import random

# Set image directory
images_directory = 'satellite_images'

# Append each image path into a list
images_paths = [os.path.join(images_directory, image_path) for image_path in os.listdir(images_directory)]

# ---------------------------------------Image Processing-------------------------------------- #
def process_image(img):

    # Resize image
    res_img = cv.resize(img, None, fx=0.5, fy=0.5, interpolation=cv.INTER_CUBIC)

    # Convert to grayscale
    gray_image = cv.cvtColor(res_img, cv.COLOR_BGR2GRAY)

    # Binarize image

    # bin_img = cv.threshold(gray_image, 120, 255, cv.THRESH_BINARY)
    (thresh, im_bw) = cv.threshold(gray_image, 128, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)

    # Translate image : shift its location

    # Translating the image randomly in the x axis
    translate_x = random.randrange(-rows, rows + 1)
    tf_matrix = np.float32([[1, 0, translate_x], [0, 1, 0]])
    shifted_img = cv.warpAffine(gray_image, tf_matrix, (cols, rows))


    # Try to match the shifted image to the original(gray) one
    # while not (shifted_img == gray_image).all():

# --------------------------------------------------------------------------------------------- #

# Read the image
for image in images_paths:
    img = cv.imread(images_paths[0])
    rows, cols, channels = img.shape


# Display images
cv.imshow('Original Image', img)
cv.imshow('Processed Image', im_bw)


# Close the window when the user presses the ESC key
while True:
    if cv.waitKey(0) == 27:
        cv.destroyAllWindows()
        break

