import numpy as np
import cv2 as cv
import os

source_directory = 'datasets/sources/source_diverse/images'
templates_directory = 'datasets/templates/templates-diverse/images'
crop_width = 200

# Open text file that will include image name and most left pixel
file = open("datasets/templates/templates-diverse/dataset-diverse-loc.txt", "w")

# Append each image path into a list
source_paths = [os.path.join(source_directory, image_path) for image_path in os.listdir(source_directory)]

source_paths.sort()

# Crop each image into Xpx width parts
images = []
for image_path in source_paths:
    images.append(cv.imread(image_path))


counter = 0
for img in images:
    img_height, img_width, _ = img.shape
    for most_left_pixel in range(img_width - crop_width):
        cropped_img = img[0:img_height, most_left_pixel:most_left_pixel + crop_width]
        print(cropped_img.shape)
        cv.imwrite('/home/haistudent/PycharmProjects/diphaiopencv/datasets/templates/templates-diverse/images/{}.png'.format(counter), cropped_img)
        file.write("{}\n".format(most_left_pixel))
        counter += 1









