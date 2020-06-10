import numpy as np
import cv2 as cv
import os

images_directory = 'satellite_images'
templates_directory = 'templates-x'
crop_width = 100

# Open text file that will include image name and most left pixel
file = open("dataset-img-info.txt", "w")

# Append each image path into a list
images_paths = [os.path.join(images_directory, image_path) for image_path in os.listdir(images_directory)]


# Crop each image into 100px width parts
images = []
for image_path in images_paths:
    images.append(cv.imread(image_path))

img_height, img_width, _ = images[0].shape

counter = 0
for img in images:
    for most_left_pixel in range(img_width - crop_width):
        cropped_img = img[0:img_height, most_left_pixel:most_left_pixel + crop_width]
        print(cropped_img.shape)
        #cv.imwrite('/home/haistudent/PycharmProjects/diphaiopencv/templates-x/temp-x-{}.png'.format(counter), cropped_img)
        file.write("{}\n".format(most_left_pixel))
        counter += 1









