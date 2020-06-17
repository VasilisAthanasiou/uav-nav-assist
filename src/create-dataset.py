# This script is used to generate experiment datasets
# Should fix weird hardcoded path strings
#
#
import cv2 as cv
import os

source_directory = '../datasets/sources/source-less-features/source'
templates_directory = '../datasets/templates/templates-less-features/images'
crop_height = 200
crop_width = 200

# Open text file that will include image name and most left pixel
file = open("../datasets/templates/templates-less-features/less-features-loc.txt", "w")

# Append each image path into a list
source_paths = [os.path.join(source_directory, image_path) for image_path in os.listdir(source_directory)]

source_paths.sort()

images = []
for image_path in source_paths:
    images.append(cv.imread(image_path))


counter = 0
for img in images:
    img_height, img_width, _ = img.shape
    for most_top_pixel in range(0, img_height - crop_height, crop_height):
        for most_left_pixel in range(0, img_width - crop_width, crop_width):
            cropped_img = img[most_top_pixel:most_top_pixel + crop_height, most_left_pixel:most_left_pixel + crop_width]
            print(cropped_img.shape)
            cv.imwrite('{}/{}.png'.format(templates_directory, counter), cropped_img)
            file.write("{},{},\n".format(most_left_pixel, most_top_pixel))
            counter += 1





