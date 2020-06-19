# Abstract
This project's uses OpenCV's Template Matching to match one satellite image and one UAV image and find where the image taken by the drone is relative to the satellite image

Note : This program has been tested on a varriaty of images, containing different amount of features, noise and blur.

# Project Guide
- datasets : Contains a sources and a templates directory. Images from these directories can be linked inside main.py to produce experimentS. Sources contain "satellite" images taken from Google Maps, and some of them were edited in GIMP to add noise to them like clouds and Gaussian blur
- report : Contains images, plots, and documents that describe the project's use case and analizes it's performance from data produced by src/create-dataset.py and evaluated in src/main.py
- src : Contains the source code of the project. The main.py file reads two images and calls `find_target()`, which uses `cv2.matchTemplate()` to find where the template image is located on the source image. The sensed locations are stored in a variable, checked against a txt file located at datasets/templates/'template dataset name' and evaluated using the `evaluate()` method. Finally the results are plotted.
The create-dataset.py file uses a sources, templates, and actual-location.txt path and crop the all the images in source into smaller 200x200 pixel images. While the images are being cropped, the top-left coordinate of the sub-image is written in the actual-location.txt file, so it can be used in `main.py::evaluate()`

# Project resources

### Digital Scene Matching Area Correlator

 - [DSMAC : Image Processing for Tomahawk scene matching](https://www.jhuapl.edu/Content/techdigest/pdf/V15-N03/15-03-Irani.pdf)
 
#
### Template Matching

- [Template matching wiki](https://en.wikipedia.org/wiki/Template_matching)
- [Template matching OpenCV](https://docs.opencv.org/2.4/doc/tutorials/imgproc/histograms/template_matching/template_matching.html)
- [Template matching source code](https://github.com/opencv/opencv/blob/master/modules/imgproc/src/templmatch.cpp)
