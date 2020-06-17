# Problem definition
This project's goal is to create a system that will efficiently perform **Optical Odometry** using a **Scene Matching Correlator**, for UAV guidance.
The finished program will take two images (one large source image, and one smaller template image) and should successfully be able to :
1. Transform the template image, to match the orrientation of the source image. This inludes, rotational and perspective tranformations.
2. Find the location of the template image inside the source image. 
3. Calculate the vertical and horizontal pixel displacement and convert it to coordinate difference

Note : Before the program is considered operational, it should be tested in various datasets, that will include added noise and tranformations

# Project resources

### Digital Scene Matching Area Correlator

 - [DSMAC : Image Processing for Tomahawk scene matching](https://www.jhuapl.edu/Content/techdigest/pdf/V15-N03/15-03-Irani.pdf)
 
#
### Visual Odometry

- [Visual Odometry and Visual SLAM Overview](https://link.springer.com/article/10.1007/s40903-015-0032-7)
- [Lucas-Kanade Optical Flow : OpenCV](https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_video/py_lucas_kanade/py_lucas_kanade.html)

#
### Tools and Methods

- [Template Matching : OpenCV](https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_template_matching/py_template_matching.html)

## Setup guide