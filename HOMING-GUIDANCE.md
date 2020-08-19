# Image Oriented Homing Guidance

## Abstract
**Image Oriented Homing Guidance** is a process that guides a UAV to a chosen target.

## Program guide:
- Tracker : Is used to identify and track an object from a given image. This is accomplished by first extracting and matching features from the target image and the video feed, using OpenCV's [ORB](https://medium.com/data-breach/introduction-to-orb-oriented-fast-and-rotated-brief-4220e8ec40cf) feature matching algorithm and then performing clustering on the matched features, in order to select the cluster with the largest number of matched features. The centroid of the selected cluster is the point towards which the UAV must orient it self to.    

## Project resources

- [ORB : Oriented FAST and Rotated BRIEF](https://medium.com/data-breach/introduction-to-orb-oriented-fast-and-rotated-brief-4220e8ec40cf)
- [FAST : Features from Accelerated Segment Test](https://medium.com/data-breach/introduction-to-fast-features-from-accelerated-segment-test-4ed33dde6d65)
- [BRIEF : Binary Robust Independent Elementary Features](https://medium.com/data-breach/introduction-to-brief-binary-robust-independent-elementary-features-436f4a31a0e6)
