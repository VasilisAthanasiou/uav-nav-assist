# UAV Navigation Assist
UAV Navigation Assist is a set of tools that could be used to guide UAV's under certain scenarios.

## Disclamer
The software included on this repository is not meant for real life use. It was developed as an internship project to evaluate the developer's skills.

## Use Cases
1. Long Range Cruising:<br>
The travel_assist module corrects the UAV's Inertial Navigation System inaccuracy using a [DSMAC](https://en.wikipedia.org/wiki/TERCOM#DSMAC) algorithm based on OpenCV's [Template Matching](https://en.wikipedia.org/wiki/Template_matching#:~:text=Template%20matching%20is%20a%20technique,to%20detect%20edges%20in%20images.).
2. Target Locking and Homing:<br>
The image_homing_guidance module tracks a target on a video, given an image of said target, using Shi-Tomashi corner detector, ORB for feature description, and a custom clustering algorithm.

## How to run
- First off, you will need to create a Python 3 virtual environment that will have installed all the modules from the [requirements.txt](https://github.com/VasilisAthanasiou/uav-nav-assist/blob/master/requirements.txt) file. This program has been tested only on Python 3.7.7 but it could work on other versions of Python 3<br>
- If you wish to run the travel_assist module you need to do the following:<br>
1. Create and populate a directory under `app/datasets/target-assist/sources`. The directory must contain the satellite images that will be used as checkpoints for the flight.
2. To run a simulation, set the working directory to the app/ directory (where main.py is located) and execute the following command `python main.py travel-assist fast`.

- If you wish to run the image_homing_guidance module you need to:<br>
1. Set working directory to the app/ directory and execute the following:
```python main.py homing -c <camera URL> -i <webcam index for VideoCapture> -m <feature extraction method> -n <number of extracted features> -d <minimum distance between neighbors for clustering algorithm> -v <Video file path> -t <Path of target image> -w <Either yes or no. Determines whether log data to app/dataset/experiment-results.txt or not> -k <Clustering method. Either RoiCluster or MeanShift>```
<br>Where `feature extraction method` can be either 'SURF', 'GOOD' (Shi-Tomashi) or 'ORB'. Running `python main.py homing` without options will execute the modules with default values.
2. If you wish to run an experiment execute the following while on app/ directory:<br>
```chmod -x ./experiment.sh```<br>```./experiment.sh```<br>
You can edit `experiment.sh` to run a custom experiment. The results will be stored at `app/datasets/experiment-results.txt`

## References
Check out [HOMING-GUIDANCE.md](https://github.com/VasilisAthanasiou/uav-nav-assist/blob/master/HOMING-GUIDANCE.md) and [TRAVEL-ASSIST.md](https://github.com/VasilisAthanasiou/uav-nav-assist/blob/master/TRAVEL-ASSIST.md) for additional information for each module.


   

