#!/bin/bash

for ((dist = 50; dist <= 300; dist+=50))
do
	eval python main.py homing -m GOOD -v app/datasets/flight-video/flight-footage.mpg -t app/datasets/targets/close.jpg -d $dist

done
