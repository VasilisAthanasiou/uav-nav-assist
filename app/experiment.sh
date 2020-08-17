#!/bin/bash

for ((dist = 50; dist <= 300; dist+=50))
do
	eval python main.py homing -m GOOD -v datasets/flight-video/flight-footage.mpg -t datasets/targets/close.jpg -d $dist -w y
done
