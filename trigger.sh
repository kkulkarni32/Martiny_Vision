#!/bin/bash

FLAG=true
while true
do
	if $FLAG;
	then
		echo 'triggering capture image'
		python3 Capture_Image.py
		FLAG=false
	fi
	
	echo 'triggering shadow detection'
	python3 BDRAR/infer.py 

	echo 'waiting 15 mins for next trigger' 
	sleep 900
done
