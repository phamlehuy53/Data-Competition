#!/bin/bash


echo "$(date) Training started"

num_train=$1

for (( c=1; c<=num_train; c++))
do	
	echo "$(date) Iter $c-th"
	python3 train.py --batch-size 64 --device 0 --name num_train 
done

echo "$(data) Training finished"