#!/bin/bash


echo "$(date) Training started"

num_train=$1
log_file=$2

if [ ! -z $log_file ]; then

	if [ ! -f log_file ]; then
		echo "File ${log_file} not existed, create new one!"
		touch $log_file
	else
		echo "File ${log_file} existed, cleaned first!"
		rm $log_file
		touch $log_file
	fi
fi

for (( c=1; c<=num_train; c++))
do	
	echo "=========================================================="
	echo "$(date) Iter $c-th"
	python3 train.py --batch-size 64 --device 0 --name $num_train --log_file $log_file
done

echo "$(date) Training finished"
if [ ! -z $log_file ]; then
	echo "Saved logs to ${log_file}"
fi