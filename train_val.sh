#!/bin/bash


echo "$(date) Training started"

num_train=$1
log_file=$2
batch_size=$3

if [ -z $batch_size ]; then
	echo "No batch_size was provided. Set default to 32"
	batch_size=32
fi

if [ -z $log_file ]; then
	log_file="../$(date '+%Y_%m_%d_%H_%M_%S').txt"
	echo "No log_file provided. Create ${log_file}"
	touch $log_file

else 

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
	python3 train.py --batch-size $batch_size --device 0 --name "${log_file}_${num_train}" --log_file $log_file
done

echo "$(date) Training finished"
if [ ! -z $log_file ]; then
	echo "Saved logs to ${log_file}"
fi