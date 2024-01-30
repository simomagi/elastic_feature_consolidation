#!/bin/bash

filename=$0
filename=${filename%.*}
mkdir -p $1/$filename
 


# Warm Start Experiments

python -u   main.py -op $1/$filename  --dataset cifar100 --n_task 11 --n_class_first_task 50 --approach efc --epochs 100 --nw 12 --seed 0

python -u   main.py -op $1/$filename  --dataset cifar100 --n_task 21 --n_class_first_task 40 --approach efc --epochs 100 --nw 12 --seed 0

 
python -u   main.py -op $1/$filename  --dataset tiny-imagenet  --n_task 11  --n_class_first_task 100 --data_path ./cl_data --approach efc --epochs 100 --nw 12 --seed 0

python -u   main.py -op $1/$filename  --dataset tiny-imagenet  --n_task 21 --n_class_first_task 100 --data_path ./cl_data --approach efc --epochs 100 --nw 12 --seed 0


python -u   main.py -op $1/$filename  --dataset imagenet-subset --n_task 11 --n_class_first_task 50 --data_path ./cl_data --approach efc --epochs 100 --nw 12 --seed 0

python -u   main.py -op $1/$filename  --dataset imagenet-subset --n_task 21 --n_class_first_task 40 --data_path ./cl_data --approach efc --epochs 100 --nw 12 --seed 0



# Cold Start Exemperiments 

python -u   main.py -op $1/$filename  --dataset cifar100 --n_task 10 --n_class_first_task 10 --data_path ./cl_data --approach efc --epochs 100 --nw 12 --seed 0

python -u   main.py -op $1/$filename  --dataset cifar100 --n_task 20 --n_class_first_task 5 --data_path ./cl_data --approach efc --epochs 100 --nw 12 --seed 0

 
python -u   main.py -op $1/$filename  --dataset tiny-imagenet  --n_task 10  --n_class_first_task 20 --data_path ./cl_data --approach efc --epochs 100 --nw 12 --seed 0

python -u   main.py -op $1/$filename  --dataset tiny-imagenet  --n_task 20 --n_class_first_task 10  --data_path ./cl_data --approach efc --epochs 100 --nw 12 --seed 0


python -u   main.py -op $1/$filename  --dataset imagenet-subset --n_task 10 --n_class_first_task 10 --data_path ./cl_data --approach efc --epochs 100 --nw 12 --seed 0

python -u   main.py -op $1/$filename  --dataset imagenet-subset --n_task 20 --n_class_first_task 5 --data_path ./cl_data --approach efc --epochs 100 --nw 12 --seed 0

