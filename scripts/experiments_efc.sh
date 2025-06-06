#!/bin/bash

filename=$0
filename=${filename%.*}
mkdir -p $1/$filename
 

data_dir=./cl_data
# Warm Start Experiments

python -u   main.py -op $1/$filename  --dataset cifar100 --n_task 11 --n_class_first_task 50 --approach efc --nw 12 --seed 0 --epochs_first_task 100  --epochs_next_task 100

python -u   main.py -op $1/$filename  --dataset cifar100 --n_task 21 --n_class_first_task 40 --approach efc  --nw 12 --seed 0 --epochs_first_task 100  --epochs_next_task 100

 
python -u   main.py -op $1/$filename  --dataset tiny-imagenet  --n_task 11 --n_class_first_task 100 --data_path  $data_dir --approach efc   --nw 12 --seed 0 --epochs_first_task 100 --epochs_next_task 100

python -u   main.py -op $1/$filename  --dataset tiny-imagenet  --n_task 21 --n_class_first_task 100 --data_path $data_dir --approach efc  --nw 12 --seed 0 --epochs_first_task 100  --epochs_next_task 100


python -u   main.py -op $1/$filename  --dataset imagenet-subset --n_task 11 --n_class_first_task 50 --data_path $data_dir --approach efc --nw 12 --seed 0 --epochs_first_task 160  --epochs_next_task 100

python -u   main.py -op $1/$filename  --dataset imagenet-subset --n_task 21 --n_class_first_task 40 --data_path $data_dir  --approach efc --nw 12 --seed 0 --epochs_first_task 160  --epochs_next_task 100



# Cold Start Exemperiments 

python -u   main.py -op $1/$filename  --dataset cifar100 --n_task 10 --n_class_first_task 10   --approach efc  --nw 12 --seed 0 --epochs_first_task 100  --epochs_next_task 100

python -u   main.py -op $1/$filename  --dataset cifar100 --n_task 20 --n_class_first_task 5    --approach efc  --nw 12 --seed 0 --epochs_first_task 100  --epochs_next_task 100

 
python -u   main.py -op $1/$filename  --dataset tiny-imagenet  --n_task 10  --n_class_first_task 20 --data_path $data_dir  --approach efc  --nw 12 --seed 0 --epochs_first_task 100  --epochs_next_task 100

python -u   main.py -op $1/$filename  --dataset tiny-imagenet  --n_task 20 --n_class_first_task 10  --data_path $data_dir  --approach efc  --nw 12 --seed 0 --epochs_first_task 100  --epochs_next_task 100


python -u   main.py -op $1/$filename  --dataset imagenet-subset --n_task 10 --n_class_first_task 10 --data_path $data_dir --approach efc  --nw 12 --seed 0 --epochs_first_task 160  --epochs_next_task 100

python -u   main.py -op $1/$filename  --dataset imagenet-subset --n_task 20 --n_class_first_task 5 --data_path $data_dir  --approach efc  --nw 12 --seed 0 --epochs_first_task 160  --epochs_next_task 100

python -u   main.py -op $1/$filename  --dataset domainnet --n_task 6 --n_class_first_task 100 --data_path $data_dir --approach  efc  --nw 12 --seed 0 --epochs_first_task 160  --epochs_next_task 100

