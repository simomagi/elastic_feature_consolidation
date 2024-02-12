# Elastic Feature Consolidation For Cold Start Exemplar-Free Incremental Learning

![](images/EFC_overview_split.png)

This repository contains all code needed to reproduce the experimental results for the paper:  
_**Elastic Feature Consolidation For Cold Start Exemplar-Free Incremental Learning**_   
*Simone Magistri, Tomaso Trinci, Albin Soutif, Joost van de Weijer, Andrew D. Bagdanov*  
[(ICLR2024)](https://openreview.net/forum?id=7D9X2cFnt1)
[(arXiv)](https://arxiv.org/abs/2402.03917)

# Cite

If this code is useful in your research, please cite it as follows:

```
@inproceedings{
magistri2024elastic,
title={Elastic Feature Consolidation For Cold Start Exemplar-Free Incremental Learning},
author={Simone Magistri and Tomaso Trinci and Albin Soutif and Joost van de Weijer and Andrew D. Bagdanov},
booktitle={The Twelfth International Conference on Learning Representations},
year={2024},
url={https://openreview.net/forum?id=7D9X2cFnt1}
}
```

# Setting up the Conda environment

To run the EFC code you must create an Anaconda environment from the `environment.yml` file and activate it:

```
conda env create -n EFC -f environment.yml 
conda activate EFC
```
# Project Description

This codebase is inspired by [FACIL](https://github.com/mmasana/FACIL) and is structured as follows:

- `main.py`: This script is used to run the experiments.

### Incremental Learning Modules:

- `continual_learning/IncrementalApproach.py`: This is the base class for `ElasticFeatureConsolidation`. It allows you to set optimization settings, such as the scheduler, learning rate, and optimizer, using the `OptimizerManager` class.

- `continual_learning/ElasticFeatureConsolidation.py`: This module implements Elastic Feature Consolidation. This class inherits from the `IncrementalApproach` class.

### Utility Modules:

- `continual_learning/utils/OptimizerManager.py`: This class sets the optimizer for running the experiment.

- `continual_learning/utils/empirical_feature_matrix.py`: This code is responsible for computing the Empirical Feature Matrix.

- `continual_learning/utils/proto.py`: This module contains the prototype generator class.

- `continual_learning/models/BaseModel.py`: This module defines the incremental model.

- `continual_learning/utilities`: This directory contains various scripts to compute metrics. The `SummaryLogger` generates the `summary.csv` file, while the `Logger` class generates accuracy matrices for each task.

# Analyzing the Results

The results are stored in the path specified by the `-op` flag. A file named `summary.csv` will be generated, which contains the following performance metrics:

- `Per_step_taw_acc`: Per-step task-aware accuracy after each task.

- `Last_per_step_taw_acc`: Per-step task-aware accuracy after the last task.

- `Per_step_tag_acc`: Per-step task-agnostic accuracy after each task.

- `Last_per_step_tag_acc`: Per-step task-agnostic accuracy after the last task (as described in the left formula in Equation 16 in the main paper).

- `Average_inc_acc`: Average incremental accuracy (as described in the right formula in Equation 16 in the main paper).

 
# Main Command-Line Arguments

Use the following command-line arguments to configure the behavior of the code:

- `-op`: The folder path where results are stored. The name of the experiment is randomly generated.
- `--nw`: Number of workers for data loaders.
- `--epochs_first_task`: Number of epochs for the first task (default=100).
- `--epochs_next_task`: Number of epochs for tasks after the first one (default=100).
- `--seed`: Random seed (default=0).
- `--device`: GPU device to use (default=0).
- `--n_task`: Number of tasks, including the first task.
- `--n_class_first_task`: Number of classes in the first task.
- `--efc_lamb`: Lambda value associated with the empirical feature matrix (default=10).
- `--efc_damping`: Eta value as described in the main paper (default=0.1).
- `--efc_protoupdate`: Whether to update the prototype using the empirical feature matrix with a specified sigma for the Gaussian kernel. If set to -1, no prototype update is performed (default=0.2).
- `--dataset`: Dataset name (default=cifar100).
- `--data_path`: The data folder where imagenet subset and tiny-imagenet datasets are stored.
- `--firsttask_modelpath`: Start the training from a checkpoint for the first task if available.


# Running the code for CIFAR-100 experiments - Warm-Start (WS)

The default hyperparameters are the ones used to compute the Table 1 in the main paper.


1. 10 Step

```
python -u   main.py -op ./ws_cifar100_10step --dataset cifar100 --n_task 11 --n_class_first_task 50 --approach efc --nw 12 --seed 0 --epochs_first_task 100  --epochs_next_task 100

```

2. 20 Step

```
python -u   main.py -op ./ws_cifar100_20step --dataset cifar100 --n_task 21 --n_class_first_task 40 --approach efc  --nw 12 --seed 0 --epochs_first_task 100  --epochs_next_task 100

```

# Running the code for CIFAR-100 experiments - Cold-Start (CS)

The default hyperparameters are the ones used to compute the Table 1 in the main paper.


1. 10 Step

```
python -u   main.py -op ./cs_cifar100_10step --dataset cifar100 --n_task 10 --n_class_first_task 10 --data_path ./cl_data --approach efc  --nw 12 --seed 0 --epochs_first_task 100  --epochs_next_task 100

```

2. 20 Step

```
python -u   main.py -op ./cs_cifar100_20step   --dataset cifar100 --n_task 20 --n_class_first_task 5 --data_path ./cl_data --approach efc  --nw 12 --seed 0 --epochs_first_task 100  --epochs_next_task 100
```



# Running the Tiny-Imagenet and ImageNet-Subset experiments

The commands are similar, with the only difference being the data-folder "cl_data," where both datasets are downloaded, should be specified.

Here the 10-step and 20-step scenario **Warm Start** (WS) for Tiny-ImageNet and ImageNet-Subset.

```
python -u   main.py -op ./ws_tinyimagenet_10step  --dataset tiny-imagenet  --n_task 11  --n_class_first_task 100 --data_path ./cl_data --approach efc   --nw 12 --seed 0 --epochs_first_task 100  --epochs_next_task 100
```

```
python -u   main.py -op ./ws_imagenetsubset_10task --dataset imagenet-subset --n_task 11 --n_class_first_task 50 --data_path ./cl_data --approach efc --nw 12 --seed 0 --epochs_first_task 160  --epochs_next_task 100
```

```
python -u   main.py -op ./ws_tinyimagenet_20step  --dataset tiny-imagenet  --n_task 21 --n_class_first_task 100 --data_path ./cl_data --approach efc  --nw 12 --seed 0 --epochs_first_task 100  --epochs_next_task 100
```

```
python -u   main.py -op ./ws_imagenetsubset_20task --dataset imagenet-subset --n_task 21 --n_class_first_task 40 --data_path ./cl_data --approach efc --nw 12 --seed 0 --epochs_first_task 160  --epochs_next_task 100
```

Here the 10-step and 20-step scenario **Cold Start** (CS) for Tiny-ImageNet and ImageNet-Subset.

```
python -u   main.py -op ./cs_tinyimagenet_10step  --dataset tiny-imagenet  --n_task 10  --n_class_first_task 20 --data_path ./cl_data --approach efc  --nw 12 --seed 0 --epochs_first_task 100  --epochs_next_task 100
```

```
python -u   main.py -op ./cs_imagenetsubset_10step --dataset imagenet-subset --n_task 10 --n_class_first_task 10 --data_path ./cl_data --approach efc  --nw 12 --seed 0 --epochs_first_task 160  --epochs_next_task 100
```

```
python -u   main.py -op ./cs_tinyimagenet_20step  --dataset tiny-imagenet  --n_task 20 --n_class_first_task 10  --data_path ./cl_data --approach efc  --nw 12 --seed 0 --epochs_first_task 100  --epochs_next_task 100
```

```
python -u   main.py -op ./cs_imagenetsubset_20step --dataset imagenet-subset --n_task 20 --n_class_first_task 5 --data_path ./cl_data --approach efc  --nw 12 --seed 0 --epochs_first_task 160  --epochs_next_task 100
```



In the bash file `experiments.sh` all the experiments for all the scenarios can be run. 

# License

Please check the MIT license that is listed in this repository.






