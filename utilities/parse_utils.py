import argparse
from ast import parse
import sys
    
def get_args():
    parser = argparse.ArgumentParser()

    """
    Structural hyperparams 
    """
    parser.add_argument("--approach", type=str,default="efc", choices=["efc"])
    parser.add_argument("--outpath", "-op",default="./", type=str) 
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--nw", type=int, default=4, help="num workers for data loader")
 
    """
    Training hyperparams 
    """
    parser.add_argument("--epochs_first_task", type=int, default=100,help="epochs first task, should be changed to 160 for imagenet-subset and imagenet-1k")
    parser.add_argument("--epochs_next_task",type=int, default=100, help="epochs next task")
    parser.add_argument("--batch_size", type=int, default=64, help="batch size of data, should be changed to 256 for imagenet-1k")
    parser.add_argument("--device", type=int, default=0)
    
    "Dataset Settings"
    parser.add_argument("--dataset", type=str, default="cifar100", choices=["cifar100","tiny-imagenet", "imagenet-subset", "imagenet-1k"], help="dataset to use") 
    parser.add_argument("--data_path",type=str, default="/cl_data",help="path where imagenet subset, imagenet-1k, tiny-imagenet are saved")
    
    parser.add_argument("--n_class_first_task", type=int, default=50, help="if greater than -1 use a larger number of classes for the first class, n_task include this one")
    parser.add_argument("--n_task", type=int, default=6, help="number of task")
    parser.add_argument("--valid_size", type=float, default=0.0, help="percentage of train for validation set, default not use validation set")
    """
    Network Params
    """
    parser.add_argument("--backbone", type=str, default="resnet18", choices=["resnet18"])
    parser.add_argument("--firsttask_modelpath", type=str, default="None", help="specify model path if start from a pre-trained model after first task")

    """
    EFC  Hyperparams
    """
    parser.add_argument("--efc_lamb", default=10.0, type=float, help="lambda associated to EFM")  
    parser.add_argument("--efc_protobatchsize", type=int, default=64, help="batch size of prototypes, should be changed to 256 for imagenet-1k")
    parser.add_argument("--efc_damping", type=float, default=0.1, help="damping hyperparameter, eta in the paper")
    parser.add_argument("--efc_protoupdate",type=float, default=0.2, help=["update proto using  gaussian sigma (in the paper) 0.2, if -1 it is specified it does not update the prototype"] )
 
    args = parser.parse_args()

    non_default_args = {
            opt.dest: getattr(args, opt.dest)
            for opt in parser._option_string_actions.values()
            if hasattr(args, opt.dest) and opt.default != getattr(args, opt.dest)
    }

    default_args = {
            opt.dest: opt.default
            for opt in parser._option_string_actions.values()
            if hasattr(args, opt.dest)
    }

    all_args = vars(args)    
    
    return args, all_args, non_default_args, default_args