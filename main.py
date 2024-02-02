

from utilities.generic_utils import experiment_folder, result_folder, \
                            get_task_dict, seed_everything, rollback_model, \
                            store_model, store_valid_loader, get_class_per_task, remap_targets 
from utilities.parse_utils import get_args
from utilities.matrix_logger import Logger
from torch.utils.data.dataloader import DataLoader

# approaches 
from continual_learning.ElasticFeatureConsolidation import ElasticFeatureConsolidation

# dataset 
from dataset.continual_learning_dataset import ContinualLearningDataset
from dataset.dataset_utils import get_dataset 
import sys 

from utilities.summary_logger import SummaryLogger
import os 
from copy import deepcopy
import math
 

if __name__ == "__main__":
    
    # args
    args, all_args, non_default_args, all_default_args = get_args()
    
    print(args.outpath)
    if not os.path.exists(args.outpath):
        os.mkdir(args.outpath)
        
    # device
    device = "cuda:" + str(args.device)
     
    # if True create folder exp_folder, else create folder with the name of the approach
    dev_mode = False

    # generate output folder and result folders
    out_path, exp_name = experiment_folder(args.outpath, dev_mode, args.approach)
 

    print("Current Seed {}".format(args.seed))
    
    # fix seed
    seed_everything(seed=args.seed)
    
    """
    Dataset Preparation
    """
    train_set, test_set, total_classes = get_dataset(args.dataset, args.data_path)
    
    # mapping between classes and shuffled classes and re-map dataset classes for different order of classes 
    # for imagenet-1k we use the order of classes of DER  https://github.com/Rhyssiyan/DER-ClassIL.pytorch/blob/main/inclearn/datasets/dataset.py 
    train_set, test_set, label_mapping = remap_targets(train_set, test_set, total_classes, args.dataset)
    
    # class_per_task: number of classes not in the first task, if the first is larger, otherwise it is equal to total_classes/n_task
    class_per_task = get_class_per_task(args.n_class_first_task, total_classes, args.n_task)
    
    # task_dict = {task_id: list_of_class_ids}
    task_dict = get_task_dict(args.n_task, total_classes, class_per_task, args.n_class_first_task)   
    
    print("Dataset: {}, N task: {}, Large First Task Classes: {}, Classes Per Task : {}".format(args.dataset,
                                                                                                  args.n_task,
                                                                                                  args.n_class_first_task,
                                                                                                  class_per_task))
        
    """
    Generate Subset For Each Task
    """
    cl_train_val = ContinualLearningDataset(train_set, task_dict,  
                                            args.n_task, args.n_class_first_task, 
                                            class_per_task,total_classes,
                                            valid_size=args.valid_size, train=True)

    train_dataset_list, train_sizes, val_dataset_list, val_sizes = cl_train_val.collect()

 
    cl_test = ContinualLearningDataset(test_set, task_dict,  
                                       args.n_task, args.n_class_first_task, 
                                       class_per_task,total_classes,
                                       train=False)

    test_dataset_list, test_sizes, _, _  = cl_test.collect()
    test_loaders = [DataLoader(test, batch_size=args.batch_size*4, shuffle=False, num_workers=args.nw) for test in test_dataset_list]
    
    if args.valid_size > 0:
        print("Creating Validation Set")
        train_loaders = [DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=args.nw) for train in train_dataset_list]
        valid_loaders = [DataLoader(valid, batch_size=args.batch_size*4, shuffle=False, num_workers=args.nw) for valid in val_dataset_list]
    
    else:
        print("Not using Validation")
        train_loaders = [DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=args.nw) for train in train_dataset_list]
        valid_loaders = test_loaders
 

    
    """
    Logger Init
    """
    logger = Logger(out_path=out_path, n_task=args.n_task, task_dict=task_dict, test_sizes=test_sizes)
    result_folder(out_path, "logger")
 
 
    if args.approach == "efc":
        approach = ElasticFeatureConsolidation(args=args, device=device, 
                                                  out_path=out_path, class_per_task=class_per_task,
                                                  task_dict=task_dict)         
  
    else:
        sys.exit("Approach not Implemented")
 
    for task_id, train_loader in enumerate(train_loaders):


        if  task_id == 0 and args.firsttask_modelpath != "None":
            # useful for retrieving a pre-trained model, it must be saved in  a path {args.firsttask_modelpath}/{args.dataset}_{args.seed}/0_model.pth"
            approach.pre_train(task_id, train_loader,  valid_loaders[task_id])
 
            print("Loading model from path {}".format(os.path.join(args.firsttask_modelpath, "{}_seed_{}".format(args.dataset, args.seed), "0_model.pth")))

            rollback_model(approach, os.path.join(args.firsttask_modelpath, "{}_seed_{}".format(args.dataset, args.seed),"0_model.pth"), device, name=str(task_id))
  
            epoch = 100
        else:
 
            
            print("#"*40 + " --> TRAINING HEAD {}".format(task_id))

            """
            Pre-train
            """

            approach.pre_train(task_id)

            best_taw_accuracy,  best_tag_accuracy = 0, 0
            best_loss = math.inf 
            
            
            """
            Main train Loop
            """
                    
            for epoch in range(approach.total_epochs):
                print("Epoch {}/{}".format(epoch, approach.total_epochs))

                approach.train(task_id, train_loader, epoch, approach.total_epochs)
                taw_acc, tag_acc, test_loss  = approach.eval(task_id, task_id, valid_loaders[task_id], epoch,  verbose=True)
                approach.reduce_lr_on_plateau.step()
                    
        """
        Test Final Model on all the encountered tasks
        """
    
        store_model(approach, out_path, name=str(task_id))
 
        for test_id in range(task_id + 1):
            acc_taw_value, acc_tag_value, _,  = approach.eval(task_id, test_id, test_loaders[test_id], epoch,  verbose=False)                                                                                                            
            logger.update_accuracy(current_training_task_id=task_id, test_id=test_id, acc_taw_value=acc_taw_value,  acc_tag_value=acc_tag_value)
            if test_id < task_id:
                logger.update_forgetting(current_training_task_id=task_id, test_id=test_id)
            logger.print_latest(current_training_task_id=task_id, test_id=test_id)

 
        """
        Post Training
        """
        logger.compute_average()
        logger.print_file()
        approach.post_train(task_id=task_id, trn_loader=train_loader)

    
            
        
    summary_logger = SummaryLogger(all_args, all_default_args, args.outpath)
    summary_logger.update_summary(exp_name, logger)
    store_valid_loader(out_path, valid_loaders, False)



