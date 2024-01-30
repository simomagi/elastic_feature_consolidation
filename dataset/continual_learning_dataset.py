from torch.utils.data import Subset
import numpy as np

class ContinualLearningDataset():
    def __init__(self, dataset, task_dictionary,  
                    n_task, n_class_first_task, class_per_task,
                    total_classes, valid_size=0.0, train=True):
        
        self.dataset = dataset
        self.train = train 
        self.valid_size = valid_size 
        self.task_dictionary = task_dictionary 
        self.n_class_first_task = n_class_first_task
        self.class_per_task = class_per_task
        self.n_task = n_task
        self.len_dataset = len(dataset)
        self.total_classes = total_classes

    def collect(self):
        if self.train or self.valid_size > 0:
            train_indices_list = [[] for _ in range(self.n_task)]  # List of list, each list contains indices in the entire dataset to accumulate for the task
            val_indices_list = [[] for _ in range(self.n_task)]
            valid_size_per_class = int((self.len_dataset/self.total_classes) * self.valid_size)
            train_size_per_class = int((self.len_dataset/self.total_classes)) - valid_size_per_class
            
            for cc in range(self.total_classes):
                current_class_indices = np.where(np.array(self.dataset.targets) == cc)[0]
                train_indices = current_class_indices[:train_size_per_class]
                val_indices  = current_class_indices[train_size_per_class:]
                for task_id, task_classes in self.task_dictionary.items():
                    if cc in task_classes:
                        train_indices_list[task_id].extend(list(train_indices))
                        val_indices_list[task_id].extend(list(val_indices))
                        
            
            cl_train_dataset = [Subset(self.dataset, ids)  for ids in train_indices_list]
            cl_train_sizes = [len(ids) for ids in train_indices_list]

            cl_val_dataset = [Subset(self.dataset, ids)  for ids in val_indices_list]
            cl_val_sizes = [len(ids) for ids in val_indices_list]
  
            return cl_train_dataset, cl_train_sizes, cl_val_dataset, cl_val_sizes
        
        else:
            test_indices_list = [[] for _ in range(self.n_task)] 
            for cc in range(self.total_classes):
                test_indices  = np.where(np.array(self.dataset.targets) == cc)[0]
                for task_id, task_classes in self.task_dictionary.items():
                    if cc in task_classes:
                        test_indices_list[task_id].extend(list(test_indices))
            
            cl_test_dataset = [Subset(self.dataset, ids)  for ids in test_indices_list]
            cl_test_sizes =[len(ids) for ids in test_indices_list]

            return cl_test_dataset, cl_test_sizes, None, None