import torch
import os
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np

class MetricEvaluator():
    def __init__(self, out_path, task_dict):
        self.out_path = out_path

        self.taw_probabilities = []
        self.taw_labels = []

        self.tag_probabilities = []
    
        self.tag_labels = []

        self.task_dict = task_dict

        self.revesed_task_dict = {}
        for task_id, classes in self.task_dict.items():
            for class_label in classes:
                self.revesed_task_dict[class_label] = task_id


    def update(self, original_labels, labels, tag_probabilities , taw_probabilities):
        self.taw_probabilities.append(taw_probabilities)
        self.taw_labels.append(labels)

        self.tag_probabilities.append(tag_probabilities)
 
        self.tag_labels.append(original_labels)
    

    def get(self, verbose):
        self.taw_probabilities = torch.cat(self.taw_probabilities)
        self.taw_labels = torch.cat(self.taw_labels).cpu().numpy()

        self.tag_probabilities= torch.cat(self.tag_probabilities)
 
        self.tag_labels = torch.cat(self.tag_labels).cpu().numpy()

        taw_acc = accuracy_score(self.taw_labels, torch.max(self.taw_probabilities, axis = 1)[1].cpu().numpy())
        tag_acc = accuracy_score(self.tag_labels, torch.max(self.tag_probabilities, axis = 1)[1].cpu().numpy())
 

        if verbose:
            print(" - task agnostic accuracy : {}".format(tag_acc))
            print(" - task aware accuracy: {}".format(taw_acc))

        return taw_acc,   tag_acc 
    


    def prediction_ditribution_over_heads(self, current_training_task, test_id):
        print()
        print("Distribution tag over heads: test on task {}".format(test_id))

        out_index = torch.argmax(self.tag_probabilities, axis = 1).cpu().numpy()
        head_index= np.array([self.revesed_task_dict[out_idx] for out_idx in list(out_index)])

        head_value = torch.max(self.tag_probabilities, axis = 1)[0].cpu().numpy()

        current_heads = list(range(current_training_task+1))
        n_pred = head_index.shape[0]

        for item in current_heads:
            mask = (head_index==item)
            count  = mask.sum()
            if count > 0:
                avg_prob= head_value[mask].mean()
                print(" - head {} chosen with a percentage {:.1%}. Average confidence on the class {:.2f}". format(item, count /n_pred, avg_prob))
            else:
                print(" - head {} never chosen". format(item))