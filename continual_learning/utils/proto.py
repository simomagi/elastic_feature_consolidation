import abc
import torch 
import numpy as np 
from copy import deepcopy
from torch.distributions import MultivariateNormal
import torch.nn.functional as F 
import os 


 
class ProtoManager(metaclass=abc.ABCMeta):
    def __init__(self, device, task_dict, batch_size, feature_space_size) -> None:
        self.device = device 
        self.task_dict = task_dict
        self.batch_size = batch_size
        self.feature_space_size = feature_space_size
        self.prototype = []
        self.variances = []
        self.class_label = []
 

    @abc.abstractmethod 
    def compute(self, model, loader, current_task):
        pass 

    @abc.abstractmethod 
    def perturbe(self, *args):
        pass 

    @abc.abstractmethod
    def update(self, *args):
        pass 




class  ProtoGenerator(ProtoManager):
    
    def __init__(self, device, task_dict, batch_size, out_path, feature_space_size) -> None:
        
        super(ProtoGenerator, self).__init__(device, task_dict, batch_size, feature_space_size)

        self.R = None 
        self.running_proto = None 
        self.running_proto_variance = []
        self.rank = None 
        self.out_path = out_path
        self.current_mean = None
        self.current_std = None 
        self.gaussians = {}
        self.rank_list = []

        
        
    def compute(self, model, loader, current_task):
        model.eval()
        
        features_list = []
        label_list = []
      
 

        with torch.no_grad():
            for images, labels in loader:
                images = images.to(self.device)
                labels = labels.type(dtype=torch.int64).to(self.device)
                _, features  = model(images)

                label_list.append(labels.cpu())
                features_list.append(features.cpu()) 
        
        label_list = torch.cat(label_list)
        features_list = torch.cat(features_list)
 
        for label in self.task_dict[current_task]:
            mask = (label_list == label)
            feature_classwise = features_list[mask]
           
        
            proto = feature_classwise.mean(dim=0)
            
           
            covariance = torch.cov(feature_classwise.T)  
    
            self.running_proto_variance.append(covariance)
            self.prototype.append(proto)
            self.class_label.append(label)
            self.gaussians[label] = MultivariateNormal(
                                                    proto.cpu(),
                                                    covariance_matrix=covariance+ 1e-5 * torch.eye(covariance.shape[0]).cpu(),
                                    )
       
             
        self.running_proto = deepcopy(self.prototype)
 
        
    def update_gaussian(self, proto_label, mean, var):
        self.gaussians[proto_label] = MultivariateNormal(
                                                    mean.cpu(),
                                                    covariance_matrix=var+ 1e-5 * torch.eye(var.shape[0]).cpu(),
                                    )
 
                                                             
             
    def perturbe(self,  current_task, protobatchsize=64):

        # list of number of classes seen before
        
        index = list(range(0, sum([len(self.task_dict[i]) for i in range(0, current_task)])))
        np.random.shuffle(index)
        
        proto_aug_label = torch.LongTensor(self.class_label)[index] 
            
        if len(self.running_proto) < protobatchsize:
            samples_to_add = protobatchsize - len(self.running_proto)  
            proto_aug_label = torch.cat([proto_aug_label, proto_aug_label.repeat(int(np.ceil(samples_to_add/len(self.running_proto))))[:samples_to_add]])
        else:
            proto_aug_label = proto_aug_label[:protobatchsize]


        proto_aug_label, _ = torch.sort(proto_aug_label)
        samples_to_generate = torch.nn.functional.one_hot(proto_aug_label).sum(dim=0)
        proto_aug = []
        for class_idx, n_samples in enumerate(samples_to_generate):
            if n_samples > 0:
                proto_aug.append(self.gaussians[class_idx].sample((n_samples,)))
        
        proto_aug = torch.cat(proto_aug, dim=0)
        n_proto = proto_aug.shape[0]
        shuffle_indexes = torch.randperm(n_proto)
        proto_aug = proto_aug[shuffle_indexes, :]
        proto_aug_label = proto_aug_label[shuffle_indexes]


        return proto_aug, proto_aug_label, n_proto  

    def update(self, *args):
        pass 