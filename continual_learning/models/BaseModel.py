 
import torch
from .resnet18 import resnet18
from .resnet18_imagenet import resnet18_imagenet
from torch import nn
import sys 
import math
import torch.nn.functional as F 

         
class BaseModel(nn.Module):
    def __init__(self, backbone,dataset):
        super(BaseModel, self).__init__()
        self.backbone_type = backbone
        self.dataset = dataset
 
        if self.backbone_type == "resnet18" :
            if dataset == "cifar100":
                self.backbone = resnet18(avg_pool_size=4, pretrained=False)  
            elif dataset == "tiny-imagenet":
                self.backbone = resnet18(avg_pool_size=8, pretrained=False)
            elif dataset == "imagenet-subset" or self.dataset == "imagenet-1k": 
                self.backbone = resnet18_imagenet()

        else:
            sys.exit("Model Not Implemented")

        self.heads = nn.ModuleList()
    
    def get_feat_size(self):
        return self.backbone.feature_space_size

    def add_classification_head(self, n_out):
      
        self.heads.append(
            torch.nn.Sequential(nn.Linear(self.backbone.feature_space_size, n_out, bias=False)))

    
    def reset_backbone(self):

        if self.dataset == "cifar100":
            self.backbone = resnet18(avg_pool_size=4, pretrained=False)  
        elif self.dataset == "tiny-imagenet":
            self.backbone = resnet18(avg_pool_size=8, pretrained=False)  
        
        elif self.dataset == "imagenet-subset":
            self.backbone = resnet18_imagenet()
        

    def forward(self, x):
        results = {}
        features = self.backbone(x)
 
        for id, head in enumerate(self.heads):
            results[id] = head(features)
        
        return results, features
    
    
    def freeze_all(self):
        """Freeze all parameters from the model, including the heads"""
        for param in self.parameters():
            param.requires_grad = False
    

    def freeze_backbone(self):
        """Freeze all parameters from the main model, but not the heads"""
        for param in self.backbone.parameters():
            param.requires_grad = False
    
 
    def freeze_bn(self):
        """Freeze all Batch Normalization layers from the model and use them in eval() mode"""
        for m in self.backbone.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                for param in m.parameters(): 
                    param.requires_grad=False
               
