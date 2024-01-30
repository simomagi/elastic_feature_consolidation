import abc
import os
from sched import scheduler
from tabnanny import verbose
import torch
import sys
from torch.utils.tensorboard import SummaryWriter
from continual_learning.utils.OptimizerManager import OptimizerManager

class IncrementalApproach(metaclass=abc.ABCMeta):
    
   def __init__(self, args, device, out_path, class_per_task, task_dict, exemplar_loaders):
      self.device = device
      self.out_path = out_path
      self.class_per_task = class_per_task
      self.task_dict = task_dict 
      self.n_class_first_task = args.n_class_first_task
      # Model args
      self.approach = args.approach
      self.backbone = args.backbone
      # Optimizer args
      self.lr_first_task = args.lr_first_task
      self.scheduler_type  = args.scheduler_type
      self.backbone_lr = args.backbone_lr
      self.head_lr = args.head_lr

      self.batch_size = args.batch_size 
      self.total_epochs = args.epochs 
      self.logger = SummaryWriter(os.path.join(out_path, "tensorboard"))
      self.milestones_first_task = None 
      self.dataset = args.dataset
      if self.dataset=="cifar100":
         self.image_size = 32
      elif self.dataset=="tiny-imagenet":
         self.image_size = 64
      elif self.dataset=="imagenet-subset":
         self.image_size = 224
         print("Fixing the backbone lr to 1e-5 and the head lr to 1e-4")
         self.backbone_lr = 1e-5
         self.head_lr == 1e-4
         
      # SELF-ROTATION classifier
      self.auxiliary_classifier = None
      self.optimizer_manager = OptimizerManager(self.backbone_lr, self.head_lr, 
                                                self.scheduler_type, self.approach)
      
      if self.dataset == "imagenet-subset":
         self.milestones_first_task = [80, 120, 150]
      else:
         self.milestones_first_task = [45, 90]

      
      

   def pre_train(self, task_id, *args):
 
      if task_id == 0:
         params_to_optimize = [p for p in self.model.backbone.parameters() if p.requires_grad] + [p for p in self.model.heads.parameters() if p.requires_grad]
         params_to_optimize += [p for p in self.auxiliary_classifier.parameters() if p.requires_grad]
            
         if self.dataset=="imagenet-subset": 
            self.lr_first_task = 0.1 
       
            gamma = 0.1 
            custom_weight_decay = 5e-4 
            custom_momentum = 0.9  
            print("Using SGD Optimizer With PASS setting")
            self.optimizer = torch.optim.SGD(params_to_optimize, lr=self.lr_first_task, momentum=custom_momentum,
                                             weight_decay=custom_weight_decay)
            
            self.reduce_lr_on_plateau = torch.optim.lr_scheduler.MultiStepLR(self.optimizer,
                                                         milestones=self.milestones_first_task,
                                                         gamma=gamma, verbose=True
                                                         )
         else:
    
            self.optimizer = torch.optim.Adam(params_to_optimize, lr=self.lr_first_task, weight_decay=2e-4)
            self.reduce_lr_on_plateau = torch.optim.lr_scheduler.MultiStepLR(self.optimizer,
                                                      milestones=self.milestones_first_task,
                                                      gamma=0.1, verbose=True
                                                         )
      else:            

            self.optimizer, self.reduce_lr_on_plateau = self.optimizer_manager.get_optimizer(task_id, self.model, self.auxiliary_classifier)

 
             
   def rescale_targets(self, targets, t):
       offset =  (t-1)*self.class_per_task + self.n_class_first_task  if self.n_class_first_task > -1 and t > 0 else t*self.class_per_task
       targets = targets - offset
       return targets
    

    
   @abc.abstractmethod
   def train(self, *args):
      pass 

   @abc.abstractmethod
   def post_train(self, *args):
      pass 

   @abc.abstractmethod
   def eval(self, *args):
      pass

   @abc.abstractmethod
   def log(self, *args):
      pass
   

   def print_running_approach(self):
      print("#"*40 + " --> RUNNING APPROACH")
      print("- approach: {}".format(self.approach))
      print("- backbone: {}".format(self.backbone))
      print("- batch size : {}".format(self.batch_size))
      print("- lr first task: {} with decay multi-step {}".format(self.lr_first_task, self.milestones_first_task))
      print("- incremental phases: backbone lr : {}".format(self.backbone_lr))
      print("- incremental phases: head lr : {}".format(self.head_lr))
      print("- incremental phases: scheduler type  {}".format(self.scheduler_type))
      print()

   
 
   def tag_probabilities(self, outputs):
        tag_output = []
        for key in outputs.keys():
            tag_output.append(outputs[key])
        tag_output = torch.cat(tag_output, dim=1)
        probabilities = torch.nn.Softmax(dim=1)(tag_output)
        return probabilities
   

   def taw_probabilities(self, outputs, head_id):
      probabilities = torch.nn.Softmax(dim=1)(outputs[head_id])
      return probabilities