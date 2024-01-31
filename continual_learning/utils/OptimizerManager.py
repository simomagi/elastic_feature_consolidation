import torch 


class OptimizerManager:
    
    def __init__(self, approach, dataset, rotation) -> None:
        
        self.approach = approach
        self.dataset = dataset
        self.rotation = rotation
    
    def get_optimizer(self, task_id, model, auxiliary_classifier):
        ## Large First Task Training
        if task_id == 0:
            params_to_optimize = [p for p in  model.backbone.parameters() if p.requires_grad] + [p for p in model.heads.parameters() if p.requires_grad]
           
            if self.rotation:
                params_to_optimize += [p for p in auxiliary_classifier.parameters() if p.requires_grad]
            else:
                print("Optimizing Self Rotation")
                
            if self.dataset == "imagenet-subset" or self.dataset == "imagenet-1k":
                
                lr_first_task = 0.1 
                gamma = 0.1 
                custom_weight_decay = 5e-4 
                custom_momentum = 0.9  
                
                milestones_first_task = [80, 120, 150]
                optimizer = torch.optim.SGD(params_to_optimize, lr=lr_first_task, momentum=custom_momentum,
                                                weight_decay=custom_weight_decay)
                
                reduce_lr_on_plateau = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                            milestones=milestones_first_task,
                                                            gamma=gamma, verbose=True
                                                            )
                print("Using SGD Optimizer With PASS setting: \n\
                        LR: {}, Step Decay {} with Milestones {}, Weight Decay {}, Momentum {} ".format(lr_first_task,
                                                                                                            gamma,
                                                                                                            milestones_first_task,
                                                                                                            custom_weight_decay,
                                                                                                            custom_momentum,
                                                                                                            ))
            else:

                lr_first_task = 1e-3
                milestones_first_task = [45, 90]
                custom_weight_decay = 2e-4
                gamma = 0.1
                optimizer = torch.optim.Adam(params_to_optimize, lr=lr_first_task, weight_decay=custom_weight_decay)
                reduce_lr_on_plateau = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                                                milestones=milestones_first_task,
                                                                            gamma=gamma, verbose=True)
                print("Using Adam Optimizer With PASS setting: \n\
                        LR: {}, Step Decay {} with Milestones {}, Weight Decay {} ".format(lr_first_task,
                                                                                            gamma,
                                                                                            milestones_first_task,
                                                                                            custom_weight_decay
                                                                                                            )) 
            
            return optimizer, reduce_lr_on_plateau 
        
        else:
            # Oprimization Settings Next Tasks
            model.freeze_bn()
            if self.dataset == "imagenet-subset"  or self.dataset == "imagenet-1k": 
                backbone_lr = 1e-5
                head_lr = 1e-4
                custom_weight_decay = 2e-4
                backbone_params = [p for p in  model.backbone.parameters() if p.requires_grad]
            
                old_head_params = [p for p in model.heads[:-1].parameters()  if p.requires_grad]
                
                new_head_params = [p for p in  model.heads[-1].parameters() if p.requires_grad]
                head_params = old_head_params + new_head_params
                
                optimizer = torch.optim.Adam([{'params': head_params, 'lr':head_lr},
                                {'params': backbone_params}
                                    ],lr=backbone_lr, 
                                    weight_decay=custom_weight_decay)
                print("Using Adam Optimizer With Fixed LR: \n\
                        Backbone LR: {}, Head LR {}, Weight Decay {},".format(backbone_lr,
                                                                                head_lr,      
                                                                                custom_weight_decay)) 

            else:
                
                old_head_params = [p for p in model.heads[:-1].parameters()  if p.requires_grad]

                new_head_params = [p for p in  model.heads[-1].parameters() if p.requires_grad]
                head_params = old_head_params + new_head_params

                params_to_optimize = [p for p in model.backbone.parameters() if p.requires_grad] +  head_params 
                backbone_lr = head_lr = 1e-4
                custom_weight_decay = 2e-4
                optimizer =  torch.optim.Adam(params_to_optimize, lr=backbone_lr, weight_decay=2e-4)
                print("Using Adam Optimizer With Fixed LR: \n\
                        Backbone LR: {}, Head LR {}, Weight Decay {}".format(backbone_lr,
                                                                            head_lr,      
                                                                            custom_weight_decay)) 

            # Fixed Scheduler
            reduce_lr_on_plateau = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                            milestones=[1000,2000],
                                                            gamma=0.1, verbose=True
                                                            )
            return optimizer, reduce_lr_on_plateau 
            
            
            

  


    
 