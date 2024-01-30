import torch


class OptimizerManager:
    
    def __init__(self, backbone_lr, head_lr, scheduler_type, approach) -> None:
        
        self.backbone_lr = backbone_lr
        self.head_lr = head_lr 
        self.scheduler_type = scheduler_type
        self.approach = approach
    
    def get_optimizer(self, task_id, model, auxiliary_classifier):
        if task_id > 0:
            model.freeze_bn() 

        backbone_params = [p for p in  model.backbone.parameters() if p.requires_grad]
        old_head_params = [p for p in model.heads[:-1].parameters()  if p.requires_grad]
        new_head_params = [p for p in  model.heads[-1].parameters() if p.requires_grad]
        head_params = old_head_params + new_head_params
        
        if self.approach == "pass":
            print("Adding Rotation Classifier to the Optimizer")
            auxiliary_classifier_params = [p for p in auxiliary_classifier.parameters() if p.requires_grad]
           
            params = backbone_params + head_params +  auxiliary_classifier_params

            if self.backbone_lr == self.head_lr:
                print("Using Adam with a single lr {}".format(self.backbone_lr))
                optimizer =  torch.optim.Adam(params, lr=self.backbone_lr, weight_decay=2e-4)
            else:
                print("Using Adam with two lr. Backbone: {}, Head: {}".format(self.backbone_lr, self.head_lr))
                optimizer = torch.optim.Adam([{'params': head_params, 'lr':self.head_lr},
                                              {'params': auxiliary_classifier_params, 'lr':self.head_lr},
                                              {'params': backbone_params}
                                             ],lr=self.backbone_lr, weight_decay=2e-4)
        elif self.approach == "efc":
    
            params = backbone_params + head_params
            if self.backbone_lr == self.head_lr:
                print("Using Adam with a single lr {}".format(self.backbone_lr))
                optimizer =  torch.optim.Adam(params, lr=self.backbone_lr, weight_decay=2e-4)
            
            else:
                print("Using Adam with two lr. Backbone: {}, Head: {}".format(self.backbone_lr, self.head_lr))
     
                optimizer = torch.optim.Adam([{'params': head_params, 'lr':self.head_lr},
                                              {'params': backbone_params}
                                                ],lr=self.backbone_lr, 
                                                weight_decay=2e-4)
 
        if self.scheduler_type == "multi_step":
            print("Scheduling lr after 45 and 90 epochs")
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                            milestones=[45, 90],
                                                            gamma=0.1, verbose=True
                                                               )
        else:
            print("Fixed Learning Rate")
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                            milestones=[1000,2000],
                                                            gamma=0.1, verbose=True
                                                              )
            
        return optimizer, scheduler