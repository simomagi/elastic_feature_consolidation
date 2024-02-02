from copy import deepcopy
import torch
import os
from continual_learning.IncrementalApproach import IncrementalApproach
from continual_learning.metrics.metric_evaluator import MetricEvaluator
from continual_learning.models.BaseModel import BaseModel
from continual_learning.utils.proto import  ProtoGenerator
from continual_learning.utils.buffer import Buffer
from continual_learning.utils.empirical_feature_matrix import EmpiricalFeatureMatrix
from continual_learning.utils.training_utils import compute_rotations, get_old_new_features,  save_efm
from torch import  nn 
from time import time 
 


class ElasticFeatureConsolidation(IncrementalApproach):
    def __init__(self, args, device, out_path, class_per_task, task_dict):
        
        super(ElasticFeatureConsolidation, self).__init__(args, device, out_path, class_per_task, task_dict) 
        self.model = BaseModel(backbone=self.backbone,dataset=args.dataset)
        self.old_model = None 
        self.efc_lamb = args.efc_lamb
        self.damping = args.efc_damping
        self.sigma_proto_update = args.efc_protoupdate
        self.protobatch_size = args.efc_protobatchsize
    
        self.proto_generator = ProtoGenerator(device=args.device, task_dict=task_dict,
                                              batch_size=args.batch_size, out_path=out_path,
                                              feature_space_size=self.model.get_feat_size(),
                                              )

        self.matrix_rank = None
        self.R = None 
        self.auxiliary_classifier = None
        self.previous_efm = None  
        self.print_running_approach()
        self.batch_idx = 0 

    def print_running_approach(self):
        super(ElasticFeatureConsolidation, self).print_running_approach()
        print("\n efc_hyperparams")
        print("- efc_lamb: {}".format(self.efc_lamb))
        print("- damping: {}".format(self.damping)) 
        print("\n Proto_hyperparams")
        print("- sigma update prototypes {}".format(self.sigma_proto_update))
 
  
    def pre_train(self,  task_id):
        if  task_id == 0 and self.rotation:
            # Auxiliary classifier used for self rotation
            self.auxiliary_classifier = nn.Linear(512, len(self.task_dict[task_id]) * 3 )
            self.auxiliary_classifier.to(self.device)
        else:
            # Self rotation is not used after the first task
            self.auxiliary_classifier = None 
            
        # Freeze old model
        self.old_model = deepcopy(self.model)
        self.old_model.freeze_all()
        self.old_model.to(self.device)
        self.old_model.eval()
        
        # Add classification head
        self.model.add_classification_head(len(self.task_dict[task_id]))
        self.model.to(self.device)
        
        if task_id > 0:
            # initialize the buffer for PR-ACE for each new tasks
            print("Using PR-ACE")
            self.buffer = Buffer(task_id, self.task_dict)
        else:
            print("Standard training with cross entropy")
            
        super(ElasticFeatureConsolidation, self).pre_train(task_id)
    
    def efm_loss(self, features, features_old):
        features = features.unsqueeze(1)
        features_old = features_old.unsqueeze(1)
        matrix_reg = self.efc_lamb *  self.previous_efm + self.damping * torch.eye(self.previous_efm.shape[0], device=self.device) 
        efc_loss = torch.mean(torch.bmm(torch.bmm((features - features_old), matrix_reg.expand(features.shape[0], -1, -1)), (features - features_old).permute(0,2,1)))
        return  efc_loss
    
    def train_criterion(self, outputs, targets, t, features, old_features, proto_to_samples, current_batch_size):
        
        cls_loss,  loss_protoAug, reg_loss, n_proto = 0, 0, 0, 0 

        if t > 0:
            # EFM loss 
            reg_loss = self.efm_loss(features[:current_batch_size], old_features)  
                
            with torch.no_grad():
                # prototype sampling 
                proto_aug, proto_aug_label, n_proto = self.proto_generator.perturbe(t, self.protobatch_size)                
                proto_aug = proto_aug[:proto_to_samples].to(self.device)
                proto_aug_label = proto_aug_label[:proto_to_samples].to(self.device)
                n_proto = proto_to_samples
                    
            soft_feat_aug = []
            for _, head in enumerate(self.model.heads):
                soft_feat_aug.append(head(proto_aug))
            
            soft_feat_aug =  torch.cat(soft_feat_aug, dim=1)
            overall_logits = torch.cat([soft_feat_aug , torch.cat(list(outputs.values()), dim=1)[current_batch_size:, :] ], dim=0)
            overall_targets = torch.cat([proto_aug_label, targets[current_batch_size:]])
            
            # loss over all encountered classes (proto+current class samples)
            loss_protoAug =  torch.nn.functional.cross_entropy(overall_logits, overall_targets)  
        
            # loss over current classes (only current class samples)
            cls_loss = torch.nn.functional.cross_entropy(outputs[t][:current_batch_size, :], self.rescale_targets(targets[:current_batch_size], t)) 
        
        else:
            # first task only a cross entropy loss
            cls_loss = torch.nn.functional.cross_entropy(torch.cat(list(outputs.values()), dim=1), targets)  
        
        return cls_loss, reg_loss, loss_protoAug, n_proto
    
        
    def train(self, task_id, trn_loader, epoch, epochs):

        if task_id == 0:
            self.model.train()
        else:
            # following SDC https://github.com/yulu0724/SDC-IL/blob/master/train.py, we freeze batch norm after the first task.
            self.model.eval()
            
        if  task_id == 0 and self.rotation:
            self.auxiliary_classifier.train() 
                       
        start = time()
        count_batches = 0     
        for images, targets in trn_loader:
            count_batches += 1
            
            images = images.to(self.device)
            targets = targets.to(self.device)
   
            current_batch_size = images.shape[0]
            
            if  task_id == 0 and self.rotation:
                images_rot, target_rot = compute_rotations(images, self.image_size, self.task_dict, targets, task_id)
                images = torch.cat([images, images_rot], dim=0)
                targets = torch.cat([targets, target_rot], dim=0)
         
                
            if task_id > 0:
                # Forward old model
                _, old_features = self.old_model(images)
                
                if  self.buffer.previous_batch_samples is not None:
                    # Sample from a buffer of current task data for PR-ACE
                    pb, previous_batch_samples, previous_batch_labels = self.buffer.sample(self.batch_size)
                    images = torch.cat([images, previous_batch_samples], dim=0)
                    targets = torch.cat([targets, previous_batch_labels], dim=0)
                else:
                    pb =  self.protobatch_size 
            else:
                old_features = None
                pb = 0
                
            # Forward in the current model
            outputs, features = self.model(images)
            
            if  task_id == 0 and self.rotation:
                # predict the rotation the rotations
                out_rot = self.auxiliary_classifier(features)
                outputs[task_id] = torch.cat([outputs[task_id], out_rot],axis=1)
                
                
            # compute criterion
            cls_loss, efc_loss, loss_protoAug, _  = self.train_criterion(outputs, targets, task_id, features, old_features, 
                                                                    current_batch_size = current_batch_size, 
                                                                    proto_to_samples=pb)
            
            loss = cls_loss + efc_loss + loss_protoAug  
            
            self.batch_idx += 1 
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            # Add current batch in the buffer to be replayed in the next steps for PR-ACE
            if task_id > 0:
                self.buffer.add_data(current_samples=images[:current_batch_size],
                                     current_targets=targets[:current_batch_size])
                             
        end = time()
        print("Task {}, Epoch {}/{}, N_batch {}, Elapsed Time {}s".format(task_id, epoch, epochs, count_batches, end-start))

    
    def eval_criterion(self, outputs, targets, t, features, old_features):
        
        cls_loss,  loss_protoAug, reg_loss = 0, 0, 0
        n_proto = 0

        if t > 0 : 
            reg_loss = self.efm_loss(features, old_features)  
            with torch.no_grad():
                proto_aug, proto_aug_label, n_proto = self.proto_generator.perturbe(t, self.protobatch_size)
                proto_aug = proto_aug.to(self.device)
                proto_aug_label = proto_aug_label.to(self.device)
               
                soft_feat_aug = []              
                for _, head in enumerate(self.model.heads):
                    soft_feat_aug.append(head(proto_aug))
                
                soft_feat_aug =  torch.cat(soft_feat_aug, dim=1)
                loss_protoAug = nn.CrossEntropyLoss()(soft_feat_aug , proto_aug_label)
            
        cls_loss += torch.nn.functional.cross_entropy(torch.cat(list(outputs.values()), dim=1), targets)   
        return cls_loss, reg_loss, loss_protoAug, n_proto
    
        
    def eval(self, current_training_task, test_id, loader, epoch,  verbose):
        metric_evaluator = MetricEvaluator(self.out_path, self.task_dict)
        
        cls_loss, efc_loss, proto_loss, n_samples, total_prototypes  = 0, 0, 0, 0, 0
        
        with torch.no_grad():
            self.old_model.eval()
            self.model.eval()
            
            for images, targets in loader:
                images = images.to(self.device)
                targets = targets.type(dtype=torch.int64).to(self.device)
                
                current_batch_size = images.shape[0]
                original_labels = deepcopy(targets)
        
                outputs, features = self.model(images)
                _, old_features = self.old_model(images)
                cls_loss_batch, efc_loss_batch, proto_loss_batch, n_proto  = self.eval_criterion(outputs, targets,  current_training_task, 
                                                                                                 features, old_features)
                
                cls_loss += cls_loss_batch * current_batch_size        
                efc_loss += efc_loss_batch * current_batch_size
                proto_loss += proto_loss_batch * n_proto
                total_prototypes += n_proto
                n_samples += current_batch_size
                
                metric_evaluator.update(original_labels, self.rescale_targets(targets, test_id), 
                                        self.tag_probabilities(outputs), 
                                        self.taw_probabilities(outputs, test_id),
                                        )

            taw_acc,  tag_acc = metric_evaluator.get(verbose=verbose)
            cls_loss = cls_loss/n_samples 
            
            if current_training_task > 0:
                efc_loss = efc_loss/n_samples
                proto_loss = proto_loss/total_prototypes
                overall_loss = cls_loss  + efc_loss  + proto_loss 
            else:
                overall_loss = cls_loss 
            
            if verbose:
                print(" - classification loss: {}".format(cls_loss))
                if current_training_task > 0:
                    print(" - efc loss: {}".format(efc_loss))
                    print(" - proto loss: {}, N proto {}, bs proto {}".format(proto_loss, 
                                                                             total_prototypes, n_proto))
        
            return taw_acc,  tag_acc, overall_loss      
          
    
    def post_train(self, task_id, trn_loader):
        """Runs after training all the epochs of the task (after the train session)"""
        with torch.no_grad():
            if task_id > 0 and self.sigma_proto_update != -1:
                print("Final Computing Update Proto")
                start = time()
                new_features, old_features  =  get_old_new_features(self.model, self.old_model, 
                                                                                 trn_loader, self.device)
                
                drift = self.compute_drift(new_features, old_features,  device="cpu")
                drift = drift.cpu()
                end = time()
                print("Elapsed time {}".format(end-start))
                for i, (p, var, proto_label) in enumerate(zip(self.proto_generator.prototype, 
                                                            self.proto_generator.running_proto_variance,
                                                            self.proto_generator.class_label)):
            
                    mean = p + drift[i] 
                    self.proto_generator.update_gaussian(proto_label, mean, var)
                    # final update the mean
                    self.proto_generator.prototype[i] = mean 
                    
                self.proto_generator.running_proto = deepcopy(self.proto_generator.prototype)

        
            efm_matrix = EmpiricalFeatureMatrix(self.device, out_path=self.out_path)
            efm_matrix.compute(self.model,  deepcopy(trn_loader), task_id)
            self.previous_efm = efm_matrix.get()           
            R, L, V = torch.linalg.svd(self.previous_efm)
            matrix_rank = torch.linalg.matrix_rank(self.previous_efm)
            print("Computed Matrix Rank {}".format(matrix_rank))
            self.R = R
            self.L = L
            self.matrix_rank = matrix_rank

        # save matrix after each task for analysis
        save_efm(self.previous_efm, task_id, self.out_path)
            
        print("Computing New Task Prototypes")
        self.proto_generator.compute(self.model, deepcopy(trn_loader), task_id)
    

            
    def compute_drift(self,  new_features, old_features, device): 
        DY = (new_features - old_features).to(device)
        new_features =  new_features.to(device)
        old_features =  old_features.to(device)
        running_prototypes = torch.stack(self.proto_generator.running_proto, dim=0) 

        running_prototypes = running_prototypes.to(device)
        distance = torch.zeros(len(running_prototypes), new_features.shape[0])

        for i in range(running_prototypes.shape[0]): 
            # we use the EFM to update prototypes
            curr_diff = (old_features - running_prototypes[i, :].unsqueeze(0)).unsqueeze(1).to(self.device)

            distance[i] = -torch.bmm(torch.bmm(curr_diff, self.previous_efm.expand(curr_diff.shape[0], -1, -1)), curr_diff.permute(0,2,1)).flatten().cpu()

        
        scaled_distance = (distance- distance.min())/(distance.max() - distance.min())

        W  = torch.exp(scaled_distance/(2*self.sigma_proto_update ** 2)) 
        normalization_factor  = torch.sum(W, axis=1)[:, None]
        W_norm = W/torch.tile(normalization_factor, [1, W.shape[1]])

        displacement = torch.zeros((running_prototypes.shape[0], 512))

        for i in range(running_prototypes.shape[0]):  
            displacement[i] = torch.sum((W_norm[i].unsqueeze(1) * DY),dim=0)
    
        return displacement