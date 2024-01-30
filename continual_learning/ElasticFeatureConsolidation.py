from copy import deepcopy
import torch
from tqdm import tqdm
import os
import pickle
from continual_learning.IncrementalApproach import IncrementalApproach
from continual_learning.metrics.metric_evaluator import MetricEvaluator
from continual_learning.models.BaseModel import BaseModel
from continual_learning.utils.proto import  ProtoGenerator
from continual_learning.utils.buffer import Buffer
from continual_learning.utils.empirical_feature_matrix import EmpiricalFeatureMatrix

from torch import eig, nn 
import numpy as np
from time import time 
from torch.distributions import MultivariateNormal



class ElasticFeatureConsolidation(IncrementalApproach):
    def __init__(self, args, device, out_path, class_per_task, task_dict, exemplar_loaders=None):
        
        super(ElasticFeatureConsolidation, self).__init__(args, device, out_path, class_per_task, task_dict, exemplar_loaders) 
        self.model = BaseModel(backbone=self.backbone,dataset=args.dataset)
        self.old_model = None 
        self.efc_lamb = args.efc_lamb
        self.damping = args.efc_damping
        self.sigma_proto_update = args.efc_protoupdate
        self.protoloss_weight = args.efc_protoweight
        self.proto_loss = args.efc_protoloss
        self.protobatch_size = args.efc_protobatchsize
 
        self.proto_generator = ProtoGenerator(device=args.device, task_dict=task_dict,
                                              batch_size=args.batch_size,
                                              feature_space_size=self.model.get_feat_size(),
                                              )
        
    
        self.matrix_rank = None
        self.R = None 
        self.auxiliary_classifier = None
        self.print_running_approach()

        self.previous_ef = None  
        self.list_of_ranks = []
        self.list_of_traces = []
        self.batch_idx = 0 

    def print_running_approach(self):
        super(ElasticFeatureConsolidation, self).print_running_approach()
        print("\n efc_hyperparams")
        print("- applying self rotation only at task 0" )
        print("- efc_lamb: {}".format(self.efc_lamb))
        print("- damping: {}".format(self.damping)) 
        print("\n Proto_hyperparams")
        print("- proto Loss {}".format(self.proto_loss))
        if self.proto_loss == "symmetric":
            print("- proto_loss_weight: {}".format(self.protoloss_weight))
        print("- sigma update prototypes {}".format(self.sigma_proto_update))
 
  
    def pre_train(self,  task_id, trn_loader, test_loader):
        if  task_id == 0:
            self.auxiliary_classifier = nn.Linear(512, len(self.task_dict[task_id]) * 3 )
            self.auxiliary_classifier.to(self.device)
        else:
            self.auxiliary_classifier = None 
        
        self.old_model = deepcopy(self.model)
           
        self.old_model.freeze_all()
        self.old_model.to(self.device)
        self.old_model.eval()
 
        self.model.add_classification_head(len(self.task_dict[task_id]))
        self.model.to(self.device)
        
 
        # Restore best and save model for future tasks
        if task_id > 0 and self.proto_loss == "asymmetric":
            self.buffer = Buffer(task_id, self.task_dict)
            
        super(ElasticFeatureConsolidation, self).pre_train(task_id)
        
    
    def train(self, task_id, trn_loader, epoch, epochs):
        if task_id == 0:
            self.train_symmetric(task_id, trn_loader, epoch, epochs)
        else:
            if self.proto_loss == "asymmetric":
                self.train_asymmetric(task_id, trn_loader, epoch, epochs)
            elif self.proto_loss == "symmetric":
                self.train_symmetric(task_id, trn_loader, epoch, epochs)
        
    
    def post_train(self, task_id, trn_loader):
        """Runs after training all the epochs of the task (after the train session)"""
        with torch.no_grad():
            if task_id > 0 and self.sigma_proto_update != -1:
                print("Final Computing Update Proto")
                new_features, old_features, labels_list, old_outputs  = self.get_old_new_features(trn_loader)
                drift = self.compute_drift(new_features, old_features, old_outputs,  device="cpu", task_id=task_id)
                drift = drift.to(self.device)
                
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
        self.save_efm(self.previous_efm, task_id)
        
        if len(self.list_of_ranks) > 0:
            torch.save(self.list_of_ranks, os.path.join(self.out_path, "rank_list_task_{}.pt".format(task_id)))
            self.list_of_ranks = []
        if len(self.list_of_traces) > 0:
            torch.save(self.list_of_traces, os.path.join(self.out_path, "traces_list_task_{}.pt".format(task_id)))
            self.list_of_traces = []
            
        print("Computing New Task Prototypes")
        self.proto_generator.compute(self.model, deepcopy(trn_loader), task_id)
    

            
    def compute_drift(self, new_features, old_features, old_outputs, device, task_id=None): 
        DY = (new_features - old_features).to(device)
        new_features =  new_features.to(device)
        old_features =  old_features.to(device)
        running_prototypes = torch.stack(self.proto_generator.running_proto, dim=0) 

        running_prototypes = running_prototypes.to(device)
        diff = (torch.tile(old_features[None, :, :], [running_prototypes.shape[0], 1, 1])-torch.tile(running_prototypes[:, None, :], [1, old_features.shape[0], 1])) 
        distance = torch.zeros(diff.shape[0], diff.shape[1])
            
        for i in range(diff.shape[0]):
            current_diff = diff[i, :, :].to(self.device)
            distance[i, :] = -torch.bmm(torch.bmm(current_diff.unsqueeze(1), self.previous_efm.expand(diff.shape[1], -1, -1)), current_diff.unsqueeze(2)).flatten().cpu()
        
        
        scaled_distance = (distance- distance.min())/(distance.max() - distance.min())
    
        W  = torch.exp(scaled_distance/(2*self.sigma_proto_update ** 2)) 
        normalization_factor  = torch.sum(W, axis=1)[:, None]
        W_norm = W/torch.tile(normalization_factor, [1, W.shape[1]])
        displacement = torch.sum(torch.tile(W_norm[:, :, None], [
            1, 1, DY.shape[1]])*torch.tile(DY[None, :, :], [scaled_distance.shape[0], 1, 1]), axis=1)
            
        return displacement
            
    
    def efc_loss(self, features, features_old):
        features = features.unsqueeze(1)
        features_old = features_old.unsqueeze(1)
        matrix_reg = self.efc_lamb *  self.previous_efm + self.damping * torch.eye(self.previous_efm.shape[0], device=self.device) 
        efc_loss = torch.mean(torch.bmm(torch.bmm((features - features_old), matrix_reg.expand(features.shape[0], -1, -1)), (features - features_old).permute(0,2,1)))
        return  efc_loss
    
       
    def asymmetric_criterion(self, outputs, targets, t, features, old_features, proto_to_samples, 
                           current_batch_size, test_id=None):
        
        cls_loss,  loss_protoAug, reg_loss = 0, 0, 0
        n_proto = 0
        
        reg_loss = self.efc_loss(features[:current_batch_size], old_features)  
            
        with torch.no_grad():
            proto_aug, proto_aug_label, n_proto = self.proto_generator.perturbe(t, self.protobatch_size)                
            proto_aug = proto_aug[:proto_to_samples]
            proto_aug_label = proto_aug_label[:proto_to_samples]
            n_proto = proto_to_samples
            
        soft_feat_aug = []
        
        
        for _, head in enumerate(self.model.heads):
            soft_feat_aug.append(head(proto_aug))
        
        soft_feat_aug =  torch.cat(soft_feat_aug, dim=1)
        overall_logits = torch.cat([soft_feat_aug , torch.cat(list(outputs.values()), dim=1)[current_batch_size:, :] ], dim=0)
        overall_targets = torch.cat([proto_aug_label, targets[current_batch_size:]])
        
        loss_protoAug =  torch.nn.functional.cross_entropy(overall_logits, overall_targets)  
        cls_loss += torch.nn.functional.cross_entropy(outputs[test_id][:current_batch_size, :], self.rescale_targets(targets[:current_batch_size], test_id)) 
        
        return cls_loss, reg_loss, loss_protoAug, n_proto
    
    
    def symmetric_criterion(self, outputs, targets, t, features, old_features, test_id=None):
        
        cls_loss,  loss_protoAug, reg_loss = 0, 0, 0
        n_proto = 0

        if t > 0 : 
            reg_loss = self.efc_loss(features, old_features)  
            if self.protoloss_weight > 0:      
                with torch.no_grad():
                    proto_aug, proto_aug_label, n_proto = self.proto_generator.perturbe(t, self.protobatch_size)
                soft_feat_aug = []              
                for _, head in enumerate(self.model.heads):
                    soft_feat_aug.append(head(proto_aug))
                
    
                soft_feat_aug =  torch.cat(soft_feat_aug, dim=1)
    
                    
                loss_protoAug = self.protoloss_weight * nn.CrossEntropyLoss()(soft_feat_aug , proto_aug_label)
            
        cls_loss += torch.nn.functional.cross_entropy(torch.cat(list(outputs.values()), dim=1), targets)   
 
        return cls_loss, reg_loss, loss_protoAug, n_proto
    

    def train_asymmetric(self, task_id, trn_loader, epoch, epochs):

        self.model.eval()
                
        start = time()
        count_batches = 0     
    
        for images, targets in trn_loader:
            count_batches += 1
            
            images = images.to(self.device)
            targets = targets.to(self.device)
   
            current_batch_size = images.shape[0]
            
            old_features = None
            
            # Forward old model
            old_outputs, old_features = self.old_model(images)
                
            if  self.buffer.previous_batch_samples is not None:
                pb, previous_batch_samples, previous_batch_labels = self.buffer.sample(self.batch_size)
                images = torch.cat([images, previous_batch_samples], dim=0)
                targets = torch.cat([targets, previous_batch_labels], dim=0)
            else:
                pb =  self.protobatch_size 


            # Forward current model
            outputs, features = self.model(images)
                    
            cls_loss, efc_loss, loss_protoAug, _  = self.asymmetric_criterion(outputs, targets, task_id, features, old_features, 
                                                                              current_batch_size = current_batch_size, 
                                                                              proto_to_samples=pb, test_id=task_id)
            
            loss = cls_loss + efc_loss + loss_protoAug  
            self.log_batch(task_id,  self.batch_idx, loss, cls_loss, efc_loss, loss_protoAug)
            self.batch_idx += 1 


            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            # Add current batch in the buffer to be replayed in the next steps
            self.buffer.add_data(current_samples=images[:current_batch_size], current_targets=targets[:current_batch_size])
                          
        end = time()
     
        print("Task {}, Epoch {}/{}, N_batch {}, Elapsed Time {}s".format(task_id, epoch, epochs, count_batches, end-start))

                                
    def train_symmetric(self, task_id, trn_loader, epoch, epochs):
        
        if task_id == 0:
            self.auxiliary_classifier.train()
        
        self.model.train()

        if  task_id > 0:
            self.model.eval() 
                       
        start = time()
        count_batches = 0     
        
        for images, targets in trn_loader:
            count_batches += 1
            
            images = images.to(self.device)
            targets = targets.to(self.device)
            seen_classes = sum([len(self.task_dict[t]) for t in range(task_id)])
            if task_id == 0:
                images_rot = torch.stack([torch.rot90(images, k, (2, 3)) for k in range(1, 4)], 1)
                images_rot = images_rot.view(-1, 3, self.image_size, self.image_size)
                target_rot = torch.stack([(targets * 3 + k)+ len(self.task_dict[task_id]) -1 for k in range(1, 4)], 1).view(-1)
                targets = torch.cat([targets, target_rot], dim=0)
                images = torch.cat([images, images_rot], dim=0)
            else:
                target_rot = None
            
         
            old_features = None
            
            # Forward old model
            if task_id > 0:
                _, old_features = self.old_model(images)

            # Forward current model
            outputs, features = self.model(images)
            
            if task_id == 0:
                # output of rotation
                out_rot = self.auxiliary_classifier(features)
                outputs[task_id] = torch.cat([outputs[task_id], out_rot],axis=1)
                
                    
            cls_loss, efc_loss, loss_protoAug, _  = self.symmetric_criterion(outputs, targets, task_id, features, old_features, test_id=task_id)
            
            loss = cls_loss + efc_loss + loss_protoAug  
            self.log_batch(task_id,  self.batch_idx, loss, cls_loss, efc_loss, loss_protoAug)
            self.batch_idx += 1 
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
                        
        end = time()
        print("Task {}, Epoch {}/{}, N_batch {}, Elapsed Time {}s".format(task_id, epoch, epochs, count_batches, end-start))
 
    def eval(self, current_training_task, test_id, loader, epoch,  verbose):
        metric_evaluator = MetricEvaluator(self.out_path, self.task_dict)
        
        cls_loss, efc_loss, proto_loss,  = 0, 0, 0 
        n_samples = 0
        total_prototypes = 0

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
                
     
                cls_loss_batch, efc_loss_batch, proto_loss_batch, n_proto  = self.symmetric_criterion(outputs, targets, current_training_task, features, old_features, test_id=test_id)
                
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
            
            if current_training_task > 0:
                overall_loss = cls_loss/n_samples + efc_loss/n_samples + proto_loss/total_prototypes
            else:
                overall_loss = cls_loss/n_samples
                
            if current_training_task > 0:
    
                self.log(current_training_task, test_id, epoch, cls_loss/n_samples, efc_loss/n_samples, 
                        proto_loss/total_prototypes,
                        tag_acc, taw_acc) 
            else:
                
                self.log(current_training_task, test_id, epoch, cls_loss/n_samples, 0.0, 
                        0.0,
                        tag_acc, taw_acc) 
                    
            if verbose:
                print(" - classification loss: {}".format(cls_loss/n_samples))
                if current_training_task > 0:
                    print(" - efc loss: {}".format(efc_loss/n_samples))
                    print(" - proto loss: {}, N proto {}, bs proto {}".format(proto_loss/total_prototypes, 
                                                                             total_prototypes, n_proto))
        
                
            return taw_acc,  tag_acc, overall_loss      
        
    
    def get_old_new_features(self, trn_loader):
        self.model.eval()
        self.old_model.eval()

        features_list = []
        old_features_list = []
        labels_list = []
        old_outputs = []
        with torch.no_grad():
            for images, labels in  trn_loader:
                images = images.to(self.device)
                labels = labels.type(dtype=torch.int64).to(self.device)
                _, features = self.model(images)
                old_out, old_features = self.old_model(images)
                old_outputs.append(torch.cat(list(old_out.values()), dim=1))
                features_list.append(features) 
                old_features_list.append(old_features)
                labels_list.append(labels)
            
            old_outputs = torch.cat(old_outputs, dim=0)
            old_features  = torch.cat(old_features_list)
            new_features  = torch.cat(features_list)
            labels_list = torch.cat(labels_list)
            
        return new_features, old_features, labels_list, old_outputs
                    
                    
    
    def log_batch(self, current_training_task,   batch_idx,  loss, clf_loss, efc_loss, proto_loss):
        name_tb = "training_task_" + str(current_training_task) + "/batch_overall_loss" 
        self.logger.add_scalar(name_tb, loss.item(), batch_idx)
        name_tb = "training_task_" + str(current_training_task) + "/batch_classification_loss" 
        self.logger.add_scalar(name_tb, clf_loss.item(), batch_idx)
        if current_training_task > 0:
            name_tb = "training_task_" + str(current_training_task) + "/batch_proto_loss" 
            self.logger.add_scalar(name_tb, proto_loss.item(), batch_idx)
            name_tb = "training_task_" + str(current_training_task) + "/batch_efc_loss" 
            self.logger.add_scalar(name_tb, efc_loss.item(), batch_idx)
            
            
        

    def log(self, current_training_task, test_id, epoch, clf_loss, efc_loss, proto_loss,   tag_acc , taw_acc):
        name_tb = "training_task_" + str(current_training_task) + "/dataset_" + str(test_id) + "_classification_loss"
        self.logger.add_scalar(name_tb, clf_loss, epoch)

        name_tb = "training_task_" + str(current_training_task) + "/dataset_" + str(test_id) + "_TAG_accuracy"
        self.logger.add_scalar(name_tb, tag_acc, epoch)

        name_tb = "training_task_" + str(current_training_task) + "/dataset_" + str(test_id) + "_TAW_accuracy"
        self.logger.add_scalar(name_tb, taw_acc, epoch)
        
        if current_training_task > 0:
            name_tb = "training_task_" + str(current_training_task) + "/dataset_" + str(test_id) + "_efc_loss"
            self.logger.add_scalar(name_tb, efc_loss, epoch)
            
            name_tb = "training_task_" + str(current_training_task) + "/dataset_" + str(test_id) + "_proto_loss"
            self.logger.add_scalar(name_tb, proto_loss, epoch)
            
 

    def save_efm(self,cov, task_id):
        print("Saving Empirical Feature Matrix")
        torch.save(torch.trace(cov).item(), os.path.join(self.out_path, "trace_task_{}.pt".format(task_id)))
        torch.save(torch.linalg.matrix_rank(cov.cpu()).item(), os.path.join(self.out_path, "rank_task_{}.pt".format(task_id)))
        torch.save(cov, os.path.join(self.out_path, "efm_task_{}.pt".format(task_id)))

 