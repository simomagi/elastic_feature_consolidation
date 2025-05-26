from copy import deepcopy
import torch
from continual_learning.IncrementalApproach import IncrementalApproach
from continual_learning.metrics.metric_evaluator import MetricEvaluator
from continual_learning.models.BaseModel import BaseModel
from continual_learning.utils.proto import  ProtoGenerator
from continual_learning.utils.buffer import Buffer
from continual_learning.utils.empirical_feature_matrix import EmpiricalFeatureMatrix
from continual_learning.utils.training_utils import compute_rotations, get_old_new_features,  save_efm
from torch import  nn 
from time import time 
from tqdm import tqdm 
from torch.utils.data import TensorDataset, DataLoader, ConcatDataset
from sklearn.metrics import accuracy_score
import numpy as np



class ElasticFeatPlusPlus(IncrementalApproach):
    def __init__(self, args, device, out_path, class_per_task, task_dict):
        
        super(ElasticFeatPlusPlus, self).__init__(args, device, out_path, class_per_task, task_dict) 
        self.model = BaseModel(backbone=self.backbone, dataset=args.dataset)
        self.old_model = None 
        self.efc_lamb = args.efc_lamb
        self.damping = args.efc_damping
        self.sigma_proto_update = args.efc_protoupdate
        
        self.balanced_bs = args.balanced_bs
        self.balanced_epochs = args.balanced_epochs
        self.balanced_lr = args.balanced_lr
 
        
        self.proto_generator = ProtoGenerator(
            device=args.device, task_dict=task_dict,
            batch_size=args.batch_size, out_path=out_path,
            feature_space_size=self.model.get_feat_size()
        )
    
        self.matrix_rank = None
        self.R = None 
        self.auxiliary_classifier = None
        self.previous_efm = None  
        self.print_running_approach()
        self.batch_idx = 0 

    def print_running_approach(self):
        super(ElasticFeatPlusPlus, self).print_running_approach()
        
        print("\n EFC++ Hyperparams")
        print("- efc_lamb: {}".format(self.efc_lamb))
        print("- damping: {}".format(self.damping)) 
        
        print("\n Update Class Means Hyperparam")
        print("- sigma update prototypes {}".format(self.sigma_proto_update))
        
        print("\n Prototype Re-Balancing Optimization")
        print("- [rebalancing] batch size {}".format(self.balanced_bs))
        print("- [rebalancing] epochs {}".format(self.balanced_epochs))
        print("- [rebalancing] lr {}".format(self.balanced_lr))
        

  
    def pre_train(self,  task_id):
        if  task_id == 0 and self.rotation:
            # Auxiliary classifier used for self rotation
            self.auxiliary_classifier = nn.Linear(512, len(self.task_dict[task_id]) * 3 )
            self.auxiliary_classifier.to(self.device)
        else:
            # Self rotation is not used after the first task
            self.auxiliary_classifier = None
            print("\nFreezing Batch Norm")
            self.model.freeze_bn()
            
        # Freeze old model
        self.old_model = deepcopy(self.model)
        self.old_model.freeze_all()
        self.old_model.to(self.device)
        self.old_model.eval()
        
        # Add classification head
        self.model.add_classification_head(len(self.task_dict[task_id]))
        self.model.to(self.device)
            
        super(ElasticFeatPlusPlus, self).pre_train(task_id)
    

    def efm_loss(self, features, features_old):
        features = features.unsqueeze(1)
        features_old = features_old.unsqueeze(1)
        matrix_reg = self.efc_lamb *  self.previous_efm + self.damping * torch.eye(self.previous_efm.shape[0], device=self.device) 
        efc_loss = torch.mean(torch.bmm(torch.bmm((features - features_old), matrix_reg.expand(features.shape[0], -1, -1)), (features - features_old).permute(0,2,1)))
        return  efc_loss
    

    def train_criterion(self, outputs, targets, t, features, old_features, current_batch_size):
        cls_loss,  reg_loss = 0, 0
        if t > 0:
            # EFM loss 
            reg_loss = self.efm_loss(features, old_features) 
 
        cls_loss = torch.nn.functional.cross_entropy(outputs[t] , self.rescale_targets(targets, t))
  
        return cls_loss, reg_loss
    
        
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
        for images, targets in tqdm(trn_loader):
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
                with torch.no_grad():
                    _, old_features = self.old_model(images)
            else:
                old_features = None
      
            # Forward in the current model
            outputs, features = self.model(images)
            
            if  task_id == 0 and self.rotation:
                # predict the rotation the rotations
                out_rot = self.auxiliary_classifier(features)
                outputs[task_id] = torch.cat([outputs[task_id], out_rot],axis=1)
                
            # compute criterion
            cls_loss, efc_loss  = self.train_criterion(
                outputs, 
                targets, 
                task_id, 
                features, 
                old_features, 
                current_batch_size=current_batch_size
            )
            loss = cls_loss + efc_loss  
            
            self.batch_idx += 1 
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        end = time()
        print("Task {}, Epoch {}/{}, N_batch {}, Elapsed Time {}s".format(task_id, epoch, epochs, count_batches, end-start))

    
    def eval_criterion(self, outputs, targets, t, features, old_features, verbose):
        cls_loss, reg_loss = 0, 0

        if t > 0 : 
            reg_loss = self.efm_loss(features, old_features)  
        
        if verbose:
            cls_loss += torch.nn.functional.cross_entropy(outputs[t], self.rescale_targets(targets, t))  
        else:
            cls_loss += torch.nn.functional.cross_entropy(torch.cat(list(outputs.values()), dim=1), targets)  
         
        return cls_loss, reg_loss 
        
    def eval(self, current_training_task, test_id, loader, verbose):
        metric_evaluator = MetricEvaluator(self.out_path, self.task_dict)
        cls_loss, efc_loss,  n_samples   = 0, 0, 0  
        
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

                cls_loss_batch, efc_loss_batch = self.eval_criterion(outputs, targets, current_training_task, 
                                                                     features, old_features, verbose)
                
                cls_loss += cls_loss_batch * current_batch_size        
                efc_loss += efc_loss_batch * current_batch_size
           
                n_samples += current_batch_size
                
                metric_evaluator.update(
                    original_labels,
                    self.rescale_targets(targets, test_id), 
                    self.tag_probabilities(outputs), 
                    self.taw_probabilities(outputs, test_id),
                )


            cls_loss = cls_loss/n_samples 
            
            if current_training_task > 0:
                efc_loss = efc_loss/n_samples
                overall_loss = cls_loss  + efc_loss   
            else:
                overall_loss = cls_loss 
            
            if verbose:
                print(" - classification loss: {}".format(cls_loss))
                if current_training_task > 0:
                    print(" - efm loss: {}".format(efc_loss))
                    print(" - overall loss: {}".format(overall_loss))
            else:
                taw_acc, tag_acc = metric_evaluator.get()
                return taw_acc, tag_acc

 


    def balance_head(self, trn_loader, model):
        """Balance the head of the model using the prototypes"""
        model.freeze_backbone()
        
        criterion = nn.CrossEntropyLoss()
        model.eval() # added
            
        # Initialize optimizer for classifier re-balancing
        optimizer = torch.optim.SGD(
            [p for p in model.heads.parameters() if p.requires_grad],
            lr=self.balanced_lr,
            weight_decay=5e-4,
            momentum=0.9
        )
    
        if self.dataset == "imagenet-1k":
            # Extract a single time the feature if the dataset is huge
            current_features, current_targets = self.get_feature(trn_loader)
            features_dataset = TensorDataset(current_features, current_targets)
            
    
        for _ in range(0, self.balanced_epochs):
            if self.dataset != "imagenet-1k":
                current_features, current_targets = self.get_feature(trn_loader)
                features_dataset = TensorDataset(current_features, current_targets)
                
            current_prototypes, current_labels = self.generate_sample_from_proto()

            prototypes_dataset = TensorDataset(current_prototypes, current_labels)
            complete_dataset = ConcatDataset([features_dataset, prototypes_dataset])
            epoch_loader = DataLoader(complete_dataset, batch_size=self.balanced_bs, shuffle=True)

            for features, targets in epoch_loader:             
                features = features.to(self.device)
                targets = targets.to(self.device)
                    
                outputs = []
                for _, head in enumerate(model.heads):
                    outputs.append(head(features))
                    
                outputs = torch.cat(outputs, dim=1)
                loss = criterion(outputs, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    
        model.unfreeze_backbone()
        model.freeze_bn()
    
            
    
    def post_train(self, task_id, trn_loader):
        """Runs after training all the epochs of the task (after the train session)"""
        with torch.no_grad():
            if task_id > 0 and self.sigma_proto_update != -1:
                print("Final Computing Update Proto")
                start = time()
                new_features, old_features  =  get_old_new_features(
                    self.model,
                    self.old_model, 
                    trn_loader, 
                    self.device
                )
                
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
                
        ## Balance the classifiers using prototypes
        if task_id > 0:
            print("\nRe-balancing classifiers phase")
            self.balance_head(trn_loader, self.model)

        # Compute Emprirical Feature Matrix
        print("\nComputing Empirical Feature Matrix")
        with torch.no_grad():
            efm_matrix = EmpiricalFeatureMatrix(self.device, out_path=self.out_path)
            efm_matrix.compute(self.model,  deepcopy(trn_loader), task_id)
            self.previous_efm = efm_matrix.get()
           
            R, L, V = torch.linalg.svd(self.previous_efm)
            matrix_rank = torch.linalg.matrix_rank(self.previous_efm)
            print("Computed Matrix Rank {}".format(matrix_rank))
            print("Maximum eigenvalue {}".format(torch.max(L)))
            self.R = R
            self.L = L
            self.matrix_rank = matrix_rank

        # save matrix after each task for analysis
        save_efm(self.previous_efm, task_id, self.out_path)
            
        print("\nComputing New Task Prototypes")
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

        displacement = torch.zeros((running_prototypes.shape[0], self.model.get_feat_size()))

        for i in range(running_prototypes.shape[0]):  
            displacement[i] = torch.sum((W_norm[i].unsqueeze(1) * DY),dim=0)
    
        return displacement
    
    
    @torch.no_grad()
    def generate_sample_from_proto(self,):
        samples = []
        labels = []
        for l, gaussian in self.proto_generator.gaussians.items():
       
            sample_per_class = self.proto_generator.class_stats[l] 
                
            current_class_sample = gaussian.sample((sample_per_class,))
            current_labels = torch.tensor(([l] * sample_per_class))

            current_class_sample = current_class_sample.to(self.device)
 
            samples.append(current_class_sample.cpu())
            labels.append(current_labels.cpu())
 
 
        return torch.cat(samples, dim=0), torch.cat(labels, dim=0) 
    

    def get_feature(self, loader):
        features = []
        labels = []
        with torch.no_grad():
            for imgs, targets in loader:
                imgs = imgs.to(self.device)
                targets = targets.to(self.device)
                
                _, current_features  = self.model(imgs)
                features.append(current_features.cpu())
                labels.append(targets.cpu())
        return torch.cat(features, dim=0), torch.cat(labels)
        