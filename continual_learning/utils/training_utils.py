import torch 
import os 

def compute_rotations(images, image_size, task_dict, targets, task_id):
    # compute self-rotation for the first task following PASS https://github.com/Impression2805/CVPR21_PASS
    images_rot = torch.stack([torch.rot90(images, k, (2, 3)) for k in range(1, 4)], 1)
    images_rot = images_rot.view(-1, 3, image_size,  image_size)
    target_rot = torch.stack([(targets * 3 + k)+ len(task_dict[task_id]) -1 for k in range(1, 4)], 1).view(-1)
    return images_rot, target_rot 


def save_efm(cov, task_id, out_path):
    print("Saving Empirical Feature Matrix")
    torch.save(cov, os.path.join(out_path, "efm_task_{}.pt".format(task_id)))

def get_old_new_features(model, old_model, trn_loader, device):
    model.eval()
    old_model.eval()

    features_list = []
    old_features_list = []
    labels_list = []
    old_outputs = []
    with torch.no_grad():
        for images, labels in  trn_loader:
            images = images.to(device)
            labels = labels.type(dtype=torch.int64).to(device)
            _, features =  model(images)
            old_out, old_features = old_model(images)
            old_outputs.append(torch.cat(list(old_out.values()), dim=1))
            features_list.append(features) 
            old_features_list.append(old_features)
            labels_list.append(labels)
        
        old_outputs = torch.cat(old_outputs, dim=0)
        old_features  = torch.cat(old_features_list)
        new_features  = torch.cat(features_list)
        labels_list = torch.cat(labels_list)
        
    return new_features, old_features 




                    