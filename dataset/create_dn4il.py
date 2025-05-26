import os
from itertools import compress
import numpy as np 
import argparse 

def create_domain(path, classes_per_domain=100, num_tasks=6):
   
    domains = ["clipart", "infograph", "painting",  "quickdraw", "real", "sketch"] * (num_tasks // 6)
    for set_type in ["train", "test"]:
        samples = []
        for i, domain in enumerate(domains):
            with open(f"{path}/{domain}_{set_type}.txt", 'r') as f:
                lines = list(map(lambda x: x.replace("\n", "").split(" "), f.readlines()))
            paths, classes = zip(*lines)
            classes = np.array(list(map(float, classes)))
            offset = classes_per_domain * i
            for c in range(classes_per_domain):
                is_class = classes == c + ((i // 6) * classes_per_domain)
                class_samples = list(compress(paths, is_class))
                samples.extend([*[f"{row} {c + offset}" for row in class_samples]])
                
        with open(f"{path}/cs_{set_type}_{num_tasks}.txt", 'wt') as f:
            for sample in samples:
                f.write(f"{sample}\n")

def get_args():
    parser = argparse.ArgumentParser("Create DN4IL") 
    parser.add_argument("--domainnet_path", type=str, default="../cl_data/DomainNet")
 
    args = parser.parse_args()
    return args 
            
                
if __name__ == "__main__":
    parser = get_args()
    domainnet_path = parser.domainnet_path
 
    assert os.path.exists(domainnet_path), f"Please first download and extract dataset from: http://ai.bu.edu/M3SDA/#dataset into:{domainnet_path}"
  
    create_domain(domainnet_path)
 