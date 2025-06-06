import os
import random
import string
import shutil
import json
import random
import numpy as np
import torch
import pandas as pd

from copy import deepcopy


def seed_everything(seed=0):
    """Fix all random seeds"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True


def experiment_folder(root_path, dev_mode, approach_name):
    if os.path.exists(os.path.join(root_path, 'exp_folder')):
        shutil.rmtree(os.path.join(root_path, 'exp_folder'), ignore_errors=True)

    if dev_mode:
        exp_folder = 'exp_folder'
    else:
        exp_folder = approach_name + '_' + ''.join(random.choices(string.ascii_letters + string.digits, k=8))

    out_path = os.path.join(root_path, exp_folder)
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    return out_path, exp_folder

def get_class_per_task(n_class_first_task, total_classes, n_task):
    assert n_class_first_task <= total_classes
    
    if n_class_first_task > -1:
        class_per_task = int((total_classes - n_class_first_task)/(n_task - 1))
        assert class_per_task > 1 
        assert n_class_first_task + (n_task - 1) * class_per_task  == total_classes
    else:
        class_per_task =  int(total_classes/ n_task)   
        assert class_per_task > 1 
        assert n_task * class_per_task == total_classes

    return class_per_task

def result_folder(out_path, name):
    if not os.path.exists(os.path.join(out_path, name)):
        os.mkdir(os.path.join(out_path, name))


def store_params(out_path, n_epoch, bs, n_task, old_reconstruction, loss_weight):
    params = {}
    params['n_epoch'] = n_epoch
    params['bs'] = bs
    params['n_task'] = n_task
    params['old_reconstruction'] = old_reconstruction
    params['loss_weight'] = loss_weight
    store_dictionary(params, out_path, name='params')


def get_task_dict(n_task, total_classes, class_per_task, n_class_first_task):
    d = {}
    l = list(range(total_classes))
    
    if n_class_first_task > - 1:
        offset = n_class_first_task
        for i in range(n_task):
            if i == 0:
                d[i] = [i for i in range(0, n_class_first_task)]
            else:
                d[i] = l[offset + (i-1)*class_per_task: offset+ (i-1)* class_per_task + class_per_task]     
            
    else:
      
        for i in range(n_task):
            d[i] = l[i*class_per_task:  i* class_per_task + class_per_task]     
               
    return d 



            


def store_dictionary(d, out_path, name):
    d={str(k): v for k, v in d.items()}
    with open(os.path.join(out_path, name+'.json'), 'w', encoding='utf-8') as f:
        json.dump(d, f, ensure_ascii=False, indent=4)


def rollback_model(approach, model_path, device):
    approach.model.load_state_dict(torch.load(model_path, map_location=device))
    print("Model Loaded {}".format(model_path))
 


def store_model(approach, out_path, name=""):
    torch.save(deepcopy(approach.model.state_dict()), os.path.join(out_path, name+"_model.pth"))


def remap_targets(train_set, test_set, total_classes, dataset):
    l = list(range(total_classes))
    l_sorted = deepcopy(l)
    
    if dataset=="imagenet-1k":
         l = [
            54, 7, 894, 512, 126, 337, 988, 11, 284, 493, 133, 783, 192, 979, 622, 215, 240, 548, 238, 419, 274, 108,
            928, 856, 494, 836, 473, 650, 85, 262, 508, 590, 390, 174, 637, 288, 658, 219, 912, 142, 852, 160, 704, 289,
            123, 323, 600, 542, 999, 634, 391, 761, 490, 842, 127, 850, 665, 990, 597, 722, 748, 14, 77, 437, 394, 859,
            279, 539, 75, 466, 886, 312, 303, 62, 966, 413, 959, 782, 509, 400, 471, 632, 275, 730, 105, 523, 224, 186,
            478, 507, 470, 906, 699, 989, 324, 812, 260, 911, 446, 44, 765, 759, 67, 36, 5, 30, 184, 797, 159, 741, 954,
            465, 533, 585, 150, 101, 897, 363, 818, 620, 824, 154, 956, 176, 588, 986, 172, 223, 461, 94, 141, 621, 659,
            360, 136, 578, 163, 427, 70, 226, 925, 596, 336, 412, 731, 755, 381, 810, 69, 898, 310, 120, 752, 93, 39,
            326, 537, 905, 448, 347, 51, 615, 601, 229, 947, 348, 220, 949, 972, 73, 913, 522, 193, 753, 921, 257, 957,
            691, 155, 820, 584, 948, 92, 582, 89, 379, 392, 64, 904, 169, 216, 694, 103, 410, 374, 515, 484, 624, 409,
            156, 455, 846, 344, 371, 468, 844, 276, 740, 562, 503, 831, 516, 663, 630, 763, 456, 179, 996, 936, 248,
            333, 941, 63, 738, 802, 372, 828, 74, 540, 299, 750, 335, 177, 822, 643, 593, 800, 459, 580, 933, 306, 378,
            76, 227, 426, 403, 322, 321, 808, 393, 27, 200, 764, 651, 244, 479, 3, 415, 23, 964, 671, 195, 569, 917,
            611, 644, 707, 355, 855, 8, 534, 657, 571, 811, 681, 543, 313, 129, 978, 592, 573, 128, 243, 520, 887, 892,
            696, 26, 551, 168, 71, 398, 778, 529, 526, 792, 868, 266, 443, 24, 57, 15, 871, 678, 745, 845, 208, 188,
            674, 175, 406, 421, 833, 106, 994, 815, 581, 676, 49, 619, 217, 631, 934, 932, 568, 353, 863, 827, 425, 420,
            99, 823, 113, 974, 438, 874, 343, 118, 340, 472, 552, 937, 0, 10, 675, 316, 879, 561, 387, 726, 255, 407,
            56, 927, 655, 809, 839, 640, 297, 34, 497, 210, 606, 971, 589, 138, 263, 587, 993, 973, 382, 572, 735, 535,
            139, 524, 314, 463, 895, 376, 939, 157, 858, 457, 935, 183, 114, 903, 767, 666, 22, 525, 902, 233, 250, 825,
            79, 843, 221, 214, 205, 166, 431, 860, 292, 976, 739, 899, 475, 242, 961, 531, 110, 769, 55, 701, 532, 586,
            729, 253, 486, 787, 774, 165, 627, 32, 291, 962, 922, 222, 705, 454, 356, 445, 746, 776, 404, 950, 241, 452,
            245, 487, 706, 2, 137, 6, 98, 647, 50, 91, 202, 556, 38, 68, 649, 258, 345, 361, 464, 514, 958, 504, 826,
            668, 880, 28, 920, 918, 339, 315, 320, 768, 201, 733, 575, 781, 864, 617, 171, 795, 132, 145, 368, 147, 327,
            713, 688, 848, 690, 975, 354, 853, 148, 648, 300, 436, 780, 693, 682, 246, 449, 492, 162, 97, 59, 357, 198,
            519, 90, 236, 375, 359, 230, 476, 784, 117, 940, 396, 849, 102, 122, 282, 181, 130, 467, 88, 271, 793, 151,
            847, 914, 42, 834, 521, 121, 29, 806, 607, 510, 837, 301, 669, 78, 256, 474, 840, 52, 505, 547, 641, 987,
            801, 629, 491, 605, 112, 429, 401, 742, 528, 87, 442, 910, 638, 785, 264, 711, 369, 428, 805, 744, 380, 725,
            480, 318, 997, 153, 384, 252, 985, 538, 654, 388, 100, 432, 832, 565, 908, 367, 591, 294, 272, 231, 213,
            196, 743, 817, 433, 328, 970, 969, 4, 613, 182, 685, 724, 915, 311, 931, 865, 86, 119, 203, 268, 718, 317,
            926, 269, 161, 209, 807, 645, 513, 261, 518, 305, 758, 872, 58, 65, 146, 395, 481, 747, 41, 283, 204, 564,
            185, 777, 33, 500, 609, 286, 567, 80, 228, 683, 757, 942, 134, 673, 616, 960, 450, 350, 544, 830, 736, 170,
            679, 838, 819, 485, 430, 190, 566, 511, 482, 232, 527, 411, 560, 281, 342, 614, 662, 47, 771, 861, 692, 686,
            277, 373, 16, 946, 265, 35, 9, 884, 909, 610, 358, 18, 737, 977, 677, 803, 595, 135, 458, 12, 46, 418, 599,
            187, 107, 992, 770, 298, 104, 351, 893, 698, 929, 502, 273, 20, 96, 791, 636, 708, 267, 867, 772, 604, 618,
            346, 330, 554, 816, 664, 716, 189, 31, 721, 712, 397, 43, 943, 804, 296, 109, 576, 869, 955, 17, 506, 963,
            786, 720, 628, 779, 982, 633, 891, 734, 980, 386, 365, 794, 325, 841, 878, 370, 695, 293, 951, 66, 594, 717,
            116, 488, 796, 983, 646, 499, 53, 1, 603, 45, 424, 875, 254, 237, 199, 414, 307, 362, 557, 866, 341, 19,
            965, 143, 555, 687, 235, 790, 125, 173, 364, 882, 727, 728, 563, 495, 21, 558, 709, 719, 877, 352, 83, 998,
            991, 469, 967, 760, 498, 814, 612, 715, 290, 72, 131, 259, 441, 924, 773, 48, 625, 501, 440, 82, 684, 862,
            574, 309, 408, 680, 623, 439, 180, 652, 968, 889, 334, 61, 766, 399, 598, 798, 653, 930, 149, 249, 890, 308,
            881, 40, 835, 577, 422, 703, 813, 857, 995, 602, 583, 167, 670, 212, 751, 496, 608, 84, 639, 579, 178, 489,
            37, 197, 789, 530, 111, 876, 570, 700, 444, 287, 366, 883, 385, 536, 460, 851, 81, 144, 60, 251, 13, 953,
            270, 944, 319, 885, 710, 952, 517, 278, 656, 919, 377, 550, 207, 660, 984, 447, 553, 338, 234, 383, 749,
            916, 626, 462, 788, 434, 714, 799, 821, 477, 549, 661, 206, 667, 541, 642, 689, 194, 152, 981, 938, 854,
            483, 332, 280, 546, 389, 405, 545, 239, 896, 672, 923, 402, 423, 907, 888, 140, 870, 559, 756, 25, 211, 158,
            723, 635, 302, 702, 453, 218, 164, 829, 247, 775, 191, 732, 115, 331, 901, 416, 873, 754, 900, 435, 762,
            124, 304, 329, 349, 295, 95, 451, 285, 225, 945, 697, 417
        ]
         
    elif dataset == "domainnet":
         pass # not shuffle the dataset in case of domainet
     
    else:
        random.shuffle(l)
    
    label_mapping = dict(zip(l_sorted, l))
    
    if dataset == "domainnet":
        # is not necessary to remap label for domainet
        return train_set, test_set, label_mapping
    
    # remap train labels following label_mapping    
    
    for i in range(len(train_set.targets)):
        train_set.targets[i]=label_mapping[train_set.targets[i]]
    
 
    for key in train_set.class_to_idx.keys():
        train_set.class_to_idx[key] = label_mapping[train_set.class_to_idx[key]]
        
    # remap test labels following label_mapping    
    
    for i in range(len(test_set.targets)):
        test_set.targets[i]=label_mapping[test_set.targets[i]]
    
 
    for key in test_set.class_to_idx.keys():
        test_set.class_to_idx[key] = label_mapping[test_set.class_to_idx[key]]
        
        
     
    return train_set, test_set, label_mapping

