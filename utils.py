# This program is free software; you can redistribute it and/or modify it under the terms of the GNU Affero General Public License version 3 as published by the Free Software Foundation:
# http://www.gnu.org/licenses/agpl-3.0.txt
############################################################

from torch.utils.data import Dataset
import torch
import random

class CustomMatrixDataset(Dataset):
    def __init__(self, patient_dict, labels, nb_patch, fixed=False):
        self.mat_dict = patient_dict
        self.nb_patch=nb_patch
        self.labels=labels
        self.fixed=fixed
        if self.fixed:
            self.patch_per_patient={}
    
    def random_patch_selection(self, patch_list, slide):
        if len(patch_list) >= slef.nb_patch:
            patch_sel=random.sample(patch_list, self.nb_patch)
        else:
            patch_sel=patch_list
            n=self.nb_patch-len(patch_list)
            while n>len(patch_list):
                patch_sel.extend(patch_list)
                n-=len(patch_list)
            last_patch_sel=random.sample(patch_list, n)
            patch_sel.extend(last_patch_sel)
        if fixed:
            self.patch_per_patient[slide]=patch_sel
        return(patch_sel)
    

    def __len__(self):
        return len(self.mat_dict)
    
    def __getitem__(self, idx):
        slide=list(self.mat_dict.keys())[idx]
        
        patch_list = self.mat_dict[slide]
        label = torch.tensor(self.labels.loc[self.labels['patient_id'] == slide]['label'].iloc[0]).to(torch.int64)
        
        if self.fixed:
            if slide in self.patch_per_patient.keys():
                patch_sel=patch_per_patient[slide]
            else:
                patch_sel=random_patch_selection(patch_list, slide)
        else:
            patch_sel=random_patch_selection(patch_list, slide)
        
        return torch.stack(patch_sel), label


class CustomMatrixDataset_augmentation(Dataset):
    def __init__(self, patient_patch_dict, labels, nb_patch, augmentations, encoder, fixed=False):
        self.mat_dict = patient_dict
        self.nb_patch=nb_patch
        self.labels=labels
        self.fixed=fixed
        if self.fixed:
            self.patch_per_patient={}
    
    def random_patch_selection(self, patch_list, slide):
        if len(patch_list) >= slef.nb_patch:
            patch_sel=random.sample(patch_list, self.nb_patch)
        else:
            patch_sel=patch_list
            n=self.nb_patch-len(patch_list)
            while n>len(patch_list):
                patch_sel.extend(patch_list)
                n-=len(patch_list)
            last_patch_sel=random.sample(patch_list, n)
            patch_sel.extend(last_patch_sel)
        if fixed:
            self.patch_per_patient[slide]=patch_sel
        return(patch_sel)
    

    def __len__(self):
        return len(self.mat_dict)
    
    def __getitem__(self, idx):
        slide=list(self.mat_dict.keys())[idx]
        
        patch_list = self.mat_dict[slide]
        label = torch.tensor(self.img_labels.loc[self.img_labels['patient_id'] == slide]['label'].iloc[0]).to(torch.int64)
        
        if self.fixed:
            if slide in self.patch_per_patient.keys():
                patch_sel=patch_per_patient[slide]
            else:
                patch_sel=random_patch_selection(patch_list, slide)
        else:
            patch_sel=random_patch_selection(patch_list, slide)
        
        return torch.stack(patch_sel), label

class BalancedBatchSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, dataset, bs, labels=None):
        self.labels = labels
        self.dataset = dict()
        self.bs=bs
        self.balanced_min = 100000
        # Save all the indices for all the classes
        for idx in range(0, len(dataset)):
            label = self._get_label(dataset, idx)
            if label not in self.dataset:
                self.dataset[label] = list()
            self.dataset[label].append(idx)
        
        self.keys = list(self.dataset.keys())
        self.balanced_min = min([len(self.dataset[key]) for key in self.keys])
        self.chosen_idxs={}
        self.currentkey = 0
        self.n_batch=0
        
        self.nb_batch_max=self.balanced_min*len(self.keys)//self.bs
        if self.balanced_min*len(self.keys)%self.bs != 0:
            self.nb_batch_max+=1
        
    def _get_idxs(self):
        chosen_idxs={}
        for key in self.keys:
            idxs=random.sample(range(len(self.dataset[key])), self.balanced_min)
            chosen_idxs[key]=[self.dataset[key][chosen_idx] for chosen_idx in idxs]

        return(chosen_idxs)
        

    def __iter__(self):
        
        if self.n_batch==0:
            self.chosen_idxs=self._get_idxs()

        for i in range(self.balanced_min):
            for key in self.keys:
                yield self.chosen_idxs[key][i]
            if i%16==0:
                self.n_batch+=1
            if self.n_batch==self.nb_batch_max:
                self.n_batch=0
    
    def _get_label(self, dataset, idx, labels = None):
        if self.labels is not None:
            return int(self.labels[idx])
        else:
            # Trying guessing
            dataset_type = type(dataset)
            if is_torchvision_installed and dataset_type is torchvision.datasets.MNIST:
                return dataset.train_labels[idx].item()
            elif is_torchvision_installed and dataset_type is torchvision.datasets.ImageFolder:
                return dataset.imgs[idx][1]
            else:
                raise Exception("You should pass the tensor of labels to the constructor as second argument")

    def __len__(self):
        return self.balanced_min*len(self.keys)
    
