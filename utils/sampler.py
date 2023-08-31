import pandas as pd
import torch
from torch.utils.data.sampler import WeightedRandomSampler
import os

def getSampler(root_path, label_column):
    df = pd.read_pickle(os.path.join(root_path,[filename for filename in os.listdir(root_path) if 'TRAIN' in filename][0]))
    class_occ = pd.DataFrame()
    class_occ['Occ'] = df[label_column].value_counts().sort_index()
    class_occ['weight'] = class_occ['Occ'].apply(lambda x: 1./x)
    #print(class_occ)
    
    weights_dict = class_occ['weight'].to_dict()
    df['weight'] = df[label_column].apply(lambda x: weights_dict[x])
    #print(df)
    
    weights = torch.DoubleTensor(df.weight.values)
    #print('weights: ', weights)

    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
    return sampler

def countLabels(dataloader):
    class_names = ['F0L', 'F0M', 'F1L', 'F1M', 'F2L', 'F2M', 'F3L', 'F3M', 'F4L', 'F4M', 'F5L', 'F5M', 'F6L', 'F6M', 'F7L', 'F7M']
    labels_count = {x: 0 for x in class_names}

    for _, labels, _ in dataloader:
        for idx in range(len(class_names)):
            labels_count[class_names[idx]] += torch.sum(labels == idx).item()
        
    return labels_count