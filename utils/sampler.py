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