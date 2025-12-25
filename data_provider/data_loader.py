import os
import re
import glob
import torch
import numpy as np
import pandas as pd
import json
from datetime import datetime,timedelta
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset, random_split

class MultiDataModule:
    def __init__(self, 
                 datasets, 
                 shuffle, 
                 num_workers, 
                 pin_memory,
                 persistent_workers,
                 batch_size=32):
        super().__init__()
        self.datasets = datasets  
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.train_steps = None  
    
    def _get_dataloader(self, dataset_groups, shuffle=False):
        func=self.collate
        loader = DataLoader(
                dataset_groups,
                batch_size=self.batch_size,
                shuffle=shuffle,
                num_workers=self.num_workers,
                collate_fn=func,
                pin_memory=self.pin_memory,
                persistent_workers=self.persistent_workers
            )
        return loader

    

    def collate(self, batch):
        dataset=batch[0][0]
        start_days=torch.stack([item[1] for item in batch])
        start_intervals=torch.stack([item[2] for item in batch])
        x = torch.stack([item[3] for item in batch])  # 堆叠NxD数据
        y = torch.stack([item[4] for item in batch])
        mean= torch.stack([item[5] for item in batch])
        std= torch.stack([item[6] for item in batch])
        poi=self.datasets.city_poi
        satellite=self.datasets.city_satellite
        loc=self.datasets.city_loc
        adj=self.datasets.adj
        state=self.datasets.city_state
        topk_indices=self.datasets.topk_indices
        topk_values=self.datasets.topk_values
        return x, start_days,start_intervals,poi, satellite,loc,adj,y, mean, std, state, dataset, topk_indices, topk_values
  
def get_topk_per_row(adj, k=20):

    partitioned_indices = np.argpartition(-adj, k, axis=1)[:, :k]

    partitioned_values = np.take_along_axis(adj, partitioned_indices, axis=1)

    sorted_indices = np.argsort(-partitioned_values, axis=1)
    topk_indices = np.take_along_axis(partitioned_indices, sorted_indices, axis=1)
    topk_values = np.take_along_axis(partitioned_values, sorted_indices, axis=1)

    topk_values=np.where(topk_values<1e-3, 1e-3, topk_values)
    return topk_indices, topk_values

class MM_data(Dataset):  
    def __init__(self, root_path: str,
                 stage:str="train",
                 dataset="gla",
                 data_avaliage={"poi":1, "map":1, "satellite":1, "location":1},
                 input_len: int = 6,
                 pred_len: int = 6,
                 state= [1,1,1,1],
                 interval: int = 1,
                 args=None):
        self.root_path = root_path
        self.dataset=dataset
        self.stage=stage
        self.interval = interval   
        self.pred_len = pred_len
        self.input_len = input_len
        self.state=state
        self.args=args
        self.load_length()
        self.load_data()
        if self.args.is_norm:
            self.get_mean_std()
        city = self.dataset
        data_dir = os.path.join(self.root_path, "data", city, city+"_"+self.stage)
    def load_length(self):
        city=self.dataset
        file_path=os.path.join(self.root_path,"data",city,city+"_"+self.stage+"1.npz")
        data = np.load(file_path)['data']
        if self.stage == "train":
            L, N = data.shape
            data = data[int(L*(1 - self.args.train_ratio)):, :]
        self.lengths=len(data)-self.input_len-self.pred_len

    def load_data(self): 
        self.his_data={}
        lengths=[]
        self.all_length=0
        city=self.dataset
        def load_index(city):
            file_path=os.path.join(self.root_path,"data",city,city+"_"+self.stage+"1.npz")
            return np.load(file_path)['index'] 
        file_path=os.path.join(self.root_path,"data",city,city+"_"+self.stage+"1.npz")
        data = np.load(file_path)
        self.his_data["timeseries"] = data['data'].T  
        start_day=[]
        start_qarter=[]
        for time_str in data['start_time']:
            dt = datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S')
            start_day.append( dt.weekday())
            start_qarter.append( (dt.hour * 60 + dt.minute) // 15)
        self.his_data["start_day"]=start_day    
        self.his_data["start_time"]=start_qarter
        if self.stage == "train":
            L, N = data['data'].shape
            self.his_data["timeseries"] = self.his_data["timeseries"][:, int(L*(1 - self.args.train_ratio)):]
            self.his_data["start_day"] = self.his_data["start_day"][int(L*(1 - self.args.train_ratio)):]
            self.his_data["start_time"]=self.his_data["start_time"][int(L*(1 - self.args.train_ratio)):]
        
        file_path=os.path.join(self.root_path,"data",city,city+"_rn_adj.npy")
        adj=np.load(file_path) 
        np.fill_diagonal(adj, 0)
        topk_indices, topk_values=get_topk_per_row(adj, k=20)
        self.topk_indices,self.topk_values =torch.from_numpy(topk_indices).long(), torch.from_numpy(topk_values).float() 

        data=pd.read_csv(os.path.join(self.root_path,"poi",city+"_poi_1000_vectors.csv"))  
        self.his_data["poi"]={int(data.loc[i,"list_id"]):np.array(json.loads(data.loc[i,"sentence_vector"])) for i in range(len(data))}
        indices = load_index(city).tolist()  
        poi_values = [self.his_data["poi"][i] for i in indices]  
        self.city_poi = torch.tensor(np.stack(poi_values), dtype=torch.float32) 
        
        data=pd.read_csv(os.path.join(self.root_path,"picture",city,"image_features.csv"))  
        self.his_data["satellite"]={int(data.loc[i,"filename"].split(".")[0]):np.array(json.loads(data.loc[i,"feature_vector"])) for i in range(len(data))}
        satellite_values = [self.his_data["satellite"][i] for i in indices]  
        self.city_satellite = torch.tensor(np.stack(satellite_values), dtype=torch.float32)

        file=os.path.join(self.root_path,"data",city,city+".json")
        data=json.loads(open(file,"r").read())
        self.city_loc = torch.tensor(np.stack(
            [(float(data[key]["lat"]), float(data[key]["lon"])) for key in data if int(key) in indices]), dtype=torch.float32)

        self.city_state = torch.tensor(np.tile(self.state, (len(indices), 1))) 


        adj=torch.from_numpy(adj)
        indices = torch.nonzero(adj).t()  
        values = adj[indices[0], indices[1]]  
        
        sparse_adj = torch.sparse_coo_tensor(
            indices=indices,
            values=values,
            size=adj.shape,  
            dtype=torch.float32
        )

        self.adj = sparse_adj 


    def get_mean_std(self):
        mean_std = np.load(os.path.join(self.root_path, "data","mean_std.npy"),allow_pickle=True).item() 
        mean_std=mean_std[self.dataset]
        self.the_mean = mean_std[0]
        self.the_std = mean_std[1]

    def normalization(self, sample):
        return (sample - self.the_mean) / (self.the_std)

    def __len__(self):
        return self.lengths
        
    # def inverse_transform(self, data):
    #     return data
    
    def __getitem__(self, index):
        city=self.dataset
        stride=self.args.model.MM.stride
        start_day=self.his_data["start_day"][index]
        start_interval=self.his_data["start_time"][index]

        start_intervals,start_days=calculate_patch_time_info(start_day, start_interval, total_intervals=self.input_len, stride_size=stride)

        input_data = self.his_data["timeseries"][:,index:index+self.input_len]
        
        input_data = input_data.astype(np.float32)
        if self.args.is_norm:
            input_data = self.normalization(input_data)

        input_mean = input_data.mean(axis=(-1), keepdims=True)
        input_std = input_data.std(axis=(-1), keepdims=True)
        normalized_input = (input_data - input_mean) / (input_std+1e-8)
        target_data = self.his_data["timeseries"][:,index+self.input_len:index+self.input_len+self.pred_len]
        target_data = target_data.astype(np.float32)
        if self.args.is_norm:
            target_data = self.normalization(target_data)
        return self.dataset,torch.tensor(start_days),torch.tensor(start_intervals), torch.as_tensor(normalized_input),torch.as_tensor(target_data),torch.as_tensor(input_mean),torch.as_tensor(input_std)

def calculate_patch_time_info(start_day_index, start_quarter_index, total_intervals=960, stride_size=96):
    total_patches = total_intervals // stride_size
    patch_time_interval=[(start_quarter_index + i * stride_size) % 96 for i in range(total_patches)]
    patch_day=[(start_day_index + (start_quarter_index + i * stride_size) // 96) % 7 for i in range(total_patches)]
    return np.array(patch_time_interval), np.array(patch_day)