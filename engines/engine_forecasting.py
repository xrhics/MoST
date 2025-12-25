import os
import torch
import warnings
import numpy as np
import math
import torch.nn as nn
warnings.filterwarnings('ignore')
from tqdm import tqdm
from utils.metrics import metric
from thop import profile, clever_format
import time
import gc

class Engine_Forecasting(object):
    def __init__(self, args):
        self.args = args
        self.data_id = args.data_id + '_' + str(args.seq_len) + '_' + str(args.pred_len)
        self.criterion = torch.nn.MSELoss()
        self.alpha= args.train.alpha

    def train_batch(self, batch, model, optimizer, batch_num):
        model.train()
        optimizer.zero_grad()
        max_buffer_size = 5
        all_loss = 0
        num_loaders = len(batch)
        node_num = batch[0].shape[1]
        x, start_days , start_intervals, poi, satellite, loc, adj, y, mean, std, state ,dataset,topk_indices, topk_values= [
            t.to(self.args.device) if isinstance(t, torch.Tensor) else t for t in batch
        ]
        assert not torch.isnan(y).any(), "y contains NaN"
        assert not torch.isinf(x).any(), "x contains INF"
        with torch.cuda.amp.autocast(dtype=torch.bfloat16): 
            y_hat, multimodal, multimodal1, multimodal2, balance_loss = model(x,start_days,start_intervals, poi, satellite, loc, adj, mean, std, state, topk_indices, topk_values)
        if self.args.is_norm:
            mean_std = np.load(os.path.join(self.args.data.root_path, "data","mean_std.npy"),allow_pickle=True).item() 
            mean,std=mean_std[dataset][0],mean_std[dataset][1]
            y_hat = y_hat * std + mean
            y = y * std + mean
        pre_loss = model.get_loss(y, y_hat) 
        if multimodal1 is not None:
            con_loss = model.get_contrastive_loss(multimodal, multimodal1, multimodal2)
        else:
            con_loss = torch.tensor(0.0)
        
        if balance_loss is not None:
            balance_loss=10 * (balance_loss)
        else:
            balance_loss = torch.tensor(0.0)
        total_loss = (pre_loss + 0.1*con_loss + balance_loss)   
        total_loss.backward() 
        optimizer.step()
        lr_now = optimizer.param_groups[0]['lr']
        all_loss = total_loss.item()    
        con_loss=con_loss.item()
        return all_loss, pre_loss.item(), con_loss, float(balance_loss), lr_now
    

    def valid(self, valid_loader, model):
        mean=[]
        std=[]
        preds = []
        trues = []
        val_losses = []
        validation_outputs = {'preds': [], 'targets': [],'mean':[],'std':[]}
        valid_loss = []
        model.eval()
        all_loss = 0
        with torch.no_grad():
            for i, batch in tqdm(enumerate(valid_loader)):
                x, start_days,start_intervals,poi, satellite, loc, adj, y, mean, std, state,dataset, topk_indices, topk_values = [
            t.to(self.args.device) if isinstance(t, torch.Tensor) else t for t in batch
        ]
                with torch.cuda.amp.autocast(dtype=torch.bfloat16): 
                    y_hat, multimodal, multimodal1, multimodal2, balance_loss = model(x,start_days,start_intervals, poi, satellite, loc, adj, mean, std, state, topk_indices, topk_values)
                pre_loss=model.get_loss(y, y_hat)
                valid_loss.append(pre_loss)
                preds.append(y_hat.detach().cpu().numpy())
                trues.append(y.detach().cpu().numpy())

        preds = np.concatenate(preds, 0)
        trues = np.concatenate(trues, 0)
        if self.args.is_norm:
            mean_std = np.load(os.path.join(self.args.data.root_path, "data","mean_std.npy"),allow_pickle=True).item() 
            vali_dataset=dataset
            mean,std=mean_std[vali_dataset][0],mean_std[vali_dataset][1]
            preds = preds * std + mean
            trues = trues * std + mean
        mae, mse, rmse, mape, mspe, smape = metric(preds, trues)
        self.args.logger.info('Setting: {}, RMSE: {:.6f}, MAE: {:.6f}, MAPE:{:.6f}, SMAPE:{:.6f}'.format(self.data_id, rmse, mae, mape, smape))

        valid_loss = torch.tensor(valid_loss) 
        valid_loss = np.average(valid_loss.cpu())
        return valid_loss, mae,rmse,mape, smape

    def test(self, test_loader, model):
        preds = []
        trues = []
        model.eval()
        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                x,start_days,start_intervals, poi, satellite, loc, adj, y, mean, std, state ,dataset, topk_indices, topk_values= [
            t.to(self.args.device) if isinstance(t, torch.Tensor) else t for t in batch
        ]
                with torch.cuda.amp.autocast(dtype=torch.bfloat16): 
                    y_hat, multimodal, multimodal1, multimodal2, balance_loss= model(x, start_days,start_intervals,poi, satellite, loc, adj, mean, std, state, topk_indices, topk_values)
                pre_loss=model.get_loss(y, y_hat)
                pre_loss = model.get_loss(y, y_hat)
                pre_loss1 = 0
                outputs = y_hat.detach().cpu().numpy()
                batch_y = y.detach().cpu().numpy()
                preds.append(outputs)
                trues.append(batch_y)

        preds = np.concatenate(preds, 0)
        trues = np.concatenate(trues, 0)
        np.savez('nmost_chicago_predictions.npz', preds=preds, trues=trues)
        if self.args.is_norm:
            mean_std = np.load(os.path.join(self.args.data.root_path, "data","mean_std.npy"),allow_pickle=True).item() 
            test_dataset=dataset
            mean,std=mean_std[test_dataset][0],mean_std[test_dataset][1]
            preds = preds * std + mean
            trues = trues * std + mean
        mae, mse, rmse, mape, mspe, smape = metric(preds, trues)
        self.args.logger.info('Setting: {}, RMSE: {:.6f}, MAE: {:.6f}, MAPE:{:.6f}, SMAPE:{:.6f}'.format(self.data_id, rmse, mae, mape, smape))

        f = open(os.path.join(self.args.checkpoint, 'result_s' + str(self.args.seed) + '.txt'), 'a')
        f.write(self.data_id + '\n')
        f.write('RMSE: {}, MAE: {}'.format(rmse, mae))
        f.write('\n')
        f.write('\n')
        f.close()

