import os
import json
import copy
import time
import torch
import random
import numpy as np
import pandas as pd
import configparser
from tqdm import tqdm
from models.MoST import Model
from utils.logger import get_logger
from data_provider.data_factory import data_provider
from engines.engine_forecasting import Engine_Forecasting
import time
from utils.c_adamw import AdamW as C_AdamW
import datetime


class Engine(object):
    def __init__(self, args):
        args.device = torch.device('cuda:{}'.format(args.gpu))
        
        self.model = Model(args=args).to(args.device)


        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=args.train.lr, weight_decay=args.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=args.train.epochs, eta_min=args.train.eta_min)
        self.args = args
        
        self._logger(self.model)
        self._print_trainable_parameters(self.model)
        self.get_model_memory_footprint(self.model)
        self._construct_unified_dataloaders()

    def _logger(self, model):
        dataset=self.args.data.train_dataset_list
        timestamp = datetime.datetime.now().strftime("%m%d_%H%M")
        self.args.checkpoint = 'checkpoint{}_{}_{}_timestamp[{}]'.format(str(self.args.seed),str(self.args.data.val_dataset_list), str(self.model.get_setting(self.args)), timestamp)
        self.args.checkpoint = os.path.join("checkpoints", self.args.checkpoint)

        if not os.path.exists(self.args.checkpoint):
            os.makedirs(self.args.checkpoint)
        logger = get_logger(self.args.checkpoint, __name__, 'record_s' + str(self.args.seed) + '.log')
        logger.info(self.args)
        self.args.logger = logger
        
    def _print_trainable_parameters(self, model):
        freeze = 0
        trainable = 0
        for name, param in model.named_parameters():
            if param.requires_grad:
                trainable += param.nelement()
            else:
                freeze += param.nelement()
        self.args.logger.info('Trainable Params: {}, All Params: {}, Percent: {}'.format(
                              trainable, freeze + trainable, trainable / (freeze + trainable)))

    def get_model_memory_footprint(self, model):
        param_size = 0
        for param in model.parameters():
            param_size += param.numel() * param.element_size() 
        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.numel() * buffer.element_size()  
        size_all_mb = (param_size + buffer_size) / 1024**2  
    
    


    def _construct_unified_dataloaders(self):
        '''
        f = open(self.args.instruct_path)
        instruct_list = json.load(f)
        f.close()
        '''
        args = copy.deepcopy(self.args)
        train_dataset_list=args.data.train_dataset_list
        valid_dataset_list=args.data.val_dataset_list
        test_dataset_list=args.data.test_dataset_list
        self.train_batches = 0
        self.train_loaders = []
        self.train_engines = []
        self.valid_loaders = []
        self.valid_engines = []
        self.test_loaders = []
        self.test_engines = []
        self.test_loaders = []
        self.test_engines = []
        args.seq_len = args.model.seq_len
        args.stride = args.data.batch_size
        args.batch_size = args.data.batch_size
        args.pred_len = args.model.pred_len
        args.data_reader = "mm"
        if self.args.is_training:
            for dataset in train_dataset_list:
                args.data_path = "dataset/"+dataset
                args.data_id = dataset
                eng = Engine_Forecasting(args)
                setting = '{}_{}_{}_{}_{}'.format(args.data_id, args.seq_len, args.pred_len, args.stride, args.batch_size, args.train.lr)
                self.args.logger.info('***** Task: {} *****'.format(setting))
                _, train_loader = data_provider(args, 'train')
                self.train_batches += len(train_loader)
                print(self.train_batches)
                self.train_loaders.append(train_loader)
                self.train_engines.append(eng)
            for dataset in valid_dataset_list:
                args.data_path = "dataset/"+dataset
                args.data_id = dataset
                eng = Engine_Forecasting(args)
                setting = '{}_{}_{}_{}_{}'.format(args.data_id,  args.seq_len, args.pred_len, args.stride, args.batch_size, args.train.lr)
                self.args.logger.info('***** Task: {} *****'.format(setting))
                _, valid_loader = data_provider(args, 'val')
                self.valid_loaders.append(valid_loader)
                self.valid_engines.append(eng)
        for dataset in test_dataset_list:
            args.data_path = "dataset/"+dataset
            args.data_id = dataset
            eng = Engine_Forecasting(args)
            setting = '{}_{}_{}_{}_{}'.format(args.data_id, args.seq_len, args.pred_len, args.stride, args.batch_size, args.train.lr)
            self.args.logger.info('***** Task: {} *****'.format(setting))
            _, test_loader = data_provider(args, 'test')
            self.test_loaders.append(test_loader)
            self.test_engines.append(eng)




    def train(self):
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        non_trainable_params = sum(p.numel() for p in self.model.parameters() if not p.requires_grad)
        print(f"Trainable: {trainable_params}, Non-trainable: {non_trainable_params}")
        self.args.logger.info('Start training!')
        wait = 0
        best_valid_loss = np.array([5] * len(self.valid_loaders))
        best_mae = 10000.0
        for e in range(self.args.train.epochs):
            iterators = [d._get_iterator() for d in self.train_loaders]
            length = len(self.train_loaders)
            batch_cnt = [0] * length
            dataset_lengths=[len(iterators[i]) for i in range(length)]
            probabilities = np.array(dataset_lengths) / sum(dataset_lengths)
            # train
            t1 = time.time()
            train_loss = []
            gate_weights_buffer=[]
            batch_bar = tqdm(total=self.train_batches, desc=f'Epoch {e+1}', position=1, leave=False)
            mean_std = np.load(os.path.join(self.args.data.root_path, "data","mean_std.npy"),allow_pickle=True).item() 

            while True:
                idx = np.random.choice(length, p=probabilities)
                try:
                    loader = iterators[idx]
                    batch = next(loader)
                    all_loss, pre_loss, con_loss, balance_loss,lr_now = self.train_engines[idx].train_batch(batch, self.model, self.optimizer,batch_bar.last_print_n)
                    train_loss.append(all_loss)
                    batch_cnt[idx] += 1
                    batch_bar.update(1)  
                    batch_bar.set_postfix(loss=f"{all_loss:.4f}")  
                except StopIteration:
                    continue
                batch_num=sum(batch_cnt)
                if batch_num % 5000 == 0:
                    self.args.logger.info('all_loss: {}, avgpre_loss: {:.6f}, avgcon_loss: {:.6f}, balance_loss: {:.6f}, lr: {:.6f}'.format(all_loss, pre_loss, con_loss, balance_loss, lr_now))
                if  batch_num >= self.train_batches:
                    batch_bar.close()  
                    break
            mtrain_loss = np.mean(train_loss)
            t2 = time.time()
            self.args.logger.info('Epoch: {}, Train Time: {:.4f}, Train Loss: {:.6f}'.format(e + 1, t2 - t1, mtrain_loss))

            # valid
            v1 = time.time()
            valid_loss = []
            mae_all=[]
            mse_all=[]
            mape_all=[]
            smape_all=[]
            for loader, eng in zip(self.valid_loaders, self.valid_engines):
                loss,mae, mse, mape, smape = eng.valid(loader, self.model)
                valid_loss.append(loss)
                mae_all.append(mae)
                mse_all.append(mse)
                mape_all.append(mape)
                smape_all.append(smape)
            valid_loss = np.array(valid_loss)
            mvalid_loss = np.mean(valid_loss)
            mae_all = np.array(mae_all)
            mae_all = np.mean(mae_all)
            mse_all = np.array(mse_all)
            mse_all = np.mean(mse_all)
            mape_all = np.array(mape_all)
            smape_all = np.array(smape_all)
            improve = np.sum((best_valid_loss - valid_loss) / best_valid_loss)
            v2 = time.time()
            self.args.logger.info('Epoch: {}, Valid Time: {:.6f}, Valid Loss: {:.6f}, Valid Loss Improve: {:.6f}'.format(e + 1, v2 - v1, mvalid_loss, improve))
            mae_improve = np.sum((float(best_mae) - float(mae_all)) / best_mae)
            if mae_improve >= 0:
                torch.save(self.model.state_dict(), os.path.join(self.args.checkpoint, 'model_s' + str(self.args.seed) + '.pth'))
                self.args.logger.info('Saving best model!')
                best_mae = mae_all
                wait = 0
            else:
                torch.save(self.model.state_dict(), os.path.join(self.args.checkpoint, 'model_s' + str(self.args.seed) + '_e' + str(e + 1) + '.pth'))
                wait += 1
                if wait == self.args.train.patience:
                    self.args.logger.info('Early stop at epoch {}'.format(e + 1))
                    break

            self.scheduler.step()
            self.args.logger.info('Update learning rate to {}'.format(self.scheduler.get_last_lr()[0]))

        self.test(flag=1)


    def test(self,flag=0):
        self.args.logger.info('Start testing!')
        if flag==0:
            path = self.args.eval_model_path
        else:
            path = os.path.join(self.args.checkpoint, 'model_s' + str(self.args.seed) + '.pth')
        print(path)
        self.model.load_state_dict(torch.load(path))

        for loader, eng in zip(self.test_loaders, self.test_engines):
            eng.test(loader, self.model)

    def fine_tune(self):
        path = self.args.pretrain_model_path
        self.model.load_state_dict(torch.load(path))
        self.train()