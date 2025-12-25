import yaml
import os
import logging
import sys
import torch
import numpy as np
import torch.nn as nn
import random
import datetime

def is_main_process():
    return os.environ.get('RANK') == '0'

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def parsing_syntax(unknown):
    unknown_dict = {}
    key = None
    for arg in unknown:
        if arg.startswith('--'):
            key = arg.lstrip('--')
            unknown_dict[key] = None
        else:
            if key:
                unknown_dict[key] = arg
                key = None
    return unknown_dict


class ConfigDict(dict):
    def __init__(self, *args, **kwargs):
        super(ConfigDict, self).__init__(*args, **kwargs)
        for key, value in self.items():
            if isinstance(value, dict):
                self[key] = ConfigDict(value)
            if key == 'data' and isinstance(value, str):
                dataset_config = load_config("../Model_Config/dataset_config/{}".format(value + ".yaml"))
                self[key]= ConfigDict(dataset_config)

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError:
            raise AttributeError(f"'ConfigDict' object has no attribute '{item}'")

    def __setattr__(self, key, value):
        self[key] = value


def update_config(config, unknown_args):
    for key, value in unknown_args.items():
        config_path = key.split('-')
        cur = config
        for node in config_path:
            assert node in cur.keys(), "path not exist"
            if isinstance(cur[node], ConfigDict):
                cur = cur[node]
            else:
                try:
                    cur[node] = eval(value)
                except NameError:
                    cur[node] = value
    return config



def fix_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_logger(log_dir, to_stdout=True):
    curr_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = 'info_' + curr_time + '.log'
    logger = logging.getLogger(log_dir)
    logger.setLevel('INFO')
    # Add console handler.
    if to_stdout:
        console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
    # Add file handler and stdout handler
    if log_dir:
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%m-%d %H:%M')
        file_handler = logging.FileHandler(os.path.join(log_dir, log_filename))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.info('Log directory: %s', log_dir)

    return logger

def main_process_check(func):
    def wrapper(self, *args, **kwargs):
        pass
    return wrapper