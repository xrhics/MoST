from torch.utils.data import DataLoader
from data_provider.data_loader import MM_data, MultiDataModule


def data_provider(args, flag):
    if flag == 'test':
        shuffle_flag = False
        batch_size = args.batch_size
    else:
        shuffle_flag = True
        batch_size = args.batch_size
    Data = MM_data
    data_args = args.data
    model_args = args.model
    data_set = Data(
            root_path=data_args.root_path,
            stage=flag,
            dataset=args.data_id,
            data_avaliage=data_args.data_avaliage,
            input_len=model_args.seq_len,
            pred_len=model_args.pred_len,
            state=data_args.state,
            interval=data_args.interval,
            args=args
        )
    MultiData = MultiDataModule(
        data_set,
        shuffle=shuffle_flag,
        num_workers=data_args.num_workers,
        pin_memory=data_args.pin_memory,
        persistent_workers=data_args.persistent_workers,
        batch_size=batch_size
        )
    
    data_loader = MultiData._get_dataloader(data_set, shuffle_flag)
    args.logger.info('Mode: {}, Sample Num: {}, Batch Num: {}'.format(flag, len(data_set), len(data_loader)))
    return data_set, data_loader
