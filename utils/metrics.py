import numpy as np
import torch 

def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred, true):
    pred= torch.tensor(pred)
    true= torch.tensor(true)
    return torch.mean(np.abs(pred - true))


def MSE(pred, true):
    pred= torch.tensor(pred)
    true= torch.tensor(true)
    return torch.mean((pred - true) ** 2)


def RMSE(pred, true):
    return torch.sqrt(MSE(pred, true))


def MAPE(pred, true):
    pred= torch.tensor(pred)
    true= torch.tensor(true)
    mask_value= 1e-3
    mask = torch.gt(true, mask_value)
    pred = torch.masked_select(pred, mask)
    true = torch.masked_select(true, mask)
    return torch.mean(torch.abs(torch.div((true - pred), true)))*100


def SMAPE(y_pred,y_true, return_elements: bool = False) -> float:
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    
    if len(y_true) != len(y_pred):
        raise ValueError(f"y_true({len(y_true)}) ≠ y_pred({len(y_pred)})")
    
    numerator = np.abs(y_true - y_pred)
    denominator = np.abs(y_true) + np.abs(y_pred)
    epsilon = 1e-8  
    safe_denominator = np.where(denominator < epsilon, epsilon, denominator)
    per_element_smape = 200.0 * numerator / safe_denominator
    
    zero_mask = (np.abs(y_true) < epsilon) & (np.abs(y_pred) < epsilon)
    per_element_smape[zero_mask] = 0.0

    return per_element_smape if return_elements else np.mean(per_element_smape)


def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))


def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    smape = SMAPE(pred, true)

    return mae, mse, rmse, mape, mspe, smape
