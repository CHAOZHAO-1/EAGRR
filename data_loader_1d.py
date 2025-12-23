import torch
import numpy as np
from scipy.fft import fft
import scipy.io as scio

# 数据归一化（Z-score标准化）
def zscore(Z):
    for i in range(Z.shape[1]):
        Zmax, Zmin = Z[:, i, :].max(axis=1), Z[:, i, :].min(axis=1)
        diff = Zmax - Zmin
        diff[diff == 0] = 1
        Z[:, i, :] = (Z[:, i, :] - Zmin[:, np.newaxis]) / diff[:, np.newaxis]
    return Z



def l2_normalize_per_channel(Z):
    """
    对输入张量 Z 的每个通道进行 L2 归一化。
    输入:
        Z: numpy array，形状为 [batch_size, num_channels, feature_dim]
    输出:
        Z_normed: L2归一化后的张量
    """
    for i in range(Z.shape[1]):
        norm = np.linalg.norm(Z[:, i, :], axis=1, keepdims=True)
        norm[norm == 0] = 1  # 避免除以0
        Z[:, i, :] = Z[:, i, :] / norm
    return Z


# Min-Max归一化 + log压缩
def min_max(Z):
    Zmin = np.min(Z, axis=-1, keepdims=True)
    return np.log(Z - Zmin + 1)

# 加载单个源数据并处理
def load_single_source(data, key, fft_enabled):
    
    X = data[key]
    
    

    if fft_enabled=='FRE':
        
        

        X = abs(fft(X, axis=-1))[:, :, :1024]  # 对最后一维做 FFT，并取前 1024 个频率点
        
       
        
        X = min_max(X)

        return zscore(X)


    if fft_enabled == 'TIME':
        
        # return zscore(X)

        return l2_normalize_per_channel(X)



    if fft_enabled == 'TIME_FRE':

        TIME_DATA=zscore(X)

        X = abs(fft(X, axis=-1))[:, :, :1024]  # 对最后一维做 FFT，并取前 1024 个频率点

        FRE_DATA = min_max(X)

        ALL = np.concatenate((TIME_DATA, FRE_DATA), axis=1)


        return ALL


# 构造训练集
def load_training(root_path, src_list, fft_enabled, class_num, batch_size, kwargs):
    
    class_per_source = 300
    
    data = scio.loadmat(root_path)
    
   
    
  

    all_features = []
    all_labels = []

    for domain_id, src in enumerate(src_list):
        
        key = 'load' + str(src)
     
        
       
        
        X = load_single_source(data, key, fft_enabled)
     
        
        
      
        
        y = torch.zeros((class_per_source * class_num, 2))
        y[:, 0] = torch.arange(class_per_source * class_num) // class_per_source
        y[:, 1] = domain_id

        all_features.append(X)
        all_labels.append(y)

    features = np.vstack(all_features)
    labels = torch.cat(all_labels, dim=0).long()
    
    features = torch.tensor(features, dtype=torch.float32)
    

    dataset = torch.utils.data.TensorDataset(features, labels)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, **kwargs)

# 构造测试集

def load_testing(root_path, tgt_list, fft_enabled, class_num, batch_size, kwargs):
    
    class_per_source = 300
    
    data = scio.loadmat(root_path)

    all_features = []
    all_labels = []

    for domain_id, src in enumerate(tgt_list):
        
        key = 'load' + str(src)
        
               
        X = load_single_source(data, key, fft_enabled)       
            
        
       
        y = torch.arange(class_per_source * class_num) // class_per_source
       

        all_features.append(X)
        all_labels.append(y)

    features = np.vstack(all_features)
    labels = torch.cat(all_labels, dim=0).long()
    features = torch.tensor(features, dtype=torch.float32)

    dataset = torch.utils.data.TensorDataset(features, labels)
    
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False, **kwargs)

