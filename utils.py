#author:zhaochao time:2021/5/18

import torch as t
import torch.nn.functional as F
import numpy as np
import  random
import torch.nn as nn




def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    t.manual_seed(seed)  # cpu
    t.cuda.manual_seed_all(seed)  # 并行gpu
    t.backends.cudnn.deterministic = True  # cpu/gpu结果一致
    t.backends.cudnn.benchmark = True  # 训练集变化不大时使训练加速






def log(name1,name2,Train_Loss_list,Train_Accuracy_list,Test_Loss_list,Test_Accuracy_list,Train_Time):
    f = open('./'+name1+'/'+name2+'.txt', 'w')

    f.write('train_loss:')
    f.write(str(Train_Loss_list))
    f.write('\r\n')
    f.write('train_acc:')
    f.write(str(Train_Accuracy_list))
    f.write('\r\n')
    f.write('test_loss:')
    f.write(str(Test_Loss_list))
    f.write('\r\n')
    f.write('test_acc:')
    f.write(str(Test_Accuracy_list))
    f.write('\r\n')
    f.write('train_time:')
    f.write(str(Train_Time))
    f.close()




