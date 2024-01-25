import torch
import torch.nn as nn
import matplotlib as plt
import numpy as np
torch.manual_seed(10)

# 1. 生成数据
sample_nums = 100
mean_values = 1.7
bias = 1
n_data = torch.ones(sample_nums,2)
x0 = torch.normal(mean_values * n_data,1)+bias
y0 = torch.zeros(sample_nums)

x1 = torch.normal(-mean_values * n_data, 1)+bias
y1 = torch.ones(sample_nums)
train_x = torch.cat((x0,x1),0)
train_y = torch.cat((y0,y1),0)



# 2. 选择模型
class LR(nn.Module):
    def __init__(self):
        super(LR,self).__init__()
        self.features = nn.Linear(2,1)
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        x = self.features(x)
        x = self.sigmoid(x)
        return
lr_net = LR()


# 3. 选择损失函数
loss_fn = nn.BCELoss()

# 4. 选择优化器
lr = 0.01
optimizer = torch.optim.SGD(lr_net.parameters(),lr=lr, momentum=0.9)

# 5. 模型训练
for iter in range(1000):
    y_pred = lr_net(train_x) #前向传播
    loss = loss_fn(y_pred.squeeze(), train_y) #计算loss
    loss.backward() #反向传播
    optimizer.step() #更新参数


