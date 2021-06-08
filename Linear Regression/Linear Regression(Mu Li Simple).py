import torch
import numpy as np
from torch.utils import data
from d2l import torch as d2l

true_w=torch.tensor([2,-3.4])
true_b=4.2
features,labels=d2l.synthetic_data(true_w,true_b,1000)#构造真实的数据

def load_array(data_arrays,batch_size,is_train=True):
    '''构造一个Pytorch的迭代器'''
    dataset=data.TensorDataset(*data_arrays)#*表示解包操作
    return data.DataLoader(dataset,batch_size,shuffle=is_train)

batch_size=10
data_iter=load_array((features,labels),batch_size)

print(next(iter(data_iter)))

from torch import nn

#Sequential就是一个list of layer
net=nn.Sequential(
    nn.Linear((2,1))#第一个指定输入特征形状，即 2，第二个指定输出特征形状，输出特征形状为单个标量，因此为 1。
)

net[0].weight.data.normal_(0,0.01)#网络的第一层net[0]
net[0].bias.data.fiil_(0)

loss=nn.MSELoss()
trainer = torch.optim.SGD(net.parameters(), lr=0.03)

num_epochs = 3
for epoch in range(num_epochs):
    for X, y in data_iter:
        l = loss(net(X), y)
        trainer.zero_grad()#梯度清0
        l.backward()#反向传播
        trainer.step()#更新参数
    l = loss(net(features), labels)
    print(f'epoch {epoch + 1}, loss {l:f}')

w = net[0].weight.data
print('w的估计误差：', true_w - w.reshape(true_w.shape))
b = net[0].bias.data
print('b的估计误差：', true_b - b)

