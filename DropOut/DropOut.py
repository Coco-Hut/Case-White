import matplotlib.pyplot as plt
import torch
from torch import nn
from d2l import torch as d2l

#Hyperparameter:
num_inputs, num_outputs, num_hiddens1, num_hiddens2 = 784, 10, 256, 256
num_epochs,lr,batch_size=10,0.5,256
loss=nn.CrossEntropyLoss()
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)


def dropout_layer(X,dropout):
    assert 0<=dropout<=1
    if(dropout==0):
        return X
    if(dropout==1):
        return torch.zeros_like(X)
    mask=(torch.rand(X.shape)>dropout).float()
    return mask*X/(1.0-dropout)

dropout1,dropout2=0.2,0.5

class Net(nn.Module):
    def __init__(self,num_inputs,num_outputs,num_hiddens1,num_hiddens2,
                 is_Training=True):
        super(Net,self).__init__()
        self.num_inputs=num_inputs
        self.traning=is_Training
        self.lin1=nn.Linear(num_inputs,num_hiddens1)#定义全连接层网络，输入层到第一个隐藏层
        self.lin2=nn.Linear(num_hiddens1,num_hiddens2)#第一个隐藏层到第二个隐藏层
        self.lin3=nn.Linear(num_hiddens2,num_outputs)#第二个隐藏层到输出层
        self.relu=nn.ReLU()#激活函数

    def forward(self,X):
        H1=self.relu(self.lin1(X.reshape((-1,num_inputs))))

        if(self.training==True):
            H1=dropout_layer(H1,dropout1)

        H2=self.relu(self.lin2(H1))

        if(self.traning==True):
            H2=dropout_layer(H2,dropout2)

        out=self.lin3(H2)
        return out

net=Net(num_inputs,num_outputs,num_hiddens1,num_hiddens2,True)
trainer=torch.optim.SGD(net.parameters(),lr=lr)
d2l.train_ch3(net,train_iter,test_iter,loss,num_epochs,trainer)
plt.show()
print(list(net.parameters()))




