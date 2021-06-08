import matplotlib.pyplot as plt
import torch
from torch import nn
from d2l import torch as d2l
import torchvision
from torch.utils import data
from torchvision import transforms

def get_dataloader_workers(process):
    """使用process个进程来读取数据。"""
    return process

batch_size=256
trans = transforms.ToTensor()
mnist_train = torchvision.datasets.FashionMNIST(root="../data", train=True,
                                                transform=trans,
                                                download=False)
mnist_test = torchvision.datasets.FashionMNIST(root="../data", train=False,
                                               transform=trans, download=False)

#注意DataLoader返回的是一个迭代器，即读完一个batch后下一次读的是下一个batch
train_iter=data.DataLoader(mnist_train, batch_size, shuffle=True,
                            num_workers=get_dataloader_workers(4))

test_iter=data.DataLoader(mnist_test, batch_size, shuffle=False,
                            num_workers=get_dataloader_workers(4))

num_inputs,num_outputs,num_hiddens=784,10,256

#初始化参数
W1=nn.Parameter(
    torch.randn(num_inputs,num_hiddens,requires_grad=True)*0.01)

b1=nn.Parameter(torch.zeros(num_hiddens,requires_grad=True))

W2 = nn.Parameter(
    torch.randn(num_hiddens, num_outputs, requires_grad=True) * 0.01)

b2=nn.Parameter(torch.zeros(num_outputs,requires_grad=True))

params=[W1,b1,W2,b2]

def relu(X):
    a=torch.zeros_like(X)#zeros_like(X)表示和X的形状一样但是全部是0的矩阵
    return torch.max(X,a)

def net(X):
    X=X.reshape((-1,num_inputs))#自动计算实际上是256行，784列
    H=relu(X @ W1+b1)
    return(H @ W2+b2)

loss = nn.CrossEntropyLoss()#交叉熵
num_epochs, lr = 10, 0.1
updater = torch.optim.SGD(params, lr=lr)
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, updater)


d2l.predict_ch3(net, test_iter)

plt.show()