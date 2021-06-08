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

#参数自动初始化设置：Flatten是将三维的图片转化为2维度的向量
net=nn.Sequential(
    nn.Flatten(),
    nn.Linear(784,256),
    nn.ReLU(),
    nn.Linear(256,10)
)

def init_weight(m):
    if type(m)==nn.Linear:
        nn.init.normal_(m.weight,std=0.01)

net.apply(init_weight)

batch_size, lr, num_epochs = 256, 0.1, 10
loss = nn.CrossEntropyLoss()
trainer = torch.optim.SGD(net.parameters(), lr=lr)

d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)

plt.show()
