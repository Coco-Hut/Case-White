import matplotlib.pyplot as plt
import torch
from torch import nn
from d2l import torch as d2l

#只使用一个含有20个样本的小数据集和一个200维的模型，这样容易过拟合，所有的weight都是0.01
n_train, n_test, num_inputs, batch_size = 20, 100, 200, 5
true_w, true_b = torch.ones((num_inputs, 1)) * 0.01, 0.05
train_data = d2l.synthetic_data(true_w, true_b, n_train)#返回一个元组
#train_data返回的是一个元组，元组的第一个变量是一个tensor矩阵，是n_train*len(true_w)的一个矩阵，每一行表示的就是一组数据即各个维度的特征
train_iter = d2l.load_array(train_data, batch_size)

#train_iter返回的是一个迭代器
test_data = d2l.synthetic_data(true_w, true_b, n_test)
test_iter = d2l.load_array(test_data, batch_size, is_train=False)

def init_params():
    w=torch.normal(0,1,size=(num_inputs,1),requires_grad=True)
    b=torch.zeros(1,requires_grad=True)
    return [w,b]

def L1_penalty(w):
    return torch.sum(abs(w))

#定义L2的范数惩罚项
def L2_penalty(w):
    return torch.sum(w.pow(2))/2

def train(lambd):
    w,b=init_params()
    net,loss=lambda X:d2l.linreg(X,w,b),d2l.squared_loss
    num_epochs,lr=100,0.003
    animator = d2l.Animator(xlabel='epochs', ylabel='loss', yscale='log',
                            xlim=[5, num_epochs], legend=['train', 'test'])
    for epoch in range(num_epochs):
        #一个一个batch取出来
        for X,y in train_iter:
            l=loss(net(X),y)+lambd*L2_penalty(w)
            l.sum().backward()#计算梯度
            d2l.sgd([w,b],lr,batch_size)
        if(epoch+1)%5==0:
            #动态图示：每5个epoch就更新一次图像。
            animator = d2l.Animator(xlabel='epochs', ylabel='loss', yscale='log',
                                    xlim=[5, num_epochs], legend=['train', 'test'])

    print("w的L2范数是：",torch.norm(w).item())


#第一步：忽略正则化直接训练
train(lambd=0)

#第二部：正则项设置为3
train(lambd=3)


#简洁实现：使用pytorch自身的API实现
def train_concise(wd):
    net=nn.Sequential(nn.Linear(num_inputs,1))
    for para in net.parameters():
        para.data.normal_()#参数初始化
    loss=nn.MSELoss()
    num_epochs,lr=100,0.003
    #偏执参数没有衰减
    trainer=torch.optim.SGD([{
        "params":net[0].weight,
        'weight_decay':wd},{
            "params":net[0].bias}],lr=lr)
    animator = d2l.Animator(xlabel='epochs', ylabel='loss', yscale='log',
                            xlim=[5, num_epochs], legend=['train', 'test'])
    for epoch in range(num_epochs):
        for X,y in train_iter:
            #标准四件套
            trainer.zero_grad()
            l=loss(net(X),y)
            l.backward()
            trainer.step()
    if (epoch + 1) % 5 == 0:
        animator.add(epoch + 1, (d2l.evaluate_loss(net, train_iter, loss),
                                 d2l.evaluate_loss(net, test_iter, loss)))
    print('w的L2范数：', net[0].weight.norm().item())

