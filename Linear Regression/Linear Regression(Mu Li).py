import matplotlib.pyplot as plt
import random
import torch
from d2l import torch as d2l

def synsetic_data(w,b,num_examples):
    X=torch.normal(0,1,(num_examples,len(w)))#（mean,std,out)第三个参数指的是输出张量，可以以元组的形式表示，第一个是个数，第二个是长度
    Y=torch.matmul(X,w)+b#构造回归值
    Y+=torch.normal(0,0.01,Y.shape)#添加噪声项
    return X,Y.reshape((-1,1))#注意举证此时是一个行向量，要转化为列向量

true_w=torch.tensor([2,-3.4])#人工设置的真实权重
true_b=4.2

features,labels=synsetic_data(true_w,true_b,1000)#产生1000个人工数据
print('features:', features[0], '\nlabel:', labels[0])

d2l.set_figsize()
d2l.plt.scatter(features[:, (1)].detach().numpy(),
                labels.detach().numpy(), 1);#涉及到梯度传播的tensor不能直接转化为numpy，必须先detach然后才能转为numpy

#batch生成器
def data_iter(batch_size,features,labels):
    num_examples=len(features)
    indices=list(range(num_examples))
    #这些样本是随机读取的没有顺序
    random.shuffle(indices)
    for i in range(0,num_examples,batch_size):
        batch_indices=torch.tensor(indices[i:min(i+batch_size,num_examples)])#放置超过数量的上限。
        yield features[batch_indices],labels[batch_indices]#yield相当生成return，但是可以承接上一步,即一调用完后，下一次紧接着上一次结束的语句继续执行后

batch_size=10

#生成一个batch
for X,y in data_iter(batch_size,features,labels):
    print(X,'\n',y)
    break

w=torch.normal(0,0.01,size=(2,1),requires_grad=True)
b=torch.zeros(1,requires_grad=True)

#定义线性的模型
def linreg(X,w,b):
    return torch.matmul(X,w)+b

#定义损失函数
def squared_loss(y_hat,y):
    return (y_hat-y.reshape(y_hat.shape))**2/2

#实现小批量的梯度更新
def SGD(params,lr,batch_size):
    #第一句表示的是更新的时候不要计算梯度
    with torch.no_grad():
        for param in params:
            param-=lr*param.grad/batch_size
            param.grad.zero_()#梯度清零

lr=0.03#学习率
num_epoch=3#扫描个数
net=linreg
loss=squared_loss

for epoch in range(num_epoch):
    for X,y in data_iter(batch_size,features,labels):
        l=loss(net(X,w,b),y)#一个mini-batch的小批量损失
        l.sum().backward()
        SGD([w,b],lr,batch_size)#使用参数的梯度更新参数
    #不需要计算机做的地方用no_grad
    with torch.no_grad():
        train_=loss(net(features,w,b),labels)
        '''
        python的print字符串前面加f表示格式化字符串，加f后可以在字符串里面使用用花括号括起来的变量和表达式，
        如果字符串里面没有表达式，那么前面加不加f输出应该都一样.
        '''
        print(f'epoch {epoch + 1}, loss {float(train_.mean()):f}')

print(f'真实的权重w为: {w}')
print(f'真实的偏差b为：{b}')





