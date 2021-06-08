import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

#Hyper-parameters超参数

input_size=1
output_size=1
num_epochs=60
learning_rate=0.001


#Toy dataset

#准备训练集和测试集：训练集和数据集一定要是二维数组
x_train = np.array([[3.3], [4.4], [5.5], [6.71], [6.93], [4.168],
                    [9.779], [6.182], [7.59], [2.167], [7.042],
                    [10.791], [5.313], [7.997], [3.1]], dtype=np.float32)

y_train = np.array([[1.7], [2.76], [2.09], [3.19], [1.694], [1.573],
                    [3.366], [2.596], [2.53], [1.221], [2.827],
                    [3.465], [1.65], [2.904], [1.3]], dtype=np.float32)


#device=torch.device("cuda:0"if torch.cuda.is_available()else"cpu")
#建立模型:Linear regression model
model=nn.Linear(input_size,output_size)
#model.to(device)

# Loss and optimizer 损失函数和优化器的设置
criterion=nn.MSELoss()#熵值为均方差

#优化算法采用随机梯度下降
optimizer=torch.optim.SGD(model.parameters(),lr=learning_rate)

#训练模型 Train the model
for epoch in range(num_epochs):
    #numpy数组转化为tensor
    inputs=torch.from_numpy(x_train)#实际上这里是二维数组
    targets=torch.from_numpy(y_train)

    #向前传播
    outputs=model(inputs)
    loss=criterion(outputs,targets)

    #Backward and optimize反向传播与优化
    optimizer.zero_grad()#将当前梯度置为0,避免梯度的重复计算
    loss.backward()#反向传播求梯度
    optimizer.step()#误差反向传播后更新参数的步骤，实际上根据梯度调整我们的参数

    if(epoch+1)%5==0:
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))

    #plot the graph 画图
    predicted=model(torch.from_numpy(x_train)).detach().numpy()
    plt.plot(x_train, y_train, 'ro', label='Original data')
    plt.plot(x_train, predicted, label='Fitted line')
    plt.legend()
    plt.show()


# 模型的保存