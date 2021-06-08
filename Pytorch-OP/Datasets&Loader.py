import torch
from torch.utils.data import Dataset,DataLoader

data_path=r'./SMSSpamCollection'

#Dataset是一个基类，可以查看其源代码发现__getitem__和__len__都没有实现,自己实现返回的标签设数据,自己定制数据集
class SMSSDsets(Dataset):
    def __init__(self):
        self.data=open(data_path,encoding='utf-8').readlines()

    def __getitem__(self, index):
        self.data[index]=self.data[index].rstrip()#去除空格
        self.label=self.data[index][:4].rstrip()
        self.context=self.data[index][4:].rstrip()
        return self.label,self.context

    def __len__(self):
        return len(self.data)

if __name__=='__main__':
    datasets=SMSSDsets()
    data_iter=DataLoader(dataset=datasets,batch_size=10,shuffle=True,num_workers=2,drop_last=True)#将最后的不足batch_size的舍去
    for index,(labels,contexts) in enumerate(data_iter):
        print('Batch_num:{}'.format(index),labels,contexts)

'''
模型保存（博文）：
https://zhuanlan.zhihu.com/p/38056115
'''