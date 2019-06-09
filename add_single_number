N=100
batchsize=64
epochs=1000
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split
import numpy as np
from itertools import product
from matplotlib import pyplot as plt
def printtrain(epoch,epochs,datasize,d,loss):
    width=50
    print('\repoch:[%d/%d] loss:%4.4f'%(epoch,epochs-1,loss),end='')
def mse(x,y):
    x=x.detach().numpy()
    y=y.detach().numpy()
    return np.sum((x-y)**2)/x.size
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1=nn.Linear(2,8)
        self.fc2=nn.Linear(8,4)
        self.fc3=nn.Linear(4,1)


    def forward(self, x):
        x=(self.fc1(x))
        x=(self.fc2(x))
        x=(self.fc3(x))
        return x
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

net=Net()


criterion=nn.MSELoss()
optimizer=optim.SGD(net.parameters(),lr=0.000001)

#make data
data=np.array([[i,j,i*j] for i,j in product(range(N),range(N))])
x_train, x_test, y_train, y_test = train_test_split(data[:,0:2],data[:,2], test_size=0.33)
x_train=torch.Tensor(x_train)
y_train=torch.FloatTensor(y_train)
x_test=torch.FloatTensor(x_test)
y_test=torch.FloatTensor(y_test)
if torch.cuda.is_available():
    x_train=x_train.cuda()
    y_train=y_train.cuda()
    x_test=x_test.cuda()
    y_test=y_test.cuda()

lossl=[]
for epoch in range(epochs):
    for i in range(0,len(x_train),batchsize):
        optimizer.zero_grad()
        output=net(x_train[i:i+batchsize,:])
        loss=criterion(output[:,0],y_train[i:i+batchsize])
        lossl.append(loss.item())
        loss.backward()
        optimizer.step()
        printtrain(epoch,epochs,len(x_train),i,loss)
    print('')

plt.plot(range(len(lossl)),lossl)
plt.show()
print(net.fc3.weight)
