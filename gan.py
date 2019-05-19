import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import datasets, transforms
from torchvision.utils import save_image

import argparse

imagesize = 28
lr=0.001


class Dloss(nn.Module):
    def __init__(self):
        super(Dloss, self).__init__()

    def forward(self, x, z):
        if x.shape != z.shape:
            raise ArithmeticError('Sizes input to DLoss are not same.')
        #if torch.max(x) >= 1 or torch.max(z) >= 1: raise ArithmeticError('Is it true D returns probability?')
        x = x.view(-1, 1)
        z = z.view(-1, 1)
        return torch.sum(torch.log(x) + torch.log(1 - z)) / len(x)


class GLoss(nn.Module):
    def __init__(self):
        super(GLoss, self).__init__()

    def forward(self, x):
        #if torch.max(z) >= 1: raise ArithmeticError('Is it true D returns probability?')
        x = x.view(-1, 1)
        return torch.sum(torch.log(x)) / len(x)*10


# network generator
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(imagesize ** 2, 32 ** 2)
        self.fc2 = nn.Linear(32 ** 2, 32 ** 2)
        self.fc3 = nn.Linear(32 ** 2, imagesize ** 2)

    def forward(self, x):
        x = x.view(-1, imagesize ** 2)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = x.view(-1, imagesize, imagesize)
        return x


# network of discriminator
# TODO USE MAXOUT AND DROPOUT
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.fc4 = nn.Linear(imagesize ** 2, 16 ** 2)
        self.fc5 = nn.Linear(16 ** 2, 8 ** 2)
        self.fc6 = nn.Linear(8 ** 2, 1)

    def forward(self, x):
        x = x.view(-1, imagesize ** 2)
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.sigmoid(self.fc6(x))
        return x


def main():
    parse = argparse.ArgumentParser('My gan trainnig')
    parse.add_argument('--batchsize', type=int, default=128)
    parse.add_argument('--epoch', type=int, default=100)
    parse.add_argument('--nocuda', action='store_false', default=True)
    parse.add_argument('--k', type=int, default=3)
    args = parse.parse_args()
    device = 'cuda' if torch.cuda.is_available() and args.nocuda else 'cpu'
    # load data
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor()
                       ])),
        batch_size=args.batchsize, shuffle=True)

    # testloader=torch.utils.data.DataLoader(
    #    datasets.MNIST('../data',train=False,transform=transforms.Compose([
    #        transforms.ToTensor()
    #    ])),
    #    batch_size=args.batchsize,shuffle=True
    # )
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    criterion_D = Dloss()
    criterion_G = GLoss()
    # train
    # for epoch
    count = 0
    for e in range(args.epoch):
        # sample train data x
        for i,(x, label) in enumerate(train_loader,0):
            # for k
            # sample noise minibatch z
            z = torch.rand(x.shape).to(device)*255
            # update discriminator by sgd
            loss_D = criterion_D(discriminator.forward(x.to(device)), discriminator.forward(generator.forward(z)))
            loss_D.backward()
            discriminator.zero_grad()
            for param in discriminator.parameters():
                param.data+=lr*param.grad.data
            count += 1
            if count <= args.k:
                continue
            count = 0
            # TODO print loss mean

            # sample noise minibatch z by sgd
            z = torch.rand(args.batchsize, imagesize, imagesize).to(device)*255
            # update generator
            loss_G = criterion_G(discriminator.forward(generator.forward(z)))
            loss_G.backward()
            generator.zero_grad()
            for param in generator.parameters():
                param.data+=lr*param.grad.data
            print("e:{} {:2.1f}% loss_D:{} loss_G:{}".format(e,i*args.batchsize*100/len(train_loader.dataset),loss_D,loss_G))

            #test
            print(
                "t:",torch.mean(discriminator.forward(x)),
                "f:",torch.mean((torch.rand(x.shape).to(device))),
                "tf",torch.mean(generator(torch.rand(x.shape).to(device)*255))
            )
        if not os.path.exists('output'):
            os.mkdir('output')
        save_image((generator(torch.rand(1,imagesize,imagesize).to(device)*255)),'output/'+str(e)+'.png')


if __name__ == '__main__':
    main()
