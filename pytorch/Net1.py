import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import torch
from torch.autograd import Variable
class Net(nn.Module):
    def __init__(self,in_dim,n_hidden1,n_hidden2,out_dim):
        super(Net, self).__init__()
        self.layer1 = nn.Linear.Sequential(in_dim,n_hidden1)
        self.layer2 = nn.Linear(n_hidden1,n_hidden2)
        self.layer3 = nn.Linear(n_hidden2,out_dim)
    def forward(self,x):
        x1 = F.relu(self.layer1(x))
        x1 = F.relu(self.layer2(x1))
        x2 = self.layer3(x1)
        print('\t In Model: input size',x.size(),"output size",x2.size())
        return x2
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net(13,16,32,1).to(device)