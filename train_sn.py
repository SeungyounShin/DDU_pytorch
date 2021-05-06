import math
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as TD
from torch.autograd import Variable
from collections import OrderedDict
from copy import deepcopy

from backbone.resnet import *
from backbone.sn_wide_resnet import *
from DDU_eval import func_eval_ood2,func_eval2
from GMM import GaussianMixture
from torchvision import datasets,transforms
#from summary import summary_string
import time

from sklearn.metrics import precision_recall_curve,roc_auc_score
import seaborn as sns
import argparse
from utils import plot_res,test_result,print_n_txt

parser = argparse.ArgumentParser(description='MLN_CIFAR_OOD')
parser.add_argument('--rs', type=float,default=0.0,help='shuffle ratio')
parser.add_argument('--rp', type=float,default=0.0,help='permuation ratio')
parser.add_argument('--ratio', type=float,default=0.1,help='hum..')
parser.add_argument('--wd', type=float,default=5e-4,help='weight decay')
parser.add_argument('--id', type=int,default=1,help='figure save index')
parser.add_argument('--opt', type=int,default=1,help='1 for sgd 2 for adw 3 RMSprop')
parser.add_argument('--lr', type=float,default=1e-1,help='learing rate')
parser.add_argument('--loss', type=int,default=1,help='2 for kendal loss 3 for candidate 1 4 for candidate 2')
parser.add_argument('--k', type=int,default=10,help='number of mixtures')
parser.add_argument('--gpu', type=int,default=0,help='gpu device')
parser.add_argument('--epoch', type=int,default=170,help='epoch')
parser.add_argument('--spectral_norm', type=bool,default=True,help='spectral_norm')
parser.add_argument('--model', type=str,default='wideresnet28x10',help='model')
parser.add_argument('--init_sigma', type=float,default=1,help='wi')
parser.add_argument('--init_pi', type=float,default=0.001,help='wi')
args = parser.parse_args()

torch.cuda.set_device(args.gpu)
device='cuda'
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

CIFAR10_train = datasets.CIFAR10(root='./data/',train=True,transform=transform_train,download=True)
#CIFAR10_train=get_train_iter2(rs_rate=args.rs,rp_rate=args.rp)

CIFAR10_test = datasets.CIFAR10(root='./data/',train=False,transform=transform_test,download=True)
CIFAR10_val, CIFAR10_test = torch.utils.data.random_split(CIFAR10_test,[int(len(CIFAR10_test)*0.5), int(len(CIFAR10_test)*0.5)])

shvn_test = datasets.SVHN(root="./data/",split="test", download=True,transform=transform_test)
false_label=torch.tensor([10])
shvn_test.labels=  deepcopy(false_label.repeat_interleave(len(shvn_test)))# manipulate train labels
shvn_val, shvn_test = torch.utils.data.random_split(shvn_test,[int(len(shvn_test)*0.5), int(len(shvn_test)*0.5)])
indices = torch.arange(5000)
shvn_test = torch.utils.data.Subset(shvn_test, indices)
shvn_val = torch.utils.data.Subset(shvn_val, indices)
#print(shvn_test.targets)
BATCH_SIZE = 128
train_iter = torch.utils.data.DataLoader(CIFAR10_train,batch_size=BATCH_SIZE,shuffle=True,num_workers=1)
val_iter_1 = torch.utils.data.DataLoader(CIFAR10_val,batch_size=BATCH_SIZE,shuffle=True,num_workers=1)
val_iter_2 = torch.utils.data.DataLoader(shvn_val,batch_size=BATCH_SIZE,shuffle=True,num_workers=1)

datasets = [CIFAR10_test,shvn_test]
datasets = torch.utils.data.ConcatDataset(datasets)
test_dataset = torch.utils.data.DataLoader(datasets,batch_size=BATCH_SIZE,shuffle=True,num_workers=1)

print("train in distribution: {}".format(len(CIFAR10_train)))
print('val in distriubtion :{} , val out distribution: {}'.format(len(CIFAR10_val), len(shvn_val)))
print('test in distriubtion :{} , test out distribution: {}'.format(len(CIFAR10_test), len(shvn_test)))

if args.model =='resnet18':
    print(args.model)
    model = resnet18(pretrained = True)
elif args.model =='wideresnet28x10':
    if not args.spectral_norm:
        model =  WideResNet(28, 10, 10, 0.3).to(device)
    else:
        print(" + using spectral normalization !")
        model =  SNWideResNet(28, 10, 10, 0.1).to(device)
gmm = GaussianMixture(n_components=args.k, n_features=640, mu_init=None, var_init=None, eps=1.e-6)

if args.opt==1:
    print('sgd')
    optm = optim.SGD(model.parameters(), lr=args.lr,momentum=0.9, weight_decay=args.wd, nesterov=True)#lr=1e-2
elif args.opt==2:
    optm = optim.AdamW(model.parameters(),lr=args.lr,weight_decay=args.wd)# lr 1e-4 wd 5e-5
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optm, milestones=[50,100,140], gamma=0.2)
else:
    optm = optim.RMSprop(model.parameters(),lr=0.001,weight_decay=1e-4)
#scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optm, T_max=300)
# scheduler = torch.optim.lr_scheduler.MultiStepLR(optm, milestones=[100,200,300], gamma=0.6)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optm, milestones=[50,100,140], gamma=0.2)
criterion = torch.nn.CrossEntropyLoss()
model.train() # train mode
#print (summary_str)
#MLN.load_state_dict('./ckpt/CIFAR_10_MLN_40.pt')
loss_name=None

best_valid_acc = 0.
EPOCHS,test_every,val_every = args.epoch,1,1
lacc,lauroc1,lauroc2,laupr1,laupr2=[],[],[],[],[]
true_len=len(shvn_test)
txtName = ('./res/res_%s.txt'%(args.id))
f = open(txtName,'w') # Open txt file
print_n_txt(_f=f,_chars='Text name: '+txtName)
print_n_txt(_f=f,_chars=str(args))
for epoch in range(EPOCHS):
    loss_sum = 0.0
    #time.sleep(1)
    full_batch_len = 0
    right = 0
    feats = list()

    for batch_in,batch_out in train_iter:
        bs = batch_in.size(0)

        feat = model.feature_extract(batch_in.to(device))
        feat = F.avg_pool2d(feat, 8, 1, 0)
        feat = feat.view(bs, -1)
        out = model.fc(feat)

        loss = criterion(out, batch_out.to(device))

        # Update
        optm.zero_grad() # reset gradient
        loss.backward() # back-propagation
        optm.step() # optimizer update

        # Track losses
        loss_sum += loss
        # feature(z) save
        feats += [i for i in feat.detach().cpu().data]

    scheduler.step()

    loss_avg = loss_sum/len(train_iter)
    feats = torch.stack(feats, axis=0).unsqueeze(1)
    print("==== fitting gmm ====")
    gmm.fit(feats, n_iter=1000)

    if ((epoch%val_every)==0) or (epoch==(EPOCHS-1)):
        train_res = func_eval2(model,gmm,train_iter,device)
        val_res_1  = func_eval2(model,gmm,val_iter_1,device)
        val_res_2  = func_eval2(model,gmm,val_iter_2,device)
        if(best_valid_acc < val_res_1['val_accr']):
            best_valid_acc = val_res_1['val_accr']
            torch.save(model.state_dict(), './ckpt/sn_wide_resnet.pth')
        strTemp =  ("epoch:[%d/%d] loss:[%.3f] train_accr:[%.3f] Val_ID_accr:[%.3f] Val_OD_accr:[%.3f]."%
                (epoch,EPOCHS,loss_avg,train_res['val_accr'],val_res_1['val_accr'],val_res_2['val_accr']))
        strTemp2 = (" [Train] alea:[%.3f] epis:[%.3f] epis_var: [%.5f]\n [Val_ID] alea:[%.3f] epis:[%.5f] epis_var: [%.1f] \n [Val_OD] alea:[%.3f] epis:[%.5f] epis_var: [%.1f]"%
                    (train_res['alea'],train_res['epis'],train_res['var'],val_res_1['alea'],val_res_1['epis'],val_res_1['var'],\
                    val_res_2['alea'],val_res_2['epis'],val_res_2['var']))
        print_n_txt(_f=f,_chars=strTemp)
        print_n_txt(_f=f,_chars=strTemp2)
    if ((epoch%test_every)==0) or (epoch==(EPOCHS-1)):
        test_res  = func_eval_ood2(model,gmm,test_dataset,device)
        #print("[Test] epis: [%.5f] alea: [%.5f]"%(test_res['epis'],test_res['alea']))
        acc,auroc1,auroc2,auroc3,aupr1,aupr2,aupr3=test_result(test_res,true_len)
        lacc.append(acc)
        laupr1.append(aupr1)
        laupr2.append(aupr2)
        lauroc1.append(auroc1)
        lauroc2.append(auroc2)
        strTemp = ('test done: acc: [%.3f], epis auroc: [%.3f], epis aupr: [%.3f], alea auroc: [%.3f], alea aupr: [%.3f], max auroc: [%.3f], max aupr: [%.3f]'%
                    (acc,auroc2,aupr2,auroc3,aupr3,auroc1,aupr1))
        print_n_txt(_f=f,_chars=strTemp)

        train_log=[lacc,lauroc1,laupr1,lauroc2,laupr2]
        plot_res(test_res,true_len,loss_name,args.id,args.rs,args.rp,train_log)
