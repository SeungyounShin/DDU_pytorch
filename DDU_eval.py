import torch
import torch.nn.functional as F
from GDA import GDA
import numpy as np
import matplotlib.pyplot as plt

def func_eval2(model,gmm,data_iter,device):
    #false_label=torch.tensor([10]).to(device)
    with torch.no_grad():
        n_total,n_correct,epis_unct_sum,alea_unct_sum = 0,0,0,0
        avg,new_avg,var=0,0,0
        y_probs= list()
        feats = list()
        labels = list()
        model.eval() # evaluate (affects DropOut and BN)
        for idx,(batch_in,batch_out) in enumerate(data_iter):
            # Foraward path
            bs = batch_in.size(0)

            feat = model.feature_extract(batch_in.to(device))
            feat = F.avg_pool2d(feat, 8, 1, 0)
            feat = feat.view(bs, -1)
            out = model.fc(feat)

            # Check predictions
            y_prob,y_pred    = torch.max(out,1)
            n_correct   += (y_pred==batch_out.to(device)).sum().item()
            #print(y_trgt)
            n_total     += batch_in.size(0)

            y_probs += list(y_prob.cpu().numpy()) # [N]

            feats  += [i for i in feat.detach().cpu().data] # [N x D]
            labels += list(batch_out.cpu().numpy()) # [N]

        val_accr  = (n_correct/n_total)
        feats = torch.stack(feats, axis=0).unsqueeze(1)
        epis = gmm.q(feats).numpy() # [N]

        model.train() # back to train mode
        out_eval = {'val_accr':val_accr,'epis':np.mean(epis),'alea':np.mean(y_probs), 'y_prob' : y_probs,'var': np.var(epis)}
    return out_eval

def func_eval_ood2(model,gmm,data_iter,device):
    indist = [0,1,2,3,4,5,6,7,8,9]
    outdist = [10]
    with torch.no_grad():
        n_total,n_correct,epis_unct_sum,alea_unct_sum = 0,0,0,0
        avg,new_avg,var=0,0,0
        epis_, alea_, y_trues =list(), list(),list()
        y_probs= list()
        feats = list()
        model.eval() # evaluate (affects DropOut and BN)
        for batch_in,batch_out in data_iter:
            # Foraward path
            bs = batch_in.size(0)

            feat = model.feature_extract(batch_in.to(device))
            feat = F.avg_pool2d(feat, 8, 1, 0)
            feat = feat.view(bs, -1)
            out = model.fc(feat)

            feats = feat.detach().cpu().unsqueeze(1)
            #print(pi)
            y_prob,y_pred = torch.max(out,1)

            # Compute uncertainty
            epis_unct   = gmm.q(feats) # [N]
            alea_unct   = y_prob # [N]
            epis_unct_sum += torch.sum(epis_unct)
            alea_unct_sum += torch.sum(alea_unct)

            # Check predictions

            indist_  = (0==batch_out)| (1==batch_out) | (2==batch_out) | (3==batch_out) | (4==batch_out) | (5==batch_out)| (6==batch_out) | (7==batch_out) | (8==batch_out) | (9==batch_out)
            outdist_ = (10==batch_out)
            indist_  = indist_.long()
            outdist_  = outdist_.long()
            n_correct   += (y_pred==batch_out.to(device)).sum().item()
            n_total     += batch_in.size(0)
            y_probs += list(1-y_prob.cpu().numpy())
            epis_ += list(epis_unct.cpu().numpy())
            alea_ += list(alea_unct.cpu().numpy())
            y_trues += list(outdist_.cpu().numpy())


        val_accr  = (n_correct/n_total)
        epis      = (epis_unct_sum/n_total).detach().cpu().item()
        alea      = (alea_unct_sum/n_total).detach().cpu().item()
        model.train() # back to train mode
        out_eval = {'val_accr':val_accr,'epis':epis,'alea':alea, 'epis_' : epis_,'alea_' : alea_, 'y_true':y_trues, 'n_correct':n_correct,'y_probs' : y_probs}
    return out_eval
