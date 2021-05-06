import torch
import torch.nn as nn
import math

class GDA:
    def __init__(self, feats , labels, classes):
        # feat [(Tensor 512),(Tensor 512), ...]
        # labels [1,3,5,2, ... ]
        self.classes = classes
        feats = torch.stack(feats, axis=0) # [DatasetSize, Z_dim]
        labels = torch.tensor(labels)

        self.mu    = list()
        self.sigma = list()
        self.d = feats[0].size(0)
        for i in range(classes):
            x_i = feats[labels==i, :]
            mu_i = x_i.mean(axis=0)
            sigma_i = torch.bmm(x_i.unsqueeze(-1), x_i.unsqueeze(-1).T.permute(-1,0,1)).mean(dim=0)
            self.mu.append(mu_i)
            #self.sigma.append(sigma_i)
            self.sigma.append(sigma_i + torch.eye(self.d))

        #print(self.mu[0].shape, self.sigma[0].shape)

    def logp(self, x, c):
        # x     [B x d]
        # mu    [C x d]
        # sigma [C x d x d]
        bs = x.size(0)
        mu_unsq    = self.mu[c].unsqueeze(0)
        sigma_unsq = self.sigma[c].unsqueeze(0)
        mu_exp = mu_unsq.expand_as(x) #.cuda()
        sigma_exp = sigma_unsq.repeat(bs,1,1) #.cuda()
        #x = x.cuda()

        logsigma = torch.slogdet(sigma_exp)[-1]
        diffcov = torch.bmm( (x.unsqueeze(-1)-mu_exp.unsqueeze(-1)).permute(0,2,1) ,torch.inverse(sigma_exp))
        expterm = torch.bmm(diffcov , (x.unsqueeze(-1)-mu_exp.unsqueeze(-1))).squeeze()
        logprob = -logsigma/2 - self.d/2*torch.log(torch.tensor(2*math.pi)) -(expterm)/2

        return logprob.cpu()


    def __call__(self,x):

        logp = list()
        for i in range(self.classes):
            logp.append(self.logp(x,i))
        logp = torch.stack(logp, dim=0)

        return torch.logsumexp(logp,dim=0)

if __name__=="__main__":
    import matplotlib.pyplot as plt

    ds_size = 1024
    dim = 512
    classes = 10

    feats = [torch.randn(dim) for i in range(ds_size)]
    labels = torch.randint(0,classes,(ds_size,))

    gda = GDA(feats, labels, classes)

    print(gda(torch.randn(4,dim)))
