import numpy as np
import torch

class LorentzianDistance(torch.nn.Module):
    def __init__(self, radius=1, dim=None):
        super(LorentzianDistance, self).__init__()
        self.beta = 1

    def forward(self, u, v):
        u0 = torch.sqrt(torch.pow(u,2).sum(-1, keepdim=True) + self.beta)
        v0 = -torch.sqrt(torch.pow(v,2).sum(-1, keepdim=True) + self.beta)
        u = torch.cat((u,u0),-1)
        v = torch.cat((v,v0),-1)
        result = - 2 * self.beta - 2 *torch.sum(u * v, dim=-1)
        return result

def givens_rotations(r, x):
    givens = r.view((r.shape[0], -1, 2))
    givens = givens / torch.norm(givens, p=2, dim=-1, keepdim=True).clamp_min(1e-15)
    x = x.view((r.shape[0], -1, 2))
    x_rot = givens[:, :, 0:1] * x + givens[:, :, 1:] * torch.cat((-x[:, :, 1:], x[:, :, 0:1]), dim=-1)
    return x_rot.view((r.shape[0], -1))

class MuREL(torch.nn.Module):
    def __init__(self, d, dim):
        super(MuREL, self).__init__()
        #lorentz
        self.E = torch.nn.Embedding(len(d.entities), dim, padding_idx=0)
        self.E.weight.data = (1e-3 * torch.randn((len(d.entities), dim), dtype=torch.double, device="cuda"))
        self.Wu = torch.nn.Parameter(torch.tensor(np.random.uniform(-1, 1, (len(d.relations), 
                                        dim)), dtype=torch.double, requires_grad=True, device="cuda"))
        self.rv = torch.nn.Embedding(len(d.relations), dim, padding_idx=0)
        self.rv.weight.data = (1e-3 * torch.randn((len(d.relations), dim), dtype=torch.double, device="cuda"))
        self.bs = torch.nn.Parameter(torch.zeros(len(d.entities), dtype=torch.double, requires_grad=True, device="cuda"))
        self.bo = torch.nn.Parameter(torch.zeros(len(d.entities), dtype=torch.double, requires_grad=True, device="cuda"))
        self.loss = torch.nn.BCEWithLogitsLoss()
        #euclidean
        self.E1 = torch.nn.Embedding(len(d.entities), dim, padding_idx=0)
        self.E1.weight.data = (1e-3 * torch.randn((len(d.entities), dim), dtype=torch.double, device="cuda"))
        self.Wu1 = torch.nn.Parameter(torch.tensor(np.random.uniform(-1, 1, (len(d.relations), 
                                        dim)), dtype=torch.double, requires_grad=True, device="cuda"))
        self.rv1 = torch.nn.Embedding(len(d.relations), dim, padding_idx=0)
        self.rv1.weight.data = (1e-3 * torch.randn((len(d.relations), dim), dtype=torch.double, device="cuda"))
       
    def forward(self, u_idx, r_idx, v_idx, func=LorentzianDistance):
        #lorentz
        u = self.E.weight[u_idx]
        v = self.E.weight[v_idx]
        Ru = self.Wu[r_idx]
        rv = self.rv.weight[r_idx]
        u_W = u * Ru
        #euclidean
        u1 = self.E1.weight[u_idx]
        v1 = self.E1.weight[v_idx]
        Ru1 = self.Wu[r_idx]
        rv1 = self.rv1.weight[r_idx]
        u_W1 = u1 * Ru1
        sqdist_lorentz = func()(u_W, v + rv)
        sqdist_euclidean = torch.sum(torch.pow(u_W1 - (v1 + rv1), 2), dim=-1)
        sqdist = sqdist_lorentz + sqdist_euclidean
        return -sqdist + self.bs[u_idx] + self.bo[v_idx] 

class RotEL(torch.nn.Module):
    def __init__(self, d, dim):
        super(RotEL, self).__init__()
        #lorentz
        self.E = torch.nn.Embedding(len(d.entities), dim, padding_idx=0)
        self.E.weight.data = (1e-3 * torch.randn((len(d.entities), dim), dtype=torch.double, device="cuda"))
        self.Wu = torch.nn.Parameter(torch.tensor(np.random.uniform(-1, 1, (len(d.relations), 
                                        dim)), dtype=torch.double, requires_grad=True, device="cuda"))
        self.rv = torch.nn.Embedding(len(d.relations), dim, padding_idx=0)
        self.rv.weight.data = (1e-3 * torch.randn((len(d.relations), dim), dtype=torch.double, device="cuda"))
        self.bs = torch.nn.Parameter(torch.zeros(len(d.entities), dtype=torch.double, requires_grad=True, device="cuda"))
        self.bo = torch.nn.Parameter(torch.zeros(len(d.entities), dtype=torch.double, requires_grad=True, device="cuda"))
        self.loss = torch.nn.BCEWithLogitsLoss()
        #euclidean
        self.E1 = torch.nn.Embedding(len(d.entities), dim, padding_idx=0)
        self.E1.weight.data = (1e-3 * torch.randn((len(d.entities), dim), dtype=torch.double, device="cuda"))
        self.Wu1 = torch.nn.Parameter(torch.tensor(np.random.uniform(-1, 1, (len(d.relations), 
                                        dim)), dtype=torch.double, requires_grad=True, device="cuda"))
        self.rv1 = torch.nn.Embedding(len(d.relations), dim, padding_idx=0)
        self.rv1.weight.data = (1e-3 * torch.randn((len(d.relations), dim), dtype=torch.double, device="cuda"))
       
    def forward(self, u_idx, r_idx, v_idx, func=LorentzianDistance):
        #lorentz
        u = self.E.weight[u_idx]
        v = self.E.weight[v_idx]
        Ru = self.Wu[r_idx]
        rv = self.rv.weight[r_idx]
        u_W = givens_rotations(Ru, u).reshape(u.shape)
        #euclidean
        u1 = self.E1.weight[u_idx]
        v1 = self.E1.weight[v_idx]
        Ru1 = self.Wu[r_idx]
        rv1 = self.rv1.weight[r_idx]
        u_W1 = givens_rotations(Ru1, u1).reshape(u1.shape)
        sqdist_lorentz = func()(u_W, v + rv)
        sqdist_euclidean = torch.sum(torch.pow(u_W1 - (v1 + rv1), 2), dim=-1)
        sqdist = sqdist_lorentz + sqdist_euclidean
        return -sqdist + self.bs[u_idx] + self.bo[v_idx] 