from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
from layer import *

class graph_constructor(nn.Module):
    def __init__(self, nnodes, k, dim, device, alpha=3, static_feat=None):
        super(graph_constructor, self).__init__()
        self.nnodes = nnodes
        self.k = k
        self.dim = dim
        self.device = device
        self.alpha = alpha
        self.static_feat = static_feat
        self.adj_cache = {}  # Cache for adj matrices
        if static_feat is not None:
            xd = static_feat.shape[1]
            self.lin1 = nn.Linear(xd, dim, bias=False)
            self.lin2 = nn.Linear(xd, dim, bias=False)
        else:
            self.lin1 = nn.Linear(nnodes, dim, bias=False)
            self.lin2 = nn.Linear(nnodes, dim, bias=False)

        # Initialize and fix linear layers
        nn.init.xavier_uniform_(self.lin1.weight)
        nn.init.xavier_uniform_(self.lin2.weight)
        self.lin1.weight.requires_grad = False
        self.lin2.weight.requires_grad = False

        # Fixed r (non-trainable)
        self.r = torch.ones(1, nnodes, device=device) / nnodes
        self.r.requires_grad = False

    def _stabilize_cov(self, cov, sub_nnodes):
        cov_mean = torch.mean(torch.abs(cov))
        cov_norm = torch.norm(cov, p='fro')
        reg = max(1e-1 * cov_mean, 1e-2 * cov_norm, 1e-3)
        cov = cov + torch.eye(cov.size(0), device=self.device) * reg
        cov = (cov + cov.transpose(0, 1)) / 2
        try:
            eigvals = torch.linalg.eigvalsh(cov)
            cond_number = eigvals.max() / (eigvals.min().abs() + 1e-10)
        except RuntimeError:
            print("eigvalsh failed in stabilize_cov, using regularized matrix")
        return cov

    def forward(self, idx):
        idx = idx.long()  # Ensure idx is of type int64
        sub_nnodes = idx.size(0)

        # Check cache
        idx_tuple = tuple(idx.cpu().numpy())
        if idx_tuple in self.adj_cache:
            return self.adj_cache[idx_tuple].to(self.device)

        # Generate node embeddings
        if self.static_feat is None:
            # One-hot encoding: [sub_nnodes, nnodes]
            nodevec = torch.zeros(sub_nnodes, self.nnodes, device=self.device)
            nodevec.scatter_(1, idx.unsqueeze(1), 1.0)  # Scatter operation with idx
            nodevec1 = nodevec
            nodevec2 = nodevec
        else:
            nodevec1 = self.static_feat[idx, :]  # [sub_nnodes, xd]
            nodevec2 = nodevec1

        # Transform to [sub_nnodes, dim]
        nodevec1 = torch.tanh(self.alpha * self.lin1(nodevec1))
        nodevec2 = torch.tanh(self.alpha * self.lin2(nodevec2))
        X_em = (nodevec1 + nodevec2) / 2  # [sub_nnodes, dim]


        # Normalize X_em
        X_em = (X_em - X_em.mean(dim=0, keepdim=True)) / (X_em.std(dim=0, keepdim=True) + 1e-6)

        # Dimensionality reduction
        if sub_nnodes > self.dim:
            try:
                U, S, Vh = torch.linalg.svd(X_em, full_matrices=False)
                S = S.clamp(min=1e-4)
                rank = min(self.dim, S.size(0))
                X_em = U[:, :rank] @ torch.diag(S[:rank])
            except RuntimeError:
                print("SVD failed, skipping dimensionality reduction")
        # Compute covariance: [sub_nnodes, sub_nnodes]
        cov = torch.matmul(X_em, X_em.transpose(0, 1))
        cov = self._stabilize_cov(cov, sub_nnodes)

        # Eigen decomposition
        try:
            eigenvalues, v = torch.linalg.eigh(cov)
            if torch.any(eigenvalues < -1e-6):
                raise RuntimeError
        except RuntimeError:
            try:
                u, s, vh = torch.linalg.svd(cov, full_matrices=True)
                s = s.clamp(min=1e-4)
                v = u
            except RuntimeError:
                v = torch.eye(sub_nnodes, device=self.device)

        M = (v.transpose(0, 1) @ v) ** 2  # [sub_nnodes, sub_nnodes]
        b = (X_em.transpose(0, 1) @ v) ** 2  # [dim, sub_nnodes]
        b = b.sum(dim=0)  # [sub_nnodes]

        r_sub = self.r[:, idx]  # [1, sub_nnodes]
        r = r_sub / (torch.matmul(r_sub, b.unsqueeze(1)) + 1e-6)

        for _ in range(10):
            r_old = r.clone()
            Mr = torch.matmul(M, r.transpose(0, 1))
            rMr = torch.matmul(r, Mr).squeeze()
            r_new = r * b * rMr / (Mr.transpose(0, 1).squeeze() + 1e-6)
            r = r_new / (r_new.sum() + 1e-6)
            diff = torch.norm(r - r_old)
            if diff < 1e-5 or (_ > 2 and diff < 1e-4):
                break

        A_s = cov * r.transpose(0, 1)  # [sub_nnodes, sub_nnodes]
        adj = F.relu(torch.tanh(self.alpha * A_s))  # [sub_nnodes, sub_nnodes]

        # Top-k sparsification
        k_num = min(self.k, sub_nnodes)
        _, indices = torch.topk(adj, k=k_num, dim=1)
        mask = torch.zeros_like(adj)
        mask.scatter_(1, indices, 1.0)
        adj = adj * mask  # [sub_nnodes, sub_nnodes]

        # Cache adj
        self.adj_cache[idx_tuple] = adj.detach()

        return adj.to(self.device)

    def fullA(self, idx):
        idx = idx.long()  # Ensure idx is of type int64
        sub_nnodes = idx.size(0)

        # Check cache
        idx_tuple = tuple(idx.cpu().numpy())
        if idx_tuple in self.adj_cache:
            return self.adj_cache[idx_tuple].to(self.device)

        if self.static_feat is None:
            nodevec = torch.zeros(sub_nnodes, self.nnodes, device=self.device)
            nodevec.scatter_(1, idx.unsqueeze(1), 1.0)  # Scatter operation with idx
            nodevec1 = nodevec
            nodevec2 = nodevec
        else:
            nodevec1 = self.static_feat[idx, :]
            nodevec2 = nodevec1

        nodevec1 = torch.tanh(self.alpha * self.lin1(nodevec1))
        nodevec2 = torch.tanh(self.alpha * self.lin2(nodevec2))
        X_em = (nodevec1 + nodevec2) / 2

        X_em = (X_em - X_em.mean(dim=0, keepdim=True)) / (X_em.std(dim=0, keepdim=True) + 1e-6)

        if sub_nnodes > self.dim:
            try:
                U, S, Vh = torch.linalg.svd(X_em, full_matrices=False)
                S = S.clamp(min=1e-4)
                rank = min(self.dim, S.size(0))
                X_em = U[:, :rank] @ torch.diag(S[:rank])
            except RuntimeError:
                print("SVD failed, skipping dimensionality reduction")

        cov = torch.matmul(X_em, X_em.transpose(0, 1))
        cov = self._stabilize_cov(cov, sub_nnodes)

        try:
            eigenvalues, v = torch.linalg.eigh(cov)
            if torch.any(eigenvalues < -1e-6):
                print("Negative eigenvalues detected, using SVD")
                raise RuntimeError
        except RuntimeError:
            try:
                u, s, vh = torch.linalg.svd(cov, full_matrices=True)
                s = s.clamp(min=1e-4)
                v = u
            except RuntimeError:
                v = torch.eye(sub_nnodes, device=self.device)

        M = (v.transpose(0, 1) @ v) ** 2
        b = (X_em.transpose(0, 1) @ v) ** 2
        b = b.sum(dim=0)

        r_sub = self.r[:, idx]
        r = r_sub / (torch.matmul(r_sub, b.unsqueeze(1)) + 1e-6)

        for _ in range(10):
            r_old = r.clone()
            Mr = torch.matmul(M, r.transpose(0, 1))
            rMr = torch.matmul(r, Mr).squeeze()
            r_new = r * b * rMr / (Mr.transpose(0, 1).squeeze() + 1e-6)
            r = r_new / (r_new.sum() + 1e-6)
            diff = torch.norm(r - r_old)
            if diff < 1e-5 or (_ > 2 and diff < 1e-4):
                break

        A_s = cov * r.transpose(0, 1)
        adj = F.relu(torch.tanh(self.alpha * A_s))

        k_num = min(self.k, sub_nnodes)
        _, indices = torch.topk(adj, k=k_num, dim=1)
        mask = torch.zeros_like(adj)
        mask.scatter_(1, indices, 1.0)
        adj = adj * mask

        self.adj_cache[idx_tuple] = adj.detach()

        return adj.to(self.device)






################## ADDITIONAL GRAPH LEARNING LAYER TO BE POTENTIALLY APPLIED ##################

class graph_global(nn.Module):
    def __init__(self, nnodes, k, dim, device, alpha=3, static_feat=None):
        super(graph_global, self).__init__()
        self.nnodes = nnodes
        self.A = nn.Parameter(torch.randn(nnodes, nnodes).to(device), requires_grad=True).to(device)

    def forward(self, idx):
        return F.relu(self.A)


class graph_undirected(nn.Module):
    def __init__(self, nnodes, k, dim, device, alpha=3, static_feat=None):
        super(graph_undirected, self).__init__()
        self.nnodes = nnodes
        if static_feat is not None:
            xd = static_feat.shape[1]
            self.lin1 = nn.Linear(xd, dim)
        else:
            self.emb1 = nn.Embedding(nnodes, dim)
            self.lin1 = nn.Linear(dim,dim)

        self.device = device
        self.k = k
        self.dim = dim
        self.alpha = alpha
        self.static_feat = static_feat

    def forward(self, idx):
        if self.static_feat is None:
            nodevec1 = self.emb1(idx)
            nodevec2 = self.emb1(idx)
        else:
            nodevec1 = self.static_feat[idx,:]
            nodevec2 = nodevec1

        nodevec1 = torch.tanh(self.alpha*self.lin1(nodevec1))
        nodevec2 = torch.tanh(self.alpha*self.lin1(nodevec2))

        a = torch.mm(nodevec1, nodevec2.transpose(1,0))
        adj = F.relu(torch.tanh(self.alpha*a))
        mask = torch.zeros(idx.size(0), idx.size(0)).to(self.device)
        mask.fill_(float('0'))
        s1,t1 = adj.topk(self.k,1)
        mask.scatter_(1,t1,s1.fill_(1))
        adj = adj*mask
        return adj



class graph_directed(nn.Module):
    def __init__(self, nnodes, k, dim, device, alpha=3, static_feat=None):
        super(graph_directed, self).__init__()
        self.nnodes = nnodes
        if static_feat is not None:
            xd = static_feat.shape[1]
            self.lin1 = nn.Linear(xd, dim)
            self.lin2 = nn.Linear(xd, dim)
        else:
            self.emb1 = nn.Embedding(nnodes, dim)
            self.emb2 = nn.Embedding(nnodes, dim)
            self.lin1 = nn.Linear(dim,dim)
            self.lin2 = nn.Linear(dim,dim)

        self.device = device
        self.k = k
        self.dim = dim
        self.alpha = alpha
        self.static_feat = static_feat

    def forward(self, idx):
        if self.static_feat is None:
            nodevec1 = self.emb1(idx)
            nodevec2 = self.emb2(idx)
        else:
            nodevec1 = self.static_feat[idx,:]
            nodevec2 = nodevec1

        nodevec1 = torch.tanh(self.alpha*self.lin1(nodevec1))
        nodevec2 = torch.tanh(self.alpha*self.lin2(nodevec2))

        a = torch.mm(nodevec1, nodevec2.transpose(1,0))
        adj = F.relu(torch.tanh(self.alpha*a))
        mask = torch.zeros(idx.size(0), idx.size(0)).to(self.device)
        mask.fill_(float('0'))
        s1,t1 = adj.topk(self.k,1)
        mask.scatter_(1,t1,s1.fill_(1))
        adj = adj*mask
        return adj
