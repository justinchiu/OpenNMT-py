import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter as P
from torch.autograd import Variable as V

D = 10
nuni = 50
nphrase = 50
nv = nuni + nphrase

lut = nn.Embedding(nv, D)
proj = lut.weight

uni = nn.Embedding(nuni, D)
phr = nn.Embedding(nphrase, D)
uni.weight.data = lut.weight.data[:nuni]
phr.weight.data = lut.weight.data[nuni:]

T = 10
N = 4

y = torch.Tensor(T, N).uniform_(nv).long()
while y.ge(nuni).sum() <= 0:
    y = torch.Tensor(bsz).uniform_(nv).long()
x = torch.Tensor(T, N).uniform_(nv).long()

# baseline
def baseline(vx, vy):
    lut.zero_grad()
    e = lut(vx) @ proj.t()
    loss = F.cross_entropy(e.view(-1, nv), vy.view(-1))
    loss.backward()
    baseline_grad = lut.weight.grad.data.clone()
    return loss, baseline_grad, e

baseline_loss_wrong, baseline_grad_wrong = baseline(V(x), V(y))
baseline_loss_right, baseline_grad_right = baseline(V(y), V(y))

# sampled
def sample(vx, vy):
    nd = 8
    perm = torch.randperm(nphrase) + nuni
    mask = y.ge(nuni)
    phrases = set(y[mask])

    def get_distractors(n, perm, phrases):
        i = 0
        d = set() 
        while len(d) < n:
            if not perm[i] in phrases:
                d.add(perm[i])
            i += 1
        return d

    distractors = get_distractors(nd, perm, phrases)

    vertices = sorted(list(phrases | distractors))
    v2i = {v: i for i, v in enumerate(vertices)}

    # only forward on unigrams and phrases
    uni.zero_grad()
    phr.zero_grad()
    verT = torch.LongTensor(vertices)
    sproj = torch.cat((uni.weight, phr(V(verT) - nuni)), 0) 
    e = lut(vx) @ sproj.t()
    y2 = torch.LongTensor(list(map(lambda x: v2i[x] + nuni if x > nuni else x, y.view(-1))))
    loss = F.cross_entropy(e.view(T * N, -1), V(y2).view(-1)) 
    loss.backward()
    return loss, torch.cat((uni.weight.grad, phr.weight.grad), 0).data, e

sample_loss_wrong, sample_grad_wrong = sample(V(x), V(y))
sample_loss_right, sample_grad_right = sample(V(y), V(y))

# biggest differences are usually with tokens that *appear* in batch
maxvr, maxir = (baseline_grad_right - sample_grad_right).max(0)
minvr, minir = (baseline_grad_right - sample_grad_right).min(0)
maxvw, maxiw = (baseline_grad_wrong - sample_grad_wrong).max(0)
minvw, miniw = (baseline_grad_wrong - sample_grad_wrong).min(0)

# suppose you have V vertices of a polytope that are drawn from a von mises - Fisher distribution
# as well as a vertice X.
# given N draws from some distribution q(V | X), what's the distribution over the percentage of the
# log partition function you can estimate?

# class
# We have a dense weight as well as a sparse weight that's expensive to evaluate, so we need the concatenation of both,
# but we only want what we need from the sparse section.
class CatLinear(nn.Module):
    def __init__(self, dense_weight, sparse_fn):
        super(CatLinear, self).__init__()
        self.dense_weight = dense_weight
        self.sparse_fn = sparse_fn

    def reset_parameters(self):
        pass

    def forward(self, input):
        pass

    def __repre__(self):
        pass
