import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter as P
from torch.autograd import Variable as V

from onmt.modules.Embeddings import PhraseEmbeddings
from onmt.modules import PhrasePolytope

D = 10
nuni = 50
nphrase = 50
nv = nuni + nphrase
nd = 8

T = 10
N = 4

y = torch.Tensor(T, N, 1).uniform_(nv).long()
while y.ge(nuni).sum() <= 0:
    y = torch.Tensor(T, N, 1).uniform_(nv).long()
x = torch.Tensor(T, N, 1).uniform_(nv).long()

phrase_mapping = {i + nuni : [i, i+1] if i < 49 else [3,4,5] for i in range(nphrase)}
comp = nn.LSTM(10, 10)

phrase_lut = PhraseEmbeddings(D, False, 0, 0, nuni, phrase_mapping, comp)
idxs = V(torch.arange(nuni, nv).long().view(-1, 1, 1))

lut = nn.Embedding(nv, D)

# baseline
def baseline(vx, vy):
    lut.zero_grad()
    e = lut(vx.view(-1)) @ lut.weight.t()
    loss = F.cross_entropy(e.view(-1, nv), vy.view(-1))
    loss.backward()
    baseline_grad = lut.weight.grad.data.clone()
    return loss, baseline_grad, e

baseline_loss_wrong, baseline_grad_wrong, baseline_e_wrong = baseline(V(x), V(y))
baseline_loss_right, baseline_grad_right, baseline_e_right = baseline(V(y), V(y))

# sampled
def get_distractors(n, perm, phrases):
    i = 0
    d = set()
    while len(d) < n:
        if not perm[i] in phrases:
            d.add(perm[i])
        i += 1
    return d

def sample(vx, vy):
    perm = torch.randperm(nphrase) + nuni
    mask = y.ge(nuni)
    phrases = set(y[mask])
    distractors = get_distractors(nd, perm, phrases)

    vertices = sorted(list(phrases | distractors))
    v2i = {v: i for i, v in enumerate(vertices)}

    # only forward on unigrams and phrases
    verT = torch.LongTensor(vertices)
    sproj = torch.cat((lut.weight[:nuni], lut(V(verT))), 0)
    e = lut(vx.view(-1)) @ sproj.t()
    y2 = torch.LongTensor(list(map(lambda x: v2i[x] + nuni if x > nuni else x, y.view(-1))))
    loss = F.cross_entropy(e.view(T * N, -1), V(y2).view(-1))
    loss.backward()
    return loss, lut.weight.grad.data, e, vertices, sproj

sample_loss_wrong, sample_grad_wrong, sample_e_wrong, vertices_wrong, sproj = sample(V(x), V(y))
sample_loss_right, sample_grad_right, sample_e_right, vertices_right, sproj = sample(V(y), V(y))

# biggest differences are usually with tokens that *appear* in batch
maxvr, maxir = (baseline_grad_right - sample_grad_right).max(0)
minvr, minir = (baseline_grad_right - sample_grad_right).min(0)
maxvw, maxiw = (baseline_grad_wrong - sample_grad_wrong).max(0)
minvw, miniw = (baseline_grad_wrong - sample_grad_wrong).min(0)

diff_wrong = baseline_e_wrong[:,torch.LongTensor(vertices_wrong)] - sample_e_wrong[:,nuni:]
diff_right = baseline_e_right[:,torch.LongTensor(vertices_right)] - sample_e_right[:,nuni:]

print(baseline_loss_wrong)
print(baseline_loss_right)
print(sample_loss_wrong)
print(sample_loss_right)
print(diff_wrong.max())
print(diff_right.max())

# baseline
def baseline(vx, vy):
    proj = torch.cat((phrase_lut.lut.weight, phrase_lut(idxs).view(-1, D)), 0)
    e = vx @ proj.t()
    loss = F.cross_entropy(e.view(-1, nv), vy.view(-1))
    loss.backward()
    baseline_grad = phrase_lut.lut.weight.grad.data
    return loss, baseline_grad, e

baseline_loss_wrong, baseline_grad_wrong, baseline_e_wrong = baseline(lut(V(x).squeeze(2)), V(y))
baseline_loss_right, baseline_grad_right, baseline_e_right = baseline(phrase_lut(V(y)), V(y))

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
    phrase_lut.zero_grad()
    verT = torch.LongTensor(vertices).view(-1, 1, 1)
    sproj = torch.cat((phrase_lut.lut.weight, phrase_lut(V(verT)).view(-1, D)), 0) 
    e = vx @ sproj.t()
    y2 = torch.LongTensor(list(map(lambda x: v2i[x] + nuni if x > nuni else x, y.view(-1))))
    loss = F.cross_entropy(e.view(T * N, -1), V(y2)) 
    loss.backward()
    return loss, phrase_lut.lut.weight.grad.data, e, vertices, sproj, perm

sample_loss_wrong, sample_grad_wrong, sample_e_wrong, vertices_wrong, sproj, perm_wrong = sample(lut(V(x).squeeze(2)), V(y))
sample_loss_right, sample_grad_right, sample_e_right, vertices_right, sproj, perm_right = sample(phrase_lut(V(y)), V(y))

maxvr, maxir = (baseline_grad_right - sample_grad_right).max(0)
minvr, minir = (baseline_grad_right - sample_grad_right).min(0)
maxvw, maxiw = (baseline_grad_wrong - sample_grad_wrong).max(0)
minvw, miniw = (baseline_grad_wrong - sample_grad_wrong).min(0)

diff_wrong = baseline_e_wrong[:,:,torch.LongTensor(vertices_wrong)] - sample_e_wrong[:,:,nuni:]
diff_right = baseline_e_right[:,:,torch.LongTensor(vertices_right)] - sample_e_right[:,:,nuni:]

print(baseline_loss_wrong)
print(baseline_loss_right)
print(sample_loss_wrong)
print(sample_loss_right)
print(diff_wrong.max())
print(diff_right.max())

# suppose you have V vertices of a polytope that are drawn from a von mises - Fisher distribution
# as well as a vertice X.
# given N draws from some distribution q(V | X), what's the distribution over the percentage of the
# log partition function you can estimate?

# We have a dense weight as well as a sparse weight that's expensive to evaluate, so we need the concatenation of both,
# but we only want what we need from the sparse section.

pp = PhrasePolytope(phrase_lut, nd)
pp.perm = perm_wrong
cy = pp.collapse_target(y)
energies = pp.forward(lut(V(x).squeeze(2)))
assert(F.cross_entropy(energies.view(T * N, -1), V(cy).view(-1)) == sample_loss_wrong)
assert((sample_e_wrong - energies).ne(0).sum() == 0)

pp.perm = perm_right
cy = pp.collapse_target(y)
energies = pp.forward(phrase_lut(V(y)))
assert(F.cross_entropy(energies.view(T * N, -1), V(cy).view(-1)) == sample_loss_right)
assert((sample_e_right - energies).ne(0).sum() == 0)
