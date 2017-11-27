from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import itertools
from torch.nn.utils.rnn import pack_padded_sequence as pack 
import torch.nn.functional as F

def atanh(x):
    return 0.5 * (torch.log(1+x) - torch.log(1-x))
# 0.5 = tanh(W*0.5 + 0.5), W = 2*(atanh(0.5) - 0.5)
#
def atanhnp(x):
    return 0.5 * (np.log(1+x) - np.log(1-x))

# rnn stuff
def tanh_cell(W):
    return lambda x, h: nn.Tanh()(x @ Variable(torch.eye(2)) + h @ W)
    
def rnn(xs, h0=None, cell=None):
    if h0 is not None:
        hs = [h0]
    else:
        h0 = [Variable(torch.zeros_like(xs[0].data))]
    for x in xs:
        hs.append(cell(x, hs[-1]))
    return torch.cat(hs[1:])

W = Variable(atanh(torch.randn(2,2).fill_(0.5)) - 0.5, requires_grad=True)
x = [Variable(torch.rand(1,1,2).fill_(0.5), requires_grad=True) for _ in range(5)]

cell = tanh_cell(W)

#o = list(itertools.accumulate(x, cell))
o = rnn(x, h0=Variable(x[0].data.new(1,1,2).fill_(0.5)), cell=cell)

real_rnn = nn.RNN(2, 2, 1, nonlinearity="tanh", bias=False)
real_rnn.weight_hh_l0.data.copy_(W.data)
real_rnn.weight_ih_l0.data.copy_(torch.eye(2))

ro = real_rnn(torch.cat(x).view(-1, 1, 2))[0]

y = np.array([0.5, 0.5])
z = np.tanh(np.tanh(y @ W.data.numpy() + y @ np.eye(2)) @ W.data.numpy() + y @ np.eye(2))

# embedding stuff
# WTS that we can using a composition function then fill in an embedding entry with the result

unigram_size = 11

phrases = {
    11: [2,3],
    12: [8,9,10],
}

sentences = [
    [1, 2, 11, 5, 6, 7, 12],
    [2, 3, 5, 11, 10, 9, 8],
]
S = torch.LongTensor(sentences).t()

sentence = [1,2,11,5,6,7,12]
sentenceT = Variable(torch.LongTensor(sentence))

lut = nn.Embedding(unigram_size + len(phrases), 5, padding_idx=0)
crnn = nn.RNN(5,5,1,bias=False)

comp = lambda x: crnn(x)[-1]

def process(val):
    if val < unigram_size:
        return lut(Variable(torch.LongTensor([val]))).view(1, 1, -1)
    return comp(lut(Variable(torch.LongTensor(phrases[val]))).view(-1, 1, 5))

lut.zero_grad()
crnn.zero_grad()
seq = torch.cat([process(val) for val in sentence], 0)
print("Unbatched output")
print(seq)
seq.backward(torch.ones_like(seq))
print("Unbatched rnn reccurrent grad")
print(crnn.weight_hh_l0.grad)
print("Lut grad")
print(lut.weight.grad)

# batched version
buf = torch.LongTensor()
copybuf = torch.LongTensor()
indices = sorted(set(S[S.ge(unigram_size)]), key=lambda x: len(phrases[x]), reverse=True)
maxlen = max(len(phrases[id]) for id in indices)
buf.resize_(maxlen, len(indices)).fill_(0)
for i, idx in enumerate(indices):
    phrase = deepcopy(phrases[idx])
    phrase = phrase + [0] * (maxlen - len(phrase))
    buf[:, i].copy_(buf.new(phrase))
iT = torch.LongTensor(list(indices))
pT = comp(lut(Variable(buf)))
# doesn't work because lut.weight is a leaf so cannot be mutated
#lut.weight[iT] = pT.view(2,5)

# maybe need to construct embeddings matrix explicitly through autograd?
# we want all matrices to be sparse

# torch.cuda.sparse() http://pytorch.org/docs/master/sparse.html
# sparse not developed enough for this.
# sadly will not work
 
lut_mat = torch.cat(
    [lut.weight] + [],
    0
)
#F.embedding()

# zero phrase embeddings and add lstm output
crnn.zero_grad()
lut.zero_grad()
for k,_ in phrases.items():
    lut.weight.data[k].fill_(0)
buf = torch.LongTensor()

indices = sorted(set(S[S.ge(unigram_size)]), key=lambda x: len(phrases[x]), reverse=True)
lengths = list(map(lambda x: len(phrases[x]), indices))
lol = {v:k for k,v in enumerate(indices)}
meh = S[S.ge(unigram_size)].apply_(lambda x: lol[x])
maxlen = max(len(phrases[id]) for id in indices)

buf.resize_(maxlen, len(indices)).fill_(0)
for i, idx in enumerate(indices):
    phrase = deepcopy(phrases[idx])
    phrase += [0] * (maxlen - len(phrase))
    buf[:, i].copy_(buf.new(phrase))
iT = torch.LongTensor(list(indices))
pT = comp(pack(lut(Variable(buf)), lengths))
o = lut(Variable(S))
y = o \
    .view(-1, 5) \
    .masked_scatter(Variable(S.ge(unigram_size).view(-1,1)), pT.view(-1,5)[meh]) \
    .view(7,2,5)
z = y[:,0,:]
z.backward(torch.ones_like(z))

print("Batched output")
print(z - seq.squeeze())
print("Batched rnn recurrent grad")
print(crnn.weight_hh_l0.grad)
print("lut grad")
print(lut.weight.grad)
