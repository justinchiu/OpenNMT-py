
import torch
import torch.nn as nn

from torch.autograd import Variable as V

import onmt
from onmt.modules.Embeddings import PhraseEmbeddings

from itertools import accumulate

"""
emb = PhraseEmbeddings(
    embedding_dim,
    False,
    0,
    0,
)
"""

# Input should be (src tokens, src lengths, repeat mask?) -> (final_hiddens, hiddens, context_lengths)
# First model will be
# Embeddings -> Encoder -> Replicator
# Need to gather into Replicator after concatenating a dummy timestep which we can use for padding
# Second will be
# Embeddings -> Encoder -> Reducer
#

# implementation 1: gather
T, N = 5, 3
x = torch.arange(T*N).view(N, T).t() + 1

padidx = -1
# RLE
lengthss = [[1,1,1,3,1], [1,2,1,1,1], [1]*T]
maxlen = max(map(sum, lengthss))
targets = [ [ x for i, count in enumerate(lengths) for x in [i] * count ] for lengths in lengthss ]
for i, target in enumerate(targets):
    diff = maxlen - len(target)
    if diff > 0:
        target.extend([padidx] * diff)


# make it really obvious where the padding is!
pad = torch.FloatTensor(1).fill_(float("nan"))
xp = torch.cat([pad.expand(1, N), x], 0)
o = xp.gather(0, (torch.LongTensor(targets)+1).t())

# Real attempt
def rle_to_idx(rle, padidx):
    maxlen = len(max())
def repeat_elements(x, targets, pad):
    T, N, H = x.size()
    # We pad on the front with NaNs
    xp = torch.cat([V(pad).expand(1, N, H), x], 0)
    # I guess since targets is the T-position, but shifted by 1, we need to shift it.
    # Also, I guess it's a list coming in, so we need to turn it into a tensor, lol.
    targets = (torch.LongTensor(targets) + 1).t()
    # targets is contiguous along T, but we need it to be contiguous along N.
    return xp.gather(0, V(targets.view(-1, N, 1).expand(-1, N, H)))

lut = nn.Embedding(50, 16)
lut.weight.data.copy_(torch.arange(50).view(-1, 1).expand(50, 16))
enc_out = lut(V(x.long()))

context = repeat_elements(enc_out, targets, pad)

diff = context[:,:,0].data - o
for lengths in lengthss:
    assert(diff[:sum(lengths),0].sum() == 0)

# TODO(justinchiu): implementation 2: flatten + index

