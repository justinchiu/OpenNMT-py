import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable as V
from torch.nn import Parameter 

class RepeatContext(nn.Module):
    def __init__(self, field=None, dataset=None, lut=None, dropout=0):
        super(RepeatContext, self).__init__()
        # We always insert padding in front
        self.padidx = -1
        self.pad = Parameter(
            data=torch.FloatTensor(1),
            requires_grad=False
        )
        self.pad.data.fill_(float("nan"))
        # Sigh
        self.register_buffer("lt", torch.LongTensor(1))

        # Add word vectors
        self.field = field
        self.dataset = dataset# NEEDS TO BE SET DEPENDING ON DATA...
        self.lut = lut
        if lut is not None:
            self.proj = nn.Linear(lut.weight.size(-1), 1)

    def rle_to_idxs(self, rles):
        lengths = list(map(sum, rles))
        maxlen = max(lengths)
        # Sigh
        self.lengths = self.lt.new(lengths)
        targets = [
            [ x for i, count in enumerate(rle) for x in [i] * count ]
            for rle in rles
        ]
        for i, target in enumerate(targets):
            diff = maxlen - len(target)
            if diff > 0:
                target.extend([self.padidx] * diff)
        # I guess since targets is the T-position, but shifted by 1, we need to shift it.
        targets = (self.lt.new(targets) + 1).t().contiguous()
        self.targets = V(targets)
        return self.targets, self.lengths

    def get_words(self, batch_indices):
        self.word_idxs, lengths = self.field.numericalize(
            self.field.pad(self.dataset[x].src for x in batch_indices.data.tolist())
        )

    def forward(self, x):
        T, N, H = x.size()
        # We pad on the front with NaNs
        px = torch.cat([self.pad.expand(1, N, H), x], 0)
        out = px.gather(0, self.targets.view(-1, N, 1).expand(-1, N, H))
        if self.lut is None:
            return out, self.lengths
        else:
            # This will be the source embeddings, either directly the words
            # or the positions.
            wout = self.lut(self.word_idxs)
            # hm...how to decide?
            p = F.sigmoid(self.proj(out + wout))
            return (1-p) * out + p * wout, self.lengths

