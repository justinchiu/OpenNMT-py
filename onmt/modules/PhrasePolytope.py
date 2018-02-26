import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable as V

# suppose you have V vertices of a polytope that are drawn from a von mises - Fisher distribution
# as well as a vertice X.
# given N draws from some distribution q(V | X), what's the distribution over the percentage of the
# log partition function you can estimate?

# We have a dense weight as well as a sparse weight that's expensive to evaluate, so we need the concatenation of both,
# but we only want what we need from the sparse section.

# TODO(justc): need to divide by q(x)

class PhrasePolytope(nn.Module):
    def __init__(self, phrase_lut, n=32):
        super(PhrasePolytope, self).__init__()
        self.phrase_lut = phrase_lut
        self.nuni = phrase_lut.lut.num_embeddings
        self.nphr = phrase_lut.phrase_vocab_size
        self.n = n

    def reset_perm(self):
        """Using a permutation like this is probably almost equivalent to reservoir sampling"""
        self.perm = torch.randperm(self.nphr) + self.nuni

    def prepare_projection(self, target, n):
        """ Sets the phrase vertices of the softmax projection polytope """
        mask = target.ge(self.nuni)
        phrases = set(target[mask].data.tolist())
        distractors = self.get_distractors(n, self.perm, phrases)
        self.vertices = sorted(list(phrases | distractors))
        self.v2i = {v: i for i, v in enumerate(self.vertices)}
        self.vertT = V(target.data.new(self.vertices).view(-1, 1, 1))

    def get_distractors(self, n, perm, phrases):
        """ Get a set of distractors """
        i = 0
        d = set() 
        while len(d) < n:
            if not perm[i] in phrases:
                d.add(perm[i])
            i += 1
        self.d = d
        return d

    def collapse_target(self, target):
        """ Collapse Target """
        self.prepare_projection(target, self.n)
        self.target = V(target.data.new(list(map(
            lambda x: self.v2i[x] + self.nuni if x > self.nuni else x,
            target.data.view(-1).tolist()
        )))).view_as(target)
        return self.target

    def forward(self, input):
        if self.training:
            self.plut_output = self.phrase_lut(self.vertT).squeeze(1)
            proj = torch.cat((self.phrase_lut.lut.weight, self.plut_output), 0)
        else:
            proj = self.phrase_lut.full_lut.weight
        bias = None # for now
        return F.linear(input, proj, bias)

