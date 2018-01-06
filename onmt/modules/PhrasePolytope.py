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

class PhrasePolytope(nn.Module):
    def __init__(self, phrase_lut, n=32):
        super(PhrasePolytope, self).__init__()
        self.phrase_lut = phrase_lut
        self.nuni = phrase_lut.lut.num_embeddings
        self.nphr = phrase_lut.phrase_vocab_size
        self.n = n

    def reset_parameters(self):
        self.dense_lut.reset_parameters()
        self.sparse_lut.reset_parameters()

    def reset_perm(self):
        """Using a permutation like this is probably almost equivalent to reservoir sampling"""
        self.perm = torch.randperm(self.nphr) + self.nuni

    def prepare_projection(self, target, n):
        """ Sets the phrase vertices of the softmax projection polytope """
        mask = target.ge(self.nuni)
        phrases = set(target[mask])
        distractors = self.get_distractors(n, self.perm, phrases)
        self.vertices = sorted(list(phrases | distractors))
        self.v2i = {v: i for i, v in enumerate(self.vertices)}
        self.vertT = target.new(self.vertices).view(-1, 1, 1)

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
        self.target = target.new(list(map(
            lambda x: self.v2i[x] + self.nuni if x > self.nuni else x,
            target.data.view(-1).tolist()
        ))).view_as(target)
        return self.target

    def forward(self, input):
        proj = torch.cat((self.phrase_lut.lut.weight, self.phrase_lut(self.vertT).squeeze(1)), 0) if self.training else self.phrase_lut.full_lut.weight
        bias = None # for now
        return F.linear(input, proj, bias)

