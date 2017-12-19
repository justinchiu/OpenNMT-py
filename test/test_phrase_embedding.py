import unittest
import torch
from torch import nn
from copy import deepcopy
import itertools
from torch.nn.utils.rnn import pack_padded_sequence as pack
import torch.nn.functional as F


import sys
sys.path.append("/n/home13/jchiu/projects/OpenNMT-py")
import onmt
from onmt.modules.Embeddings import PhraseEmbeddings

from torch.autograd import Variable

def atanh(x):
    return 0.5 * (torch.log(1+x) - torch.log(1-x))

class TestPhraseEmbedding(unittest.TestCase):
    def test_phrase_embeddings(self):

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

        # unbatched

        sentence = [1,2,11,5,6,7,12]
        sentenceT = Variable(torch.LongTensor(sentence))

        lut = nn.Embedding(unigram_size + len(phrases), 5, padding_idx=0)
        crnn = nn.LSTM(5,5,1,bias=False)

        # Return final hidden
        comp = lambda x: crnn(x)[-1][0]

        def process(val):
            if val < unigram_size:
                return lut(Variable(torch.LongTensor([val]))).view(1, 1, -1)
            return comp(lut(Variable(torch.LongTensor(phrases[val]))).view(-1, 1, 5))

        lut.zero_grad()
        crnn.zero_grad()
        seq = torch.cat([process(val) for val in sentence], 0)
        print("Unbatched output")
        print(seq.squeeze(1))
        seq.backward(torch.ones_like(seq))
        print("Unbatched rnn reccurrent grad")
        print(crnn.weight_hh_l0.grad)
        print("Lut grad")
        print(lut.weight.grad)

        # batched
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
        print(z)
        #print(z - seq.squeeze())
        print("Batched rnn recurrent grad")
        print(crnn.weight_hh_l0.grad)
        print("lut grad")
        print(lut.weight.grad)

        plut = PhraseEmbeddings(
            5,
            False,
            0,
            0,
            unigram_size,
            phrases,
            crnn
        )
        plut.full_lut.weight.data.copy_(lut.weight.data)

        lol = (plut(Variable(S.contiguous().view(S.size(0), S.size(1), 1))))
        meh = lol[:,0]
        meh.backward(torch.ones_like(meh))
        print(meh)
        print("Batched rnn recurrent grad")
        print(crnn.weight_hh_l0.grad)
        print("lut grad")
        print(lut.weight.grad)
       
