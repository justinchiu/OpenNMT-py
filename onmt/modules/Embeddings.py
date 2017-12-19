import torch
import torch.nn as nn
from torch.autograd import Variable

from torch.nn.utils.rnn import pack_padded_sequence as pack

from onmt.modules import BottleLinear, Elementwise
from onmt.Utils import aeq


class PositionalEncoding(nn.Module):

    def __init__(self, dropout, dim, max_len=5000):
        pe = torch.arange(0, max_len).unsqueeze(1).expand(max_len, dim)
        div_term = 1 / torch.pow(10000, torch.arange(0, dim * 2, 2) / dim)
        pe = pe * div_term.expand_as(pe)
        pe[:, 0::2] = torch.sin(pe[:, 0::2])
        pe[:, 1::2] = torch.cos(pe[:, 1::2])
        pe = pe.unsqueeze(1)
        super(PositionalEncoding, self).__init__()
        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, emb):
        # We must wrap the self.pe in Variable to compute, not the other
        # way - unwrap emb(i.e. emb.data). Otherwise the computation
        # wouldn't be watched to build the compute graph.
        emb = emb + Variable(self.pe[:emb.size(0), :1, :emb.size(2)]
                             .expand_as(emb), requires_grad=False)
        emb = self.dropout(emb)
        return emb


class Embeddings(nn.Module):
    """
    Words embeddings dictionary for encoder/decoder.

    Args:
        word_vec_size (int): size of the dictionary of embeddings.
        position_encoding (bool): use a sin to mark relative words positions.
        feat_merge (string): merge action for the features embeddings:
                    concat, sum or mlp.
        feat_vec_exponent (float): when using '-feat_merge concat', feature
                    embedding size is N^feat_dim_exponent, where N is the
                    number of values of feature takes.
        feat_vec_size (int): embedding dimension for features when using
                    '-feat_merge mlp'
        dropout (float): dropout probability.
        word_padding_idx (int): padding index for words in the embeddings.
        feats_padding_idx ([int]): padding index for a list of features
                                   in the embeddings.
        word_vocab_size (int): size of dictionary of embeddings for words.
        feat_vocab_sizes ([int], optional): list of size of dictionary
                                    of embeddings for each feature.
    """
    def __init__(self, word_vec_size, position_encoding, feat_merge,
                 feat_vec_exponent, feat_vec_size, dropout,
                 word_padding_idx, feat_padding_idx,
                 word_vocab_size, feat_vocab_sizes=[]):

        self.word_padding_idx = word_padding_idx

        # Dimensions and padding for constructing the word embedding matrix
        vocab_sizes = [word_vocab_size]
        emb_dims = [word_vec_size]
        pad_indices = [word_padding_idx]

        # Dimensions and padding for feature embedding matrices
        # (these have no effect if feat_vocab_sizes is empty)
        if feat_merge == 'sum':
            feat_dims = [word_vec_size] * len(feat_vocab_sizes)
        elif feat_vec_size > 0:
            feat_dims = [feat_vec_size] * len(feat_vocab_sizes)
        else:
            feat_dims = [int(vocab ** feat_vec_exponent)
                         for vocab in feat_vocab_sizes]
        vocab_sizes.extend(feat_vocab_sizes)
        emb_dims.extend(feat_dims)
        pad_indices.extend(feat_padding_idx)

        # The embedding matrix look-up tables. The first look-up table
        # is for words. Subsequent ones are for features, if any exist.
        emb_params = zip(vocab_sizes, emb_dims, pad_indices)
        embeddings = [nn.Embedding(vocab, dim, padding_idx=pad)
                      for vocab, dim, pad in emb_params]
        emb_luts = Elementwise(feat_merge, embeddings)

        # The final output size of word + feature vectors. This can vary
        # from the word vector size if and only if features are defined.
        # This is the attribute you should access if you need to know
        # how big your embeddings are going to be.
        self.embedding_size = (sum(emb_dims) if feat_merge == 'concat'
                               else word_vec_size)

        # The sequence of operations that converts the input sequence
        # into a sequence of embeddings. At minimum this consists of
        # looking up the embeddings for each word and feature in the
        # input. Model parameters may require the sequence to contain
        # additional operations as well.
        super(Embeddings, self).__init__()
        self.make_embedding = nn.Sequential()
        self.make_embedding.add_module('emb_luts', emb_luts)

        if feat_merge == 'mlp':
            in_dim = sum(emb_dims)
            out_dim = word_vec_size
            mlp = nn.Sequential(BottleLinear(in_dim, out_dim), nn.ReLU())
            self.make_embedding.add_module('mlp', mlp)

        if position_encoding:
            pe = PositionalEncoding(dropout, self.embedding_size)
            self.make_embedding.add_module('pe', pe)

    @property
    def word_lut(self):
        return self.make_embedding[0][0]

    @property
    def emb_luts(self):
        return self.make_embedding[0]

    def load_pretrained_vectors(self, emb_file, fixed):
        if emb_file:
            pretrained = torch.load(emb_file)
            self.word_lut.weight.data.copy_(pretrained)
            if fixed:
                self.word_lut.weight.requires_grad = False

    def forward(self, input):
        """
        Return the embeddings for words, and features if there are any.
        Args:
            input (LongTensor): len x batch x nfeat
        Return:
            emb (FloatTensor): len x batch x self.embedding_size
        """
        in_length, in_batch, nfeat = input.size()
        aeq(nfeat, len(self.emb_luts))

        emb = self.make_embedding(input)

        out_length, out_batch, emb_size = emb.size()
        aeq(in_length, out_length)
        aeq(in_batch, out_batch)
        aeq(emb_size, self.embedding_size)

        return emb

class PhraseEmbeddings(nn.Module):
    """
    Words embeddings dictionary for encoder/decoder.

    Args:
        word_vec_size (int): size of the dictionary of embeddings.
        position_encoding (bool): use a sin to mark relative words positions.
        feat_merge (string): merge action for the features embeddings:
                    concat, sum or mlp.
        feat_vec_exponent (float): when using '-feat_merge concat', feature
                    embedding size is N^feat_dim_exponent, where N is the
                    number of values of feature takes.
        feat_vec_size (int): embedding dimension for features when using
                    '-feat_merge mlp'
        dropout (float): dropout probability.
        word_padding_idx (int): padding index for words in the embeddings.
        feats_padding_idx ([int]): padding index for a list of features
                                   in the embeddings.
        word_vocab_size (int): size of dictionary of embeddings for words.
        feat_vocab_sizes ([int], optional): list of size of dictionary
                                    of embeddings for each feature.
    """
    def __init__(
        self,
        word_vec_size,
        position_encoding,
        dropout,
        word_padding_idx,
        word_vocab_size,
        phrase_mapping,
        comp_fn
    ):
        super(PhraseEmbeddings, self).__init__()

        self.word_vec_size     = word_vec_size
        self.position_encoding = position_encoding
        self.dropout           = dropout
        self.word_padding_idx  = word_padding_idx
        self.word_vocab_size   = word_vocab_size
        self.phrase_mapping    = phrase_mapping
        self.comp_fn           = comp_fn

        self.embedding_size = word_vec_size

        phrase_vocab_size = len(phrase_mapping)
        self.phrase_vocab_size = phrase_vocab_size

        # this will make the backward disgusting, we might as well just
        # re-assign all phrase ids to pad_idx anyway.
        # at test time we can extend embedding.
        self.full_lut = nn.Embedding(word_vocab_size + phrase_vocab_size, word_vec_size, word_padding_idx)
        self.lut = nn.Embedding(word_vocab_size, word_vec_size, word_padding_idx)
        # weight sharing
        self.lut.weight.data = self.full_lut.weight.data[:word_vocab_size]
        # zero phrase embeddings
        self.full_lut.weight.data[word_vocab_size] = 0

        self.register_buffer("_buf", torch.LongTensor())


    def forward(self, input):
        """
        Take the phrase_mapping and update the respective unigram ids,
        then simply use the lookup tables.

        TODO(justinchiu): Only works for LSTM right now because of unpacking of h,c.
        input: the input tensor with unigram and phrase ids
        phrase_mapping: mapping from phrase_ids to unigram ids
        """

        comp_fn = self.comp_fn
        nhid = self.word_vec_size
        phrase_mapping = self.phrase_mapping

        # T x N x F=1
        in_length, in_batch, nfeat = input.size()

        input = input.squeeze(2)
        mask = input.ge(self.word_vocab_size)

        # Set phrases to pad_idx because they won't be used anyway
        words = input.clone()
        words[mask] = self.word_padding_idx

        # Gather phrases and get unique
        # Introduces a sync point.
        # I think masked indexing is done out of place as well
        phrases = input[mask].data.cpu()
        indices = sorted(set(phrases), key=lambda x: len(phrase_mapping[x]), reverse=True)
        lengths = list(map(lambda x: len(phrase_mapping[x]), indices))

        # Construct mapping to go from phrases back into index positions.
        revmap = {v: k for k, v in enumerate(indices)}
        pos = phrases.apply_(lambda x: revmap[x])
        if input.is_cuda:
            pos = pos.cuda()

        max_len = lengths[0]
        self._buf.resize_(max_len, len(indices)).fill_(0)
        for i, idx in enumerate(indices):
            self._buf[:lengths[i],i].copy_(torch.LongTensor(phrase_mapping[idx]))
        _, (h, c) = comp_fn(pack(self.lut(Variable(self._buf)), lengths))
        phrases = h.permute(1,0,2).contiguous().view(-1, nhid)

        words = self.lut(words)

        output = words \
            .view(-1, nhid) \
            .masked_scatter(mask.view(-1, 1), phrases[pos]) \
            .view(in_length, in_batch, nhid)

        return output

    # remove all this later
    @property
    def word_lut(self):
        return self.make_embedding[0][0]

    @property
    def emb_luts(self):
        return self.make_embedding[0]

    def load_pretrained_vectors(self, emb_file, fixed):
        if emb_file:
            pretrained = torch.load(emb_file)
            self.word_lut.weight.data.copy_(pretrained)
            if fixed:
                self.word_lut.weight.requires_grad = False
