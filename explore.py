import os
import torch
import onmt
import pickle

from collections import Counter

univ = torch.load("/n/rush_lab/data/iwslt14-de-en/data-onmt/iwslt14.tokenized.de-en.3-3.vocab.pt")
phrasev = torch.load("/n/rush_lab/data/iwslt14-de-en/data-onmt/iwslt14.tokenized.phrase.de-en.3-3.machine.machine.vocab.pt")
phrases_m_m = pickle.load(open("/n/rush_lab/data/iwslt14-de-en/data/iwslt14.tokenized.phrase.de-en/phrase.machine.machine.tgt.pkl", "rb"))

train = torch.load("/n/rush_lab/data/iwslt14-de-en/data-onmt/iwslt14.tokenized.phrase.de-en.3-3.machine.machine.train.pt")
valid = torch.load("/n/rush_lab/data/iwslt14-de-en/data-onmt/iwslt14.tokenized.phrase.de-en.3-3.machine.machine.valid.pt")

ufields = onmt.IO.load_fields(univ)
pfields = onmt.IO.load_fields(phrasev)


