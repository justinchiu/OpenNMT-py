import torch
import numpy as n
import visdom

from os.path import join

"""
datadir = "/n/rush_lab/data/iwslt14-de-en/data-onmt"

wordworddata     = join(datadir, "iwslt14.tokenized.de-en.3-3")
phrasephrasedata = join(datadir, "iwslt14.tokenized.phrase.de-en.3-3.natural.natural.nodistill")
phraseworddata   = join(datadir, "iwslt14.tokenized.phrase.de-en.3-3.machine.word.nodistill")

wordworddatatrain     = torch.load(wordworddata     + ".train.pt")
phrasephrasedatatrain = torch.load(phrasephrasedata + ".train.pt")
phraseworddatatrain   = torch.load(phraseworddata   + ".train.pt"))
wordworddatavalid     = torch.load(wordworddata     + ".valid.pt")
phrasephrasedatavalid = torch.load(phrasephrasedata + ".valid.pt")
phraseworddatavalid   = torch.load(phraseworddata   + ".valid.pt"))

src_phrase_mappings = "/n/rush_lab/data/iwslt14-de-en/data/iwslt14.tokenized.phrase.de-en/phrase.src.pkl"
tgt_phrase_mappings = "/n/rush_lab/data/iwslt14-de-en/data/iwslt14.tokenized.phrase.de-en/phrase.natural.natural.tgt.pkl"
unigram_vocab = "/n/rush_lab/data/iwslt14-de-en/data-onmt/iwslt14.tokenized.de-en.3-3.vocab.pt"

modelstr = "/n/rush_lab/jc/onmt/models/{}/{}"
wordwordmodel = modelstr.format("baseline-brnn2", "baseline-brnn2_acc_62.80_ppl_7.72_e13.pt")
phrasephrasemodel = modelstr.format("phrase.cnatural.scnatural.nodistill", "phrase.cnatural.scnatural.nodistill.lr1.clip5_acc_40.81_ppl_49.96_e14.pt")
phrasewordmodel = modelstr.format("phrase.cnatural.word.nodistill", "phrase.cmachine.word.nodistill.lr1.clip5_acc_61.80_ppl_8.08_e18.pt")
phrasewordrepeatmodel = modelstr.format("phrase.cphrase.word.nodistill.repeat", "phrase.cphrase.word.nodistill.repeat.lr1.clip5_acc_62.25_ppl_7.94_e25.pt")
phrasewordembeddingmodel = modelstr.format("", "")

"""

vis = visdom.Visdom()


