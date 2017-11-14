import torch
import codecs

vocabs = torch.load("/n/rush_lab/data/iwslt14-de-en/data-onmt/iwslt14.tokenized.de-en.vocab.pt")
outfile = "/n/rush_lab/data/iwslt14-de-en/data-onmt/vocab."
for vocab in vocabs:
    with codecs.open(outfile + vocab[0], "w") as f:
        for i, word in enumerate(vocab[1].itos):
            f.write("{}\t{}\n".format(i,word))
