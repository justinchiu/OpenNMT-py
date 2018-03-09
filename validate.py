import argparse
import onmt
import os
import torch

from onmt.Utils import add_phrases
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable as V

from typing import NamedTuple

import numpy as np

#from tensorboardX import SummaryWriter
import visdom

from tqdm import tqdm

def get_args():
    args = argparse.ArgumentParser()
    args.add_argument(
        "--data", type=str,
        default=None,
        required=True)
    args.add_argument(
        "--datawords", type=str,
        default=None)
    args.add_argument(
        "--checkpoint_path", type=str,
        default=None,
        required=True)
    args.add_argument(
        '--src_phrase_mappings', type=str,
        default=None,
        help="""Use embeddings for phrases,
        given the phrase mapping path.""")
    args.add_argument(
        '--tgt_phrase_mappings', type=str,
        default=None,
        help="""Use embeddings for phrases,
        given the phrase mapping path.""")
    args.add_argument(
        '--unigram_vocab', type=str,
        default=None,
        help="""Use embeddings for phrases,
        given the phrase mapping path.""")
    args.add_argument("--savepath", type=str, default=None)
    args.add_argument("--modelname", type=str, default=None, required=True)
    args.add_argument("--devid", type=int, default=-1)
    args.add_argument("--worstn", type=int, default=10)
    return args.parse_args()


args = get_args()

devid = args.devid

train = torch.load(args.data + '.train.pt')
valid = torch.load(args.data + '.valid.pt')
if args.datawords is not None:
    trainwords = torch.load(args.datawords + ".train.pt")
    validwords = torch.load(args.datawords + ".valid.pt")

fields = onmt.IO.load_fields(
    torch.load(args.data + '.vocab.pt'))
# Why do we have to filter fields? Seems hacky.
fields = dict([(k, f) for (k, f) in fields.items()
    if k in train.examples[0].__dict__])
train.fields = fields
valid.fields = fields

# I guess mutating the vocabulary is as well.
if args.unigram_vocab:
    unigram_vocab = torch.load(args.unigram_vocab)
if args.src_phrase_mappings:
    fields["src"].vocab = add_phrases(unigram_vocab[0][1], args.src_phrase_mappings)
if args.tgt_phrase_mappings:
    fields["tgt"].vocab = add_phrases(unigram_vocab[1][1], args.tgt_phrase_mappings)

"""
with SummaryWriter(comment="validation") as w:
    w.add_scalar("loss", bloss)
    w.add_image("attn", output[1]["std"].data.squeeze(1))
"""
savepath = os.path.join(args.savepath, args.modelname)
nllpath = savepath + ".nlls"
attnpath = savepath + ".attns"
wordpath = savepath + ".wordnlls"

checkpoint = torch.load(
    args.checkpoint_path,
    map_location=lambda storage, loc: storage)
model_opt = checkpoint['opt']
model = onmt.ModelConstructor.make_base_model(model_opt, fields, devid, checkpoint)

nlls = None
attns = None
wordnlls = None
if os.path.isfile(nllpath) and os.path.isfile(attnpath) and os.path.isfile(wordpath):
    nlls = torch.load(nllpath)
    attns = torch.load(attnpath)
    wordnlls = torch.load(wordpath)
else:
    srcfield = fields["src"]
    tgtfield = fields["tgt"]
    # Sentence, NLL pairs, sorted decreasing (worst sentences first)
    # And attention scores
    nlls = torch.FloatTensor(len(valid))
    attns = []
    wordnlls = []
    for i, example in tqdm(enumerate(valid)):
        # 0.2.1, use field.process([example])
        x = srcfield.numericalize(srcfield.pad([example.src]), device=devid, train=False)
        y = tgtfield.numericalize(tgtfield.pad([example.tgt]), device=devid, train=False)

        #with torch.no_grad():
        if True:
            if hasattr(model.generator[0], "reset_perm"):
                model.generator[0].reset_perm()
                cy = model.generator[0].collapse_target(y)
            else:
                cy = y
            if hasattr(model, "ctxt_fn") and model.ctxt_fn is not None:
                # Singleton list because all functions expect batching
                rles = [[ word.count("_")+1 for word in example.src ]]
                model.ctxt_fn.rle_to_idxs(rles)
                if hasattr(model.ctxt_fn, "lut") and model.ctxt_fn.lut is not None:
                    model.ctxt_fn.dataset = validwords
                    model.ctxt_fn.get_words(V(torch.LongTensor([i])))
                if hasattr(model.ctxt_fn, "poslut") and model.ctxt_fn.poslut is not None:
                    # For the positions we just need the lengths of the source sentences
                    model.ctxt_fn.dataset = validwords
                    model.ctxt_fn.get_positions(V(torch.LongTensor([i])))
            output, attn_dict, decoderstate = model(x[0].view(-1, 1, 1), cy.view(-1, 1, 1), x[1])
            attn = attn_dict["std"]

            lsm = model.generator(output.squeeze(1))

            bloss = F.nll_loss(lsm, cy.view(-1)[1:], reduce=False)

            nlls[i] = bloss.mean().data[0]
            attns.append(attn)
            wordnlls.append(bloss)

    torch.save(nlls, nllpath)
    torch.save(attns, attnpath)
    torch.save(wordnlls, wordpath)

# Don't use this for now, since it requires having the class in namespace during loading.
class BadExample(NamedTuple):
    nll: float
    idx: int
    attn: torch.FloatTensor
    wordnll: torch.FloatTensor
    src: tuple
    tgt: tuple

def bad_examples():
    nlls_sorted, idxs = nlls.sort(descending=True)
    bad_examples = []
    for i in range(args.worstn):
        nll = nlls_sorted[i]
        idx = idxs[i]

        attn = attns[idx]
        wordnll = wordnlls[idx]
        example = valid[idx]

        bad_examples.append((
            nll,
            idx,
            attn,
            wordnll,
            example.src,
            example.tgt
        ))

    torch.save(bad_examples, savepath + ".bad_examples")

#bad_examples()
#with SummaryWriter(comment=args.modelname + ".validation"):
    #
    #
def get_ngram_stats():
    unigrams = [id for token, id in fields['src'].vocab.stoi.items() if len(token.split("_")) == 1]
    bigrams = [id for token, id in fields['src'].vocab.stoi.items() if len(token.split("_")) == 2]
    trigrams = [id for token, id in fields['src'].vocab.stoi.items() if len(token.split("_")) == 3]
    fourgrams = [id for token, id in fields['src'].vocab.stoi.items() if len(token.split("_")) == 4]
    unigramnorms = [model.encoder.embeddings(V(torch.LongTensor([id]).view(1,1,1).cuda())).norm() for id in unigrams]
    bigramnorms = [model.encoder.embeddings(V(torch.LongTensor([id]).view(1,1,1).cuda())).norm() for id in bigrams]
    trigramnorms = [model.encoder.embeddings(V(torch.LongTensor([id]).view(1,1,1).cuda())).norm() for id in trigrams]
    fourgramnorms = [model.encoder.embeddings(V(torch.LongTensor([id]).view(1,1,1).cuda())).norm() for id in fourgrams]

    unigramstats = (max(unigramnorms).data[0], sum(unigramnorms).data[0] / len(unigramnorms), min(unigramnorms).data[0])
    bigramstats = (max(bigramnorms).data[0], sum(bigramnorms).data[0] / len(bigramnorms), min(bigramnorms).data[0])
    trigramstats = (max(trigramnorms).data[0], sum(trigramnorms).data[0] / len(trigramnorms), min(trigramnorms).data[0])
    fourgramstats = (max(fourgramnorms).data[0], sum(fourgramnorms).data[0] / len(fourgramnorms), min(fourgramnorms).data[0])

    return unigramstats, bigramstats, trigramstats, fourgramstats
#print(get_ngram_stats())

def visualize_attn():
    import visdom
    vis = visdom.Visdom()
    #for i in np.random.permutation(len(valid))[:5]:
    for i in [3105, 2776, 2424, 2357, 1832]:
        example = valid[i]
        if args.datawords is not None:
            example = validwords[i]
        attn = attns[i]

        # unused
        nll = nlls[i]
        wordnll = wordnlls[i]

        rownames = list(example.tgt) + ["<eos>"]
        rownames = ["[{}] {} ({:.2f})".format(i, name, wordnll[i].data[0]) for i, name in enumerate(rownames)]
        columnnames = list(example.src)
        columnnames = ["[{}] {}".format(i, name) for i, name in enumerate(columnnames)]
        title = "Model {} Example {}".format(args.checkpoint_path.split("/")[-2], i)
        vis.heatmap(
            X=attn.data.cpu().squeeze(),
            opts=dict(
                rownames=rownames,
                columnnames=columnnames,
                colormap="Hot",
                title=title,
                width=750,
                height=750,
                marginleft=150,
                marginright=150,
                margintop=150,
                marginbottom=150
            ),
            win=title
        )


#visualize_attn()
idx = 2357
example = valid[idx]
srcfield = fields["src"]
tgtfield = fields["tgt"]
x = srcfield.numericalize(srcfield.pad([example.src]), device=devid, train=False)
y = tgtfield.numericalize(tgtfield.pad([example.tgt]), device=devid, train=False)
import pdb; pdb.set_trace()
