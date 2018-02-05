import torch
from tensorboardX import SummaryWriter

import os

import visdom

valdir = "/n/rush_lab/jc/onmt/validation"

valid = torch.load("/n/rush_lab/data/iwslt14-de-en/data-onmt/iwslt14.tokenized.de-en.3-3.valid.pt")
phrasevalid = torch.load("/n/rush_lab/data/iwslt14-de-en/data-onmt/iwslt14.tokenized.phrase.de-en.3-3.machine.word.nodistill.valid.pt")

baselinename = "baseline-brnn2"
baselinedir = os.path.join(valdir, baselinename)
baselinenlls = torch.load(os.path.join(baselinedir, baselinename + ".nlls"))
baselinewordnlls = torch.load(os.path.join(baselinedir, baselinename + ".wordnlls"))
baselineattns = torch.load(os.path.join(baselinedir, baselinename + ".attns"))

modelname = "phrase.cnatural.word.nodistill"
modeldir = os.path.join(valdir, modelname)
modelnlls = torch.load(os.path.join(modeldir, modelname + ".nlls"))
modelwordnlls = torch.load(os.path.join(modeldir, modelname + ".wordnlls"))
modelattns = torch.load(os.path.join(modeldir, modelname + ".attns"))

worst_nlls, worst_idxs = modelnlls.sort(descending=True)

diffs, diff_idxs = (baselinenlls - modelnlls).abs().sort(descending=True)

vis.heatmap(
    modelattns[diff_idxs[1]].data.squeeze(1).cpu(),
    opts=dict(
        columnnames=list(phrasevalid[diff_idxs[1]].src),
        rownames=list(phrasevalid[diff_idxs[1]].tgt) + ["<eos>"]
    )
)
