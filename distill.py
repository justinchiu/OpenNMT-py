from __future__ import division

import argparse
import torch
import torch.nn as nn
from torch import cuda

import onmt
import onmt.Models
import onmt.ModelConstructor
import onmt.modules
from onmt.Utils import use_gpu
from onmt.Translator import Translator

CHECKPOINT = "/n/rush_lab/jc/onmt/models/baseline_acc_62.19_ppl_8.10_e13.pt"

def setup_argparse():
    parser = argparse.ArgumentParser(description="distill.py")
    #parser.add_argument("-p", "--purpose")
    return parser

def get_logits(model, data):
    pass

def load_ngrams(filename):
    pass

def main():
    parser = setup_argparse()
    opts = parser.parse_args()

    # hm, Translator will just reload this, lol.
    teacher_model = torch.load(CHECKPOINT)
    teacher_opts = teacher_model["opt"]
    teacher_opts.model = CHECKPOINT
    translator = Translator(teacher_opts)
    import pdb; pdb.set_trace()


if __name__ == "__main__":
    main()
