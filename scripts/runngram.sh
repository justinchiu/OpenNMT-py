#!/bin/bash

/n/home13/jchiu/projects/bpephrase/bin/bpephrase \
    --train /n/rush_lab/data/iwslt14-de-en/data/iwslt14.tokenized.phrase.de-en/train.en \
    --valid /n/rush_lab/data/iwslt14-de-en/data/iwslt14.tokenized.phrase.de-en/valid.en \
    --test /n/rush_lab/data/iwslt14-de-en/data/iwslt14.tokenized.phrase.de-en/test.en \
    --vocab /n/rush_lab/data/iwslt14-de-en/data-onmt/vocab.tgt
/n/home13/jchiu/projects/bpephrase/bin/bpephrase \
    --train /n/rush_lab/data/iwslt14-de-en/data/iwslt14.tokenized.phrase.de-en/train.de \
    --valid /n/rush_lab/data/iwslt14-de-en/data/iwslt14.tokenized.phrase.de-en/valid.de \
    --test /n/rush_lab/data/iwslt14-de-en/data/iwslt14.tokenized.phrase.de-en/test.de \
    --vocab /n/rush_lab/data/iwslt14-de-en/data-onmt/vocab.src
