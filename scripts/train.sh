ROOT=/n/holylfs/LABS/rush_lab/data/iwslt14-de-en
TEXT=${ROOT}/data/iwslt14.tokenized.de-en
BPE=${ROOT}/data/iwslt14.tokenized.bpe.de-en
PHRASE=${ROOT}/data/iwslt14.tokenized.phrase.de-en

BPEDATA=${ROOT}/data-onmt/iwslt14.tokenized.bpe.de-en
DATA=${ROOT}/data-onmt/iwslt14.tokenized.de-en.3-3
PHRASEDATA_MACHINE_MACHINE=${ROOT}/data-onmt/iwslt14.tokenized.phrase.de-en.3-3.machine.machine
PHRASEDATA_MACHINE_NATURAL=${ROOT}/data-onmt/iwslt14.tokenized.phrase.de-en.3-3.machine.natural
PHRASEDATA_NATURAL_NATURAL=${ROOT}/data-onmt/iwslt14.tokenized.phrase.de-en.3-3.natural.natural
DISTILLDATA=${ROOT}/data-onmt/iwslt14.tokenized.de-en.3-3.distill
MODEL=/n/holylfs/LABS/rush_lab/jc/onmt/models
LOG=/n/holylfs/LABS/rush_lab/jc/onmt/logs

GEN=/n/rush_lab/jc/onmt/gen

train_baseline() {
    python /n/home13/jchiu/python/OpenNMT-py/train.py \
	-data $DATA \
	-save_model $MODEL/baseline3 \
	-gpuid 0 \
	| tee ${LOG}/baseline3.log
}

train_baseline_brnn() {
    python /n/home13/jchiu/python/OpenNMT-py/train.py \
    -encoder_type brnn \
	-data $DATA \
	-save_model $MODEL/baseline.brnn \
	-gpuid 2 \
	| tee ${LOG}/baseline.brnn.log
}

train_bpe_baseline() {
    python /n/home13/jchiu/python/OpenNMT-py/train.py \
	-data $BPEDATA \
	-save_model $MODEL/baseline2bpe \
	-gpuid 1 \
	| tee ${LOG}/baseline2bpe.log
}

generate_baseline_train() {
    python /n/home13/jchiu/python/OpenNMT-py/translate.py \
    -model $MODEL/baseline2_acc_62.15_ppl_8.08_e13.pt \
    -gpu 0 \
    -src /n/holylfs/LABS/rush_lab/data/iwslt14-de-en/data/iwslt14.tokenized.phrase.de-en/train.de \
    -output /n/holylfs/LABS/rush_lab/data/iwslt14-de-en/data/iwslt14.tokenized.phrase.de-en/train.en.baseline2.out \
    -replace_unk \
    -beam_size 5 
}

generate_baseline_test() {
    python /n/home13/jchiu/python/OpenNMT-py/translate.py \
        -model $MODEL/baseline2_acc_62.15_ppl_8.08_e13.pt \
        -gpu 3 \
        -src /n/holylfs/LABS/rush_lab/data/iwslt14-de-en/data/iwslt14.tokenized.phrase.de-en/test.de \
        -output /n/rush_lab/jc/onmt/gen/baseline2.test.en \ 
        -replace_unk \
        -beam_size 5 
}

convert_to_phrases() {
    python /n/home13/jchiu/projects/bpephrase/python/ngrams.py
}

train_distill() {
    python /n/home13/jchiu/python/OpenNMT-py/train.py \
    -data $DISTILLDATA \
    -save_model $MODEL/distill.lr1.clip5 \
    -gpuid 3 \
    -learning_rate 1 \
    -max_grad_norm 5 \
    -epochs 25 \
    | tee ${LOG}/distill.lr1.clip5.log
}

train_phrase_machine_machine () {
    python /n/home13/jchiu/python/OpenNMT-py/train.py \
    -data $PHRASEDATA_MACHINE_MACHINE \
    -save_model $MODEL/phrase.machine.machine.lr1.clip5 \
    -gpuid 0 \
    -learning_rate 1 \
    -max_grad_norm 5 \
    -epochs 25 \
    | tee ${LOG}/phrase.machine.machine.lr1.clip5.log
}

train_phrase_machine_natural () {
    python /n/home13/jchiu/python/OpenNMT-py/train.py \
    -data $PHRASEDATA_MACHINE_NATURAL \
    -save_model $MODEL/phrase.machine.natural.lr1.clip5 \
    -gpuid 1 \
    -learning_rate 1 \
    -max_grad_norm 5 \
    -epochs 25 \
    | tee ${LOG}/phrase.machine.natural.lr1.clip2.5.log
}

train_phrase_natural_natural () {
    python /n/home13/jchiu/python/OpenNMT-py/train.py \
    -data $PHRASEDATA_NATURAL_NATURAL \
    -save_model $MODEL/phrase.natural.natural.lr1.clip5 \
    -gpuid 2 \
    -learning_rate 1 \
    -max_grad_norm 5 \
    -epochs 25 \
    | tee ${LOG}/phrase.natural.natural.lr1.clip2.5.log
}

generate_phrase_natural_natural() {
    python /n/home13/jchiu/python/OpenNMT-py/translate.py \
        -model /n/holylfs/LABS/rush_lab/jc/onmt/models/phrase.natural.natural.lr1.clip5_acc_34.30_ppl_200.52_e8.pt \
        -gpu 3 \
        -src /n/holylfs/LABS/rush_lab/data/iwslt14-de-en/data/iwslt14.tokenized.phrase.de-en/test.de \
        -output /n/rush_lab/jc/onmt/gen/phrase.natural.natural.test.en \
        -beam_size 5
}

dbg_train_comp_phrase () {
    python /n/home13/jchiu/python/OpenNMT-py/train.py \
        # data doesn't matter anyway
        -data $PHRASEDATA_NATURAL_NATURAL \
        -save_model /tmp/trash/dbg_model
        -gpuid 3 \
        -learning_rate 1 \
        -max_grad_norm 5 \
        -epochs 2 \
        | tee /tmp/trash/dbg.log
}
