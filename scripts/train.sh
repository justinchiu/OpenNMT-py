ROOT=/n/holylfs/LABS/rush_lab/data/iwslt14-de-en
TEXT=${ROOT}/data/iwslt14.tokenized.de-en
BPE=${ROOT}/data/iwslt14.tokenized.bpe.de-en
PHRASE=${ROOT}/data/iwslt14.tokenized.phrase.de-en

BPEDATA=${ROOT}/data-onmt/iwslt14.tokenized.bpe.de-en
DATA=${ROOT}/data-onmt/iwslt14.tokenized.de-en.3-3

PHRASEDATA_MACHINE_MACHINE=${ROOT}/data-onmt/iwslt14.tokenized.phrase.de-en.3-3.machine.machine
PHRASEDATA_MACHINE_MACHINE_NODISTILL=${ROOT}/data-onmt/iwslt14.tokenized.phrase.de-en.3-3.machine.machine.nodistill
PHRASEDATA_MACHINE_NATURAL=${ROOT}/data-onmt/iwslt14.tokenized.phrase.de-en.3-3.machine.natural
PHRASEDATA_NATURAL_NATURAL=${ROOT}/data-onmt/iwslt14.tokenized.phrase.de-en.3-3.natural.natural
PHRASEDATA_NATURAL_NATURAL_NODISTILL=${ROOT}/data-onmt/iwslt14.tokenized.phrase.de-en.3-3.natural.natural.nodistill
PHRASEDATA_MACHINE_WORD=${ROOT}/data-onmt/iwslt14.tokenized.phrase.de-en.3-3.machine.word
PHRASEDATA_MACHINE_WORD_NODISTILL=${ROOT}/data-onmt/iwslt14.tokenized.phrase.de-en.3-3.machine.word.nodistill
PHRASEDATA_NATURAL_WORD=${ROOT}/data-onmt/iwslt14.tokenized.phrase.de-en.3-3.natural.word
PHRASEDATA_NATURAL_WORD_NODISTILL=${ROOT}/data-onmt/iwslt14.tokenized.phrase.de-en.3-3.natural.word.nodistill
PHRASEDATA_WORD_NATURAL=${ROOT}/data-onmt/iwslt14.tokenized.phrase.de-en.3-3.word.natural
PHRASEDATA_WORD_NATURAL_NODISTILL=${ROOT}/data-onmt/iwslt14.tokenized.phrase.de-en.3-3.word.natural.nodistill

DISTILLDATA=${ROOT}/data-onmt/iwslt14.tokenized.de-en.3-3.distill

PHRASEDATA_PHRASE_WORD_NODISTILL_REPEAT=${ROOT}/data-onmt/iwslt14.tokenized.phrase.de-en.3-3.machine.word.nodistill.repeat
PHRASEDATA_PHRASE_NATURAL_NODISTILL_REPEAT=${ROOT}/data-onmt/iwslt14.tokenized.phrase.de-en.3-3.natural.natural.nodistill.repeat

MODEL=/n/holylfs/LABS/rush_lab/jc/onmt/models
LOG=/n/holylfs/LABS/rush_lab/jc/onmt/logs

GEN=/n/rush_lab/jc/onmt/gen

train_baseline() {
    name=baseline3
    mkdir -p $MODEL/$name
    python /n/home13/jchiu/projects/OpenNMT-py/train.py \
    	-data $DATA \
    	-save_model $MODEL/$name/$name \
    	-gpuid 1 \
    	| tee ${LOG}/$name.log
}

train_baseline_brnn() {
    name=baseline-brnn2
    mkdir -p $MODEL/$name
    python /n/home13/jchiu/projects/OpenNMT-py/train.py \
        -encoder_type brnn \
    	-data $DATA \
    	-save_model $MODEL/$name/$name \
    	-gpuid 3 \
    	| tee ${LOG}/$name.log
}

train_bpe_baseline() {
    name=baseline2bpe
    mkdir -p $MODEL/$name
    python /n/home13/jchiu/projects/OpenNMT-py/train.py \
    	-data $BPEDATA \
    	-save_model $MODEL/$name/$name \
    	-gpuid 1 \
    	| tee ${LOG}/$name.log
}

generate_baseline_train() {
    python /n/home13/jchiu/projects/OpenNMT-py/translate.py \
        -model $MODEL/baseline3/baseline3_acc_61.84_ppl_8.11_e13.pt \
        -gpu 1 \
        -src /n/holylfs/LABS/rush_lab/data/iwslt14-de-en/data/iwslt14.tokenized.phrase.de-en/train.de \
        -output /n/holylfs/LABS/rush_lab/data/iwslt14-de-en/data/iwslt14.tokenized.phrase.de-en/train.en.baseline3.out \
        -replace_unk \
        -beam_size 5 
}

generate_baseline_brnn_train() {
    python /n/home13/jchiu/projects/OpenNMT-py/translate.py \
        -model $MODEL/baseline-brnn2/baseline-brnn2_acc_62.80_ppl_7.72_e13.pt \
        -gpu 3 \
        -src /n/holylfs/LABS/rush_lab/data/iwslt14-de-en/data/iwslt14.tokenized.phrase.de-en/train.de \
        -output /n/holylfs/LABS/rush_lab/data/iwslt14-de-en/data/iwslt14.tokenized.phrase.de-en/train.en.baseline.brnn.out \
        -replace_unk \
        -beam_size 5 
}

generate_baseline_test() {
    python /n/home13/jchiu/projects/OpenNMT-py/translate.py \
        -model $MODEL/baseline2/baseline2_acc_62.15_ppl_8.08_e13.pt \
        -gpu 3 \
        -src /n/holylfs/LABS/rush_lab/data/iwslt14-de-en/data/iwslt14.tokenized.phrase.de-en/test.de \
        -output /n/rush_lab/jc/onmt/gen/baseline2.test.en \ 
        -replace_unk \
        -beam_size 5 
}

generate_baseline_brnn_test() {
    python /n/home13/jchiu/projects/OpenNMT-py/translate.py \
        -model $MODEL/baseline-brnn2/baseline-brnn2_acc_62.80_ppl_7.72_e13.pt \
        -gpu 3 \
        -src /n/holylfs/LABS/rush_lab/data/iwslt14-de-en/data/iwslt14.tokenized.phrase.de-en/test.de \
        -output /n/rush_lab/jc/onmt/gen/baseline.brnn.test.en \
        -replace_unk \
        -beam_size 5 
}

convert_to_phrases() {
    python /n/home13/jchiu/projects/bpephrase/python/ngrams.py
}

train_distill() {
    name=distill
    mkdir -p $MODEL/$name
    python /n/home13/jchiu/projects/OpenNMT-py/train.py \
        -encoder_type brnn \
        -data $DISTILLDATA \
        -save_model $MODEL/$name/$name.lr1.clip5 \
        -gpuid 2 \
        -learning_rate 1 \
        -max_grad_norm 5 \
        -epochs 25 \
        | tee ${LOG}/$name.lr1.clip5.log
}

generate_distill_test() {
    python /n/home13/jchiu/projects/OpenNMT-py/translate.py \
        -model /n/rush_lab/jc/onmt/models/distill/distill.lr1.clip5_acc_59.77_ppl_24.55_e6.pt \
        -gpu 2 \
        -src /n/holylfs/LABS/rush_lab/data/iwslt14-de-en/data/iwslt14.tokenized.phrase.de-en/test.de \
        -output /n/rush_lab/jc/onmt/gen/distill.test.en \
        -beam_size 5
}

train_phrase_machine_machine () {
    name=phrase.machine.machine
    mkdir -p $MODEL/$name
    python /n/home13/jchiu/projects/OpenNMT-py/train.py \
        -encoder_type brnn \
        -data $PHRASEDATA_MACHINE_MACHINE \
        -save_model $MODEL/$name/$name.lr1.clip5 \
        -gpuid 3 \
        -learning_rate 1 \
        -max_grad_norm 5 \
        -epochs 25 \
        | tee ${LOG}/$name.lr1.clip5.log
}

train_phrase_machine_natural () {
    name=phrase.machine.natural
    mkdir -p $MODEL/$name
    python /n/home13/jchiu/projects/OpenNMT-py/train.py \
        -encoder_type brnn \
        -data $PHRASEDATA_MACHINE_NATURAL \
        -save_model $MODEL/$name/$name.lr1.clip5 \
        -gpuid 1 \
        -learning_rate 1 \
        -max_grad_norm 5 \
        -epochs 25 \
        | tee ${LOG}/$name.lr1.clip5.log
}

train_phrase_natural_natural () {
    name=phrase.natural.natural
    mkdir -p $MODEL/$name
    python /n/home13/jchiu/projects/OpenNMT-py/train.py \
        -encoder_type brnn \
        -data $PHRASEDATA_NATURAL_NATURAL \
        -save_model $MODEL/$name/$name.lr1.clip5 \
        -gpuid 2 \
        -learning_rate 1 \
        -max_grad_norm 5 \
        -epochs 25 \
        | tee ${LOG}/$name.lr1.clip5.log
}

generate_phrase_natural_natural() {
    python /n/home13/jchiu/projects/OpenNMT-py/translate.py \
        -model /n/rush_lab/jc/onmt/models/phrase.natural.natural/phrase.natural.natural.lr1.clip5_acc_37.95_ppl_265.78_e9.pt \
        -gpu 0 \
        -src /n/holylfs/LABS/rush_lab/data/iwslt14-de-en/data/iwslt14.tokenized.phrase.de-en/test.phrase.de \
        -output /n/rush_lab/jc/onmt/gen/phrase.natural.natural.test.en \
        -beam_size 5
}

generate_phrase_machine_machine() {
    python /n/home13/jchiu/projects/OpenNMT-py/translate.py \
        -model /n/rush_lab/jc/onmt/models/phrase.machine.machine/phrase.machine.machine.lr1.clip5_acc_33.27_ppl_333.15_e8.pt \
        -gpu 1 \
        -src /n/holylfs/LABS/rush_lab/data/iwslt14-de-en/data/iwslt14.tokenized.phrase.de-en/test.phrase.de \
        -output /n/rush_lab/jc/onmt/gen/phrase.machine.machine.test.en \
        -beam_size 5
}

train_phrase_machine_word() {
    name=phrase.machine.word
    mkdir -p $MODEL/$name
    python /n/home13/jchiu/projects/OpenNMT-py/train.py \
        -encoder_type brnn \
        -data $PHRASEDATA_MACHINE_WORD \
        -save_model $MODEL/$name/$name.lr1.clip5 \
        -gpuid 3 \
        -learning_rate 1 \
        -max_grad_norm 5 \
        -epochs 25 \
        | tee ${LOG}/$name.lr1.clip5.log
}

train_phrase_machine_word_nodistill() {
    name=phrase.machine.word.nodistill
    mkdir -p $MODEL/$name
    python /n/home13/jchiu/projects/OpenNMT-py/train.py \
        -encoder_type brnn \
        -data $PHRASEDATA_MACHINE_WORD_NODISTILL \
        -save_model $MODEL/$name/$name.lr1.clip5 \
        -gpuid 1 \
        -learning_rate 1 \
        -max_grad_norm 5 \
        -epochs 25 \
        | tee ${LOG}/$name.lr1.clip5.log
}

generate_phrase_machine_word() {
    python /n/home13/jchiu/projects/OpenNMT-py/translate.py \
        -model /n/rush_lab/jc/onmt/models/phrase.machine.word/phrase.machine.word.lr1.clip5_acc_57.65_ppl_23.17_e6.pt \
        -gpu 2 \
        -src /n/holylfs/LABS/rush_lab/data/iwslt14-de-en/data/iwslt14.tokenized.phrase.de-en/test.phrase.de \
        -output /n/rush_lab/jc/onmt/gen/phrase.machine.word.test.en \
        -beam_size 5
}

generate_phrase_machine_word_nodistill() {
    python /n/home13/jchiu/projects/OpenNMT-py/translate.py \
        -model /n/rush_lab/jc/onmt/models/phrase.machine.word.nodistill/phrase.machine.word.nodistill.lr1.clip5_acc_60.00_ppl_9.01_e14.pt \
        -gpu 0 \
        -src /n/holylfs/LABS/rush_lab/data/iwslt14-de-en/data/iwslt14.tokenized.phrase.de-en/test.phrase.de \
        -output /n/rush_lab/jc/onmt/gen/phrase.machine.word.nodistill.test.en \
        -beam_size 5
}

# note that the source distribution is ALWAYS the same, so this is fine
train_phrase_cmachine_word() {
    name=phrase.cmachine.word
    python /n/home13/jchiu/projects/OpenNMT-py/train.py \
        -encoder_type brnn \
        -data $PHRASEDATA_MACHINE_WORD \
        -src_phrase_mappings /n/rush_lab/data/iwslt14-de-en/data/iwslt14.tokenized.phrase.de-en/phrase.src.pkl \
        -unigram_vocab /n/rush_lab/data/iwslt14-de-en/data-onmt/iwslt14.tokenized.de-en.3-3.vocab.pt \
        -save_model $MODEL/$name/$name.lr1.clip5 \
        -gpuid 1 \
        -learning_rate 1 \
        -max_grad_norm 5 \
        -epochs 25 \
        | tee ${LOG}/$name.lr1.clip5.log
}

generate_phrase_cmachine_word() {
    python /n/home13/jchiu/projects/OpenNMT-py/translate.py \
        -model /n/rush_lab/jc/onmt/models/phrase.cmachine.word/phrase.cmachine.word.lr1.clip5_acc_58.83_ppl_25.27_e6.pt \
        -gpu 3 \
        -src /n/holylfs/LABS/rush_lab/data/iwslt14-de-en/data/iwslt14.tokenized.phrase.de-en/test.de \
        -output /n/rush_lab/jc/onmt/gen/phrase.cmachine.word.test.en \
        -beam_size 5
}

train_phrase_cmachine_word_nodistill() {
    name=phrase.cmachine.word.nodistill
    python /n/home13/jchiu/projects/OpenNMT-py/train.py \
        -encoder_type brnn \
        -data $PHRASEDATA_MACHINE_WORD_NODISTILL \
        -src_phrase_mappings /n/rush_lab/data/iwslt14-de-en/data/iwslt14.tokenized.phrase.de-en/phrase.src.pkl \
        -unigram_vocab /n/rush_lab/data/iwslt14-de-en/data-onmt/iwslt14.tokenized.de-en.3-3.vocab.pt \
        -save_model $MODEL/$name/$name.lr1.clip5 \
        -gpuid 2 \
        -learning_rate 1 \
        -max_grad_norm 5 \
        -epochs 25 \
        | tee ${LOG}/$name.lr1.clip5.log
}

generate_phrase_cmachine_word() {
    python /n/home13/jchiu/projects/OpenNMT-py/translate.py \
        -model /n/rush_lab/jc/onmt/models/phrase.cmachine.word/phrase.cmachine.word.lr1.clip5_acc_58.83_ppl_25.27_e6.pt \
        -gpu 3 \
        -src /n/holylfs/LABS/rush_lab/data/iwslt14-de-en/data/iwslt14.tokenized.phrase.de-en/test.de \
        -output /n/rush_lab/jc/onmt/gen/phrase.cmachine.word.test.en \
        -beam_size 5
}

generate_phrase_cmachine_word_nodistill() {
    python /n/home13/jchiu/projects/OpenNMT-py/translate.py \
        -model /n/rush_lab/jc/onmt/models/phrase.cmachine.word.nodistill/phrase.cmachine.word.nodistill.lr1.clip5_acc_61.80_ppl_8.08_e18.pt \
        -gpu 2 \
        -src /n/holylfs/LABS/rush_lab/data/iwslt14-de-en/data/iwslt14.tokenized.phrase.de-en/test.de \
        -output /n/rush_lab/jc/onmt/gen/phrase.cmachine.word.nodistill.test.en \
        -beam_size 5
}

train_phrase_cmachine_scmachine() {
    name=phrase.cmachine.scmachine
    python /n/home13/jchiu/projects/OpenNMT-py/train.py \
        -encoder_type brnn \
        -data $PHRASEDATA_MACHINE_MACHINE \
        -src_phrase_mappings /n/rush_lab/data/iwslt14-de-en/data/iwslt14.tokenized.phrase.de-en/phrase.src.pkl \
        -tgt_phrase_mappings /n/rush_lab/data/iwslt14-de-en/data/iwslt14.tokenized.phrase.de-en/phrase.machine.machine.tgt.pkl \
        -unigram_vocab /n/rush_lab/data/iwslt14-de-en/data-onmt/iwslt14.tokenized.de-en.3-3.vocab.pt \
        -tgt_distractors 2048 \
        -save_model $MODEL/$name/$name.lr1.clip5 \
        -gpuid 3 \
        -learning_rate 1 \
        -max_grad_norm 5 \
        -epochs 25 \
        | tee ${LOG}/$name.lr1.clip5.log
}

train_phrase_cmachine_scmachine_nodistill() {
    name=phrase.cmachine.scmachine.nodistill
    python /n/home13/jchiu/projects/OpenNMT-py/train.py \
        -encoder_type brnn \
        -data $PHRASEDATA_MACHINE_MACHINE_NODISTILL \
        -src_phrase_mappings /n/rush_lab/data/iwslt14-de-en/data/iwslt14.tokenized.phrase.de-en/phrase.src.pkl \
        -tgt_phrase_mappings /n/rush_lab/data/iwslt14-de-en/data/iwslt14.tokenized.phrase.de-en/phrase.machine.machine.tgt.pkl \
        -unigram_vocab /n/rush_lab/data/iwslt14-de-en/data-onmt/iwslt14.tokenized.de-en.3-3.vocab.pt \
        -tgt_distractors 2048 \
        -save_model $MODEL/$name/$name.lr1.clip5 \
        -gpuid 3 \
        -learning_rate 1 \
        -max_grad_norm 5 \
        -epochs 25 \
        | tee ${LOG}/$name.lr1.clip5.log
}

generate_phrase_cmachine_scmachine_nodistill() {
    python /n/home13/jchiu/projects/OpenNMT-py/translate.py \
        -model /n/rush_lab/jc/onmt/models/phrase.cmachine.scmachine.nodistill/phrase.cmachine.scmachine.nodistill.lr1.clip5_acc_40.63_ppl_47.38_e25.pt \
        -gpu 1 \
        -src /n/holylfs/LABS/rush_lab/data/iwslt14-de-en/data/iwslt14.tokenized.phrase.de-en/test.de \
        -output /n/rush_lab/jc/onmt/gen/phrase.cmachine.scmachine.nodistill.test.en \
        -beam_size 5
}

train_phrase_word_scnatural_nodistill() {
    name=phrase.word.scnatural.nodistill
    python /n/home13/jchiu/projects/OpenNMT-py/train.py \
        -encoder_type brnn \
        -data $PHRASEDATA_WORD_NATURAL_NODISTILL \
        -tgt_phrase_mappings /n/rush_lab/data/iwslt14-de-en/data/iwslt14.tokenized.phrase.de-en/phrase.natural.natural.tgt.pkl \
        -unigram_vocab /n/rush_lab/data/iwslt14-de-en/data-onmt/iwslt14.tokenized.de-en.3-3.vocab.pt \
        -tgt_distractors 2048 \
        -save_model $MODEL/$name/$name.lr1.clip5 \
        -gpuid 2 \
        -learning_rate 1 \
        -max_grad_norm 5 \
        -epochs 25 \
        | tee ${LOG}/$name.lr1.clip5.log
}

train_phrase_cnatural_scnatural_nodistill() {
    name=phrase.cnatural.scnatural.nodistill
    python /n/home13/jchiu/projects/OpenNMT-py/train.py \
        -encoder_type brnn \
        -data $PHRASEDATA_NATURAL_NATURAL_NODISTILL \
        -src_phrase_mappings /n/rush_lab/data/iwslt14-de-en/data/iwslt14.tokenized.phrase.de-en/phrase.src.pkl \
        -tgt_phrase_mappings /n/rush_lab/data/iwslt14-de-en/data/iwslt14.tokenized.phrase.de-en/phrase.natural.natural.tgt.pkl \
        -unigram_vocab /n/rush_lab/data/iwslt14-de-en/data-onmt/iwslt14.tokenized.de-en.3-3.vocab.pt \
        -tgt_distractors 2048 \
        -save_model $MODEL/$name/$name.lr1.clip5 \
        -gpuid 3 \
        -learning_rate 1 \
        -max_grad_norm 5 \
        -epochs 25 \
        | tee ${LOG}/$name.lr1.clip5.log
}

dbg_train_phrase_cnatural_scnatural_nodistill() {
    name=phrase.cnatural.scnatural.nodistill
    python /n/home13/jchiu/projects/OpenNMT-py/train.py \
        -encoder_type brnn \
        -data $PHRASEDATA_NATURAL_NATURAL_NODISTILL \
        -src_phrase_mappings /n/rush_lab/data/iwslt14-de-en/data/iwslt14.tokenized.phrase.de-en/phrase.src.pkl \
        -tgt_phrase_mappings /n/rush_lab/data/iwslt14-de-en/data/iwslt14.tokenized.phrase.de-en/phrase.natural.natural.tgt.pkl \
        -unigram_vocab /n/rush_lab/data/iwslt14-de-en/data-onmt/iwslt14.tokenized.de-en.3-3.vocab.pt \
        -tgt_distractors 2048 \
        -save_model /tmp/trash \
        -gpuid 3 \
        -learning_rate 1 \
        -max_grad_norm 5 \
        -epochs 25 
}

generate_phrase_cnatural_scnatural_nodistill() {
    python /n/home13/jchiu/projects/OpenNMT-py/translate.py \
        -model /n/rush_lab/jc/onmt/models/phrase.cnatural.scnatural.nodistill/phrase.cnatural.scnatural.nodistill.lr1.clip5_acc_40.81_ppl_49.96_e14.pt \
        -gpu 0 \
        -src /n/holylfs/LABS/rush_lab/data/iwslt14-de-en/data/iwslt14.tokenized.phrase.de-en/test.phrase.de \
        -output /n/rush_lab/jc/onmt/gen/phrase.cnatural.scnatural.nodistill.test.en \
        -beam_size 5
}

generate_phrase_word_scnatural_nodistill() {
    python /n/home13/jchiu/projects/OpenNMT-py/translate.py \
        -model /n/rush_lab/jc/onmt/models/phrase.word.scnatural.nodistill/phrase.word.scnatural.nodistill.lr1.clip5_acc_42.06_ppl_44.69_e14.pt \
        -gpu 1 \
        -src /n/holylfs/LABS/rush_lab/data/iwslt14-de-en/data/iwslt14.tokenized.phrase.de-en/test.de \
        -output /n/rush_lab/jc/onmt/gen/phrase.word.scnatural.nodistill.test.en \
        -beam_size 5
}

train_phrase_word_natural_nodistill() {
    #name=phrase.word.natural.nodistill
    name=phrase.word.natural.nodistill2
    python /n/home13/jchiu/projects/OpenNMT-py/train.py \
        -encoder_type brnn \
        -data $PHRASEDATA_WORD_NATURAL_NODISTILL \
        -save_model $MODEL/$name/$name.lr1.clip5 \
        -gpuid 0 \
        -learning_rate 1 \
        -max_grad_norm 5 \
        -epochs 25 \
        | tee ${LOG}/$name.lr1.clip5.log
}

generate_phrase_word_natural_nodistill() {
    #-model /n/rush_lab/jc/onmt/models/phrase.word.natural.nodistill/phrase.word.natural.nodistill.lr1.clip5_acc_42.02_ppl_47.73_e13.pt \
    python /n/home13/jchiu/projects/OpenNMT-py/translate.py \
        -model /n/rush_lab/jc/onmt/models/phrase.word.natural.nodistill2/phrase.word.natural.nodistill2.lr1.clip5_acc_41.84_ppl_48.18_e13.pt \
        -gpu 1 \
        -src /n/holylfs/LABS/rush_lab/data/iwslt14-de-en/data/iwslt14.tokenized.phrase.de-en/test.de \
        -output /n/rush_lab/jc/onmt/gen/phrase.word.natural.nodistill2.test.en \
        -beam_size 5
}

# Repeat Corpus

train_phrase_phrase_word_nodistill_repeat() {
    name=phrase.phrase.word.nodistill.repeat
    mkdir -p $MODEL/$name
    python /n/home13/jchiu/projects/OpenNMT-py/train.py \
        -encoder_type brnn \
        -data $PHRASEDATA_PHRASE_WORD_NODISTILL_REPEAT \
        -save_model $MODEL/$name/$name.lr1.clip5 \
        -gpuid 3 \
        -learning_rate 1 \
        -max_grad_norm 5 \
        -epochs 25 \
        | tee ${LOG}/$name.lr1.clip5.log
}

generate_phrase_phrase_word_nodistill_repeat() {
    name=phrase.phrase.word.nodistill.repeat
    python /n/home13/jchiu/projects/OpenNMT-py/translate.py \
        -model /n/rush_lab/jc/onmt/models/$name/phrase.phrase.word.nodistill.repeat.lr1.clip5_acc_60.66_ppl_8.83_e25.pt \
        -gpu 2 \
        -src /n/holylfs/LABS/rush_lab/data/iwslt14-de-en/data/iwslt14.tokenized.phrase.de-en/test.phrase.repeat.de \
        -output /n/rush_lab/jc/onmt/gen/$name.test.en \
        -beam_size 5
}

train_phrase_cphrase_word_nodistill_repeat() {
    name=phrase.cphrase.word.nodistill.repeat
    python /n/home13/jchiu/projects/OpenNMT-py/train.py \
        -encoder_type brnn \
        -data $PHRASEDATA_PHRASE_WORD_NODISTILL_REPEAT \
        -src_phrase_mappings /n/rush_lab/data/iwslt14-de-en/data/iwslt14.tokenized.phrase.de-en/phrase.src.pkl \
        -unigram_vocab /n/rush_lab/data/iwslt14-de-en/data-onmt/iwslt14.tokenized.de-en.3-3.vocab.pt \
        -save_model $MODEL/$name/$name.lr1.clip5 \
        -gpuid 2 \
        -learning_rate 1 \
        -max_grad_norm 5 \
        -epochs 25 \
        | tee ${LOG}/$name.lr1.clip5.log
}

generate_phrase_cphrase_word_nodistill_repeat() {
    name=phrase.cphrase.word.nodistill.repeat
    python /n/home13/jchiu/projects/OpenNMT-py/translate.py \
        -model /n/rush_lab/jc/onmt/models/$name/phrase.cphrase.word.nodistill.repeat.lr1.clip5_acc_62.25_ppl_7.94_e25.pt \
        -gpu 3 \
        -src /n/holylfs/LABS/rush_lab/data/iwslt14-de-en/data/iwslt14.tokenized.phrase.de-en/test.phrase.repeat.de \
        -output /n/rush_lab/jc/onmt/gen/$name.test.en \
        -beam_size 5
}

# scphrase on the source side means scaled compositional phrases
# differs from target side, which is sampled compositional phrases
train_phrase_scphrase_word_nodistill() {
    name=phrase.scphrase.word.nodistill
    python /n/home13/jchiu/projects/OpenNMT-py/train.py \
        -encoder_type brnn \
        -data $PHRASEDATA_MACHINE_WORD_NODISTILL \
        -src_phrase_mappings /n/rush_lab/data/iwslt14-de-en/data/iwslt14.tokenized.phrase.de-en/phrase.src.pkl \
        -unigram_vocab /n/rush_lab/data/iwslt14-de-en/data-onmt/iwslt14.tokenized.de-en.3-3.vocab.pt \
        -scale_phrases \
        -save_model $MODEL/$name/$name.lr1.clip5 \
        -gpuid 3 \
        -learning_rate 1 \
        -max_grad_norm 5 \
        -epochs 25 \
        | tee ${LOG}/$name.lr1.clip5.log
}

generate_phrase_scphrase_word_nodistill() {
    name=phrase.scphrase.word.nodistill
    python /n/home13/jchiu/projects/OpenNMT-py/translate.py \
        -model $MODEL/$name/phrase.scphrase.word.nodistill.lr1.clip5_acc_60.90_ppl_8.39_e17.pt \
        -gpu 3 \
        -src /n/holylfs/LABS/rush_lab/data/iwslt14-de-en/data/iwslt14.tokenized.phrase.de-en/test.phrase.de \
        -output /n/rush_lab/jc/onmt/gen/$name.test.en \
        -beam_size 5
}

train_phrase_amachine_word_nodistill() {
    name=phrase.amachine.word.nodistill
    python /n/home13/jchiu/projects/OpenNMT-py/train.py \
        -encoder_type brnn \
        -data $PHRASEDATA_MACHINE_WORD_NODISTILL \
        -src_phrase_mappings /n/rush_lab/data/iwslt14-de-en/data/iwslt14.tokenized.phrase.de-en/phrase.src.pkl \
        -unigram_vocab /n/rush_lab/data/iwslt14-de-en/data-onmt/iwslt14.tokenized.de-en.3-3.vocab.pt \
        -save_model $MODEL/$name/$name.lr1.clip5 \
        -gpuid 3 \
        -learning_rate 1 \
        -max_grad_norm 5 \
        -epochs 25 \
        | tee ${LOG}/$name.lr1.clip5.log
}

generate_phrase_amachine_word() {
    python /n/home13/jchiu/projects/OpenNMT-py/translate.py \
        -model /n/rush_lab/jc/onmt/models/phrase.cmachine.word/phrase.cmachine.word.lr1.clip5_acc_58.83_ppl_25.27_e6.pt \
        -gpu 3 \
        -src /n/holylfs/LABS/rush_lab/data/iwslt14-de-en/data/iwslt14.tokenized.phrase.de-en/test.de \
        -output /n/rush_lab/jc/onmt/gen/phrase.cmachine.word.test.en \
        -beam_size 5
}
