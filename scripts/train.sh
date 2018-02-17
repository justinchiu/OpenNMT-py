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
    seed=$(od -A n -t d -N 1 /dev/urandom |tr -d ' ')
    name=baseline3.s$seed
    mkdir -p $MODEL/$name
    python /n/home13/jchiu/projects/OpenNMT-py/train.py \
    	-data $DATA \
    	-save_model $MODEL/$name/$name \
        -seed $seed \
    	-gpuid $1 \
    	| tee ${LOG}/$name.log
}

train_baseline_brnn() {
    seed=$(od -A n -t d -N 1 /dev/urandom |tr -d ' ')
    name=baseline-brnn2.s$seed
    mkdir -p $MODEL/$name
    python /n/home13/jchiu/projects/OpenNMT-py/train.py \
        -encoder_type brnn \
    	-data $DATA \
    	-save_model $MODEL/$name/$name \
        -seed $seed \
        -epochs 25 \
    	-gpuid $1 \
    	| tee ${LOG}/$name.log
}

train_bpe_baseline() {
    seed=$(od -A n -t d -N 1 /dev/urandom |tr -d ' ')
    name=baseline2bpe.s$seed
    mkdir -p $MODEL/$name
    python /n/home13/jchiu/projects/OpenNMT-py/train.py \
    	-data $BPEDATA \
    	-save_model $MODEL/$name/$name \
        -seed $seed \
    	-gpuid $1 \
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

generate_baseline_brnn_test_FUCK() {
    python /n/home13/jchiu/projects/OpenNMT-py/translate.py \
        -model $MODEL/baseline-brnn2/baseline-brnn2_acc_62.80_ppl_7.72_e13.pt \
        -gpu 2 \
        -src /n/holylfs/LABS/rush_lab/data/iwslt14-de-en/data/iwslt14.tokenized.phrase.de-en/test.de \
        -output /n/rush_lab/jc/onmt/gen/baseline.brnn.test.en.FUCK \
        -beam_size 5 
}

convert_to_phrases() {
    python /n/home13/jchiu/projects/bpephrase/python/ngrams.py
}

train_distill() {
    seed=$(od -A n -t d -N 1 /dev/urandom |tr -d ' ')
    name=distill.s$seed
    mkdir -p $MODEL/$name
    python /n/home13/jchiu/projects/OpenNMT-py/train.py \
        -encoder_type brnn \
        -data $DISTILLDATA \
        -save_model $MODEL/$name/$name.lr1.clip5 \
        -seed $seed \
        -gpuid $1 \
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
    seed=$(od -A n -t d -N 1 /dev/urandom |tr -d ' ')
    name=phrase.machine.machine.s$seed
    mkdir -p $MODEL/$name
    python /n/home13/jchiu/projects/OpenNMT-py/train.py \
        -encoder_type brnn \
        -data $PHRASEDATA_MACHINE_MACHINE \
        -save_model $MODEL/$name/$name.lr1.clip5 \
        -seed $seed \
        -gpuid $1 \
        -learning_rate 1 \
        -max_grad_norm 5 \
        -epochs 25 \
        | tee ${LOG}/$name.lr1.clip5.log
}

train_phrase_machine_natural () {
    seed=$(od -A n -t d -N 1 /dev/urandom |tr -d ' ')
    name=phrase.machine.natural.s$seed
    mkdir -p $MODEL/$name
    python /n/home13/jchiu/projects/OpenNMT-py/train.py \
        -encoder_type brnn \
        -data $PHRASEDATA_MACHINE_NATURAL \
        -save_model $MODEL/$name/$name.lr1.clip5 \
        -seed $seed \
        -gpuid $1 \
        -learning_rate 1 \
        -max_grad_norm 5 \
        -epochs 25 \
        | tee ${LOG}/$name.lr1.clip5.log
}

train_phrase_natural_natural () {
    seed=$(od -A n -t d -N 1 /dev/urandom |tr -d ' ')
    name=phrase.natural.natural.s$seed
    mkdir -p $MODEL/$name
    python /n/home13/jchiu/projects/OpenNMT-py/train.py \
        -encoder_type brnn \
        -data $PHRASEDATA_NATURAL_NATURAL \
        -save_model $MODEL/$name/$name.lr1.clip5 \
        -seed $seed \
        -gpuid $1 \
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
    seed=$(od -A n -t d -N 1 /dev/urandom |tr -d ' ')
    name=phrase.machine.word.s$seed
    mkdir -p $MODEL/$name
    python /n/home13/jchiu/projects/OpenNMT-py/train.py \
        -encoder_type brnn \
        -data $PHRASEDATA_MACHINE_WORD \
        -save_model $MODEL/$name/$name.lr1.clip5 \
        -seed $seed \
        -gpuid $1 \
        -learning_rate 1 \
        -max_grad_norm 5 \
        -epochs 25 \
        | tee ${LOG}/$name.lr1.clip5.log
}

train_phrase_machine_word_nodistill() {
    seed=$(od -A n -t d -N 1 /dev/urandom |tr -d ' ')
    name=phrase.machine.word.nodistill.s$seed
    mkdir -p $MODEL/$name
    python /n/home13/jchiu/projects/OpenNMT-py/train.py \
        -encoder_type brnn \
        -data $PHRASEDATA_MACHINE_WORD_NODISTILL \
        -save_model $MODEL/$name/$name.lr1.clip5 \
        -seed $seed \
        -gpuid $1 \
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
    seed=$(od -A n -t d -N 1 /dev/urandom |tr -d ' ')
    name=phrase.cmachine.word.s$seed
    python /n/home13/jchiu/projects/OpenNMT-py/train.py \
        -encoder_type brnn \
        -data $PHRASEDATA_MACHINE_WORD \
        -src_phrase_mappings /n/rush_lab/data/iwslt14-de-en/data/iwslt14.tokenized.phrase.de-en/phrase.src.pkl \
        -unigram_vocab /n/rush_lab/data/iwslt14-de-en/data-onmt/iwslt14.tokenized.de-en.3-3.vocab.pt \
        -save_model $MODEL/$name/$name.lr1.clip5 \
        -seed $seed \
        -gpuid $1 \
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
    seed=$(od -A n -t d -N 1 /dev/urandom |tr -d ' ')
    name=phrase.cmachine.word.nodistill.s$seed
    python /n/home13/jchiu/projects/OpenNMT-py/train.py \
        -encoder_type brnn \
        -data $PHRASEDATA_MACHINE_WORD_NODISTILL \
        -src_phrase_mappings /n/rush_lab/data/iwslt14-de-en/data/iwslt14.tokenized.phrase.de-en/phrase.src.pkl \
        -unigram_vocab /n/rush_lab/data/iwslt14-de-en/data-onmt/iwslt14.tokenized.de-en.3-3.vocab.pt \
        -save_model $MODEL/$name/$name.lr1.clip5 \
        -seed $seed \
        -gpuid $1 \
        -learning_rate 1 \
        -max_grad_norm 5 \
        -epochs 25 \
        | tee ${LOG}/$name.lr1.clip5.log
}

generate_phrase_cmachine_word_nodistill() {
#        -model /n/rush_lab/jc/onmt/models/phrase.cmachine.word.nodistill/phrase.cmachine.word.nodistill.lr1.clip5_acc_61.80_ppl_8.08_e18.pt \
    name=phrase.cmachine.word.nodistill.s33
    python /n/home13/jchiu/projects/OpenNMT-py/translate.py \
        -model $MODEL/$name/phrase.cmachine.word.nodistill.s33.lr1.clip5_acc_61.96_ppl_8.01_e15.pt \
        -gpu 2 \
        -src /n/holylfs/LABS/rush_lab/data/iwslt14-de-en/data/iwslt14.tokenized.phrase.de-en/test.de \
        -output /n/rush_lab/jc/onmt/gen/$name.test.en \
        -beam_size 5
}

train_phrase_cmachine_scmachine() {
    seed=$(od -A n -t d -N 1 /dev/urandom |tr -d ' ')
    name=phrase.cmachine.scmachine.s$seed
    python /n/home13/jchiu/projects/OpenNMT-py/train.py \
        -encoder_type brnn \
        -data $PHRASEDATA_MACHINE_MACHINE \
        -src_phrase_mappings /n/rush_lab/data/iwslt14-de-en/data/iwslt14.tokenized.phrase.de-en/phrase.src.pkl \
        -tgt_phrase_mappings /n/rush_lab/data/iwslt14-de-en/data/iwslt14.tokenized.phrase.de-en/phrase.machine.machine.tgt.pkl \
        -unigram_vocab /n/rush_lab/data/iwslt14-de-en/data-onmt/iwslt14.tokenized.de-en.3-3.vocab.pt \
        -tgt_distractors 2048 \
        -save_model $MODEL/$name/$name.lr1.clip5 \
        -seed $seed \
        -gpuid $1 \
        -learning_rate 1 \
        -max_grad_norm 5 \
        -epochs 25 \
        | tee ${LOG}/$name.lr1.clip5.log
}

train_phrase_cmachine_scmachine_nodistill() {
    seed=$(od -A n -t d -N 1 /dev/urandom |tr -d ' ')
    name=phrase.cmachine.scmachine.nodistill.s$seed
    python /n/home13/jchiu/projects/OpenNMT-py/train.py \
        -encoder_type brnn \
        -data $PHRASEDATA_MACHINE_MACHINE_NODISTILL \
        -src_phrase_mappings /n/rush_lab/data/iwslt14-de-en/data/iwslt14.tokenized.phrase.de-en/phrase.src.pkl \
        -tgt_phrase_mappings /n/rush_lab/data/iwslt14-de-en/data/iwslt14.tokenized.phrase.de-en/phrase.machine.machine.tgt.pkl \
        -unigram_vocab /n/rush_lab/data/iwslt14-de-en/data-onmt/iwslt14.tokenized.de-en.3-3.vocab.pt \
        -tgt_distractors 2048 \
        -save_model $MODEL/$name/$name.lr1.clip5 \
        -seed $seed \
        -gpuid $1 \
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
    seed=$(od -A n -t d -N 1 /dev/urandom |tr -d ' ')
    name=phrase.word.scnatural.nodistill.s$seed
    python /n/home13/jchiu/projects/OpenNMT-py/train.py \
        -encoder_type brnn \
        -data $PHRASEDATA_WORD_NATURAL_NODISTILL \
        -tgt_phrase_mappings /n/rush_lab/data/iwslt14-de-en/data/iwslt14.tokenized.phrase.de-en/phrase.natural.natural.tgt.pkl \
        -unigram_vocab /n/rush_lab/data/iwslt14-de-en/data-onmt/iwslt14.tokenized.de-en.3-3.vocab.pt \
        -tgt_distractors 2048 \
        -save_model $MODEL/$name/$name.lr1.clip5 \
        -seed $seed \
        -gpuid $1 \
        -learning_rate 1 \
        -max_grad_norm 5 \
        -epochs 25 \
        | tee ${LOG}/$name.lr1.clip5.log
}

train_phrase_cnatural_scnatural_nodistill() {
    seed=$(od -A n -t d -N 1 /dev/urandom |tr -d ' ')
    name=phrase.cnatural.scnatural.nodistill.s$seed
    python /n/home13/jchiu/projects/OpenNMT-py/train.py \
        -encoder_type brnn \
        -data $PHRASEDATA_NATURAL_NATURAL_NODISTILL \
        -src_phrase_mappings /n/rush_lab/data/iwslt14-de-en/data/iwslt14.tokenized.phrase.de-en/phrase.src.pkl \
        -tgt_phrase_mappings /n/rush_lab/data/iwslt14-de-en/data/iwslt14.tokenized.phrase.de-en/phrase.natural.natural.tgt.pkl \
        -unigram_vocab /n/rush_lab/data/iwslt14-de-en/data-onmt/iwslt14.tokenized.de-en.3-3.vocab.pt \
        -tgt_distractors 2048 \
        -save_model $MODEL/$name/$name.lr1.clip5 \
        -seed $seed \
        -gpuid $1 \
        -learning_rate 1 \
        -max_grad_norm 5 \
        -epochs 25 \
        | tee ${LOG}/$name.lr1.clip5.log
}

dbg_train_phrase_cnatural_scnatural_nodistill() {
    seed=$(od -A n -t d -N 1 /dev/urandom |tr -d ' ')
    name=phrase.cnatural.scnatural.nodistill.s$seed
    python /n/home13/jchiu/projects/OpenNMT-py/train.py \
        -encoder_type brnn \
        -data $PHRASEDATA_NATURAL_NATURAL_NODISTILL \
        -src_phrase_mappings /n/rush_lab/data/iwslt14-de-en/data/iwslt14.tokenized.phrase.de-en/phrase.src.pkl \
        -tgt_phrase_mappings /n/rush_lab/data/iwslt14-de-en/data/iwslt14.tokenized.phrase.de-en/phrase.natural.natural.tgt.pkl \
        -unigram_vocab /n/rush_lab/data/iwslt14-de-en/data-onmt/iwslt14.tokenized.de-en.3-3.vocab.pt \
        -tgt_distractors 2048 \
        -save_model /tmp/trash \
        -seed $seed \
        -gpuid $1 \
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
    seed=$(od -A n -t d -N 1 /dev/urandom |tr -d ' ')
    #name=phrase.word.natural.nodistill
    name=phrase.word.natural.nodistill2.s$seed
    python /n/home13/jchiu/projects/OpenNMT-py/train.py \
        -encoder_type brnn \
        -data $PHRASEDATA_WORD_NATURAL_NODISTILL \
        -save_model $MODEL/$name/$name.lr1.clip5 \
        -seed $seed \
        -gpuid $1 \
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
    seed=$(od -A n -t d -N 1 /dev/urandom |tr -d ' ')
    name=phrase.phrase.word.nodistill.repeat.s$seed
    mkdir -p $MODEL/$name
    python /n/home13/jchiu/projects/OpenNMT-py/train.py \
        -encoder_type brnn \
        -data $PHRASEDATA_PHRASE_WORD_NODISTILL_REPEAT \
        -save_model $MODEL/$name/$name.lr1.clip5 \
        -seed $seed \
        -gpuid $1 \
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
    seed=$(od -A n -t d -N 1 /dev/urandom |tr -d ' ')
    name=phrase.cphrase.word.nodistill.repeat.s$seed
    python /n/home13/jchiu/projects/OpenNMT-py/train.py \
        -encoder_type brnn \
        -data $PHRASEDATA_PHRASE_WORD_NODISTILL_REPEAT \
        -src_phrase_mappings /n/rush_lab/data/iwslt14-de-en/data/iwslt14.tokenized.phrase.de-en/phrase.src.pkl \
        -unigram_vocab /n/rush_lab/data/iwslt14-de-en/data-onmt/iwslt14.tokenized.de-en.3-3.vocab.pt \
        -save_model $MODEL/$name/$name.lr1.clip5 \
        -seed $seed \
        -gpuid $1 \
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
    seed=$(od -A n -t d -N 1 /dev/urandom |tr -d ' ')
    name=phrase.scphrase.word.nodistill.s$seed
    python /n/home13/jchiu/projects/OpenNMT-py/train.py \
        -encoder_type brnn \
        -data $PHRASEDATA_MACHINE_WORD_NODISTILL \
        -src_phrase_mappings /n/rush_lab/data/iwslt14-de-en/data/iwslt14.tokenized.phrase.de-en/phrase.src.pkl \
        -unigram_vocab /n/rush_lab/data/iwslt14-de-en/data-onmt/iwslt14.tokenized.de-en.3-3.vocab.pt \
        -scale_phrases \
        -save_model $MODEL/$name/$name.lr1.clip5 \
        -seed $seed \
        -gpuid $1 \
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

# cphraser has compositional source phrase + repeated after the encoder
train_phrase_cphraser_word_nodistill() {
    seed=$(od -A n -t d -N 1 /dev/urandom |tr -d ' ')
    name=phrase.cphraser.word.nodistill.s$seed
    python /n/home13/jchiu/projects/OpenNMT-py/train.py \
        -encoder_type brnn \
        -data $PHRASEDATA_MACHINE_WORD_NODISTILL \
        -src_phrase_mappings /n/rush_lab/data/iwslt14-de-en/data/iwslt14.tokenized.phrase.de-en/phrase.src.pkl \
        -unigram_vocab /n/rush_lab/data/iwslt14-de-en/data-onmt/iwslt14.tokenized.de-en.3-3.vocab.pt \
        -repeat_encoder_phrases \
        -save_model $MODEL/$name/$name.lr1.clip5 \
        -seed $seed \
        -gpuid $1 \
        -learning_rate 1 \
        -max_grad_norm 5 \
        -epochs 25 \
        | tee ${LOG}/$name.lr1.clip5.log
}

generate_phrase_cphraser_word() {
    name=phrase.cphraser.word.nodistill.s148
    #name=phrase.cphraser.word.nodistill.s181
        #-model $MODEL/$name/$name.lr1.clip5_acc_61.88_ppl_8.03_e20.pt \
    python /n/home13/jchiu/projects/OpenNMT-py/translate.py \
        -model $MODEL/$name/$name.lr1.clip5_acc_61.88_ppl_8.06_e16.pt \
        -gpu 3 \
        -src /n/holylfs/LABS/rush_lab/data/iwslt14-de-en/data/iwslt14.tokenized.phrase.de-en/test.phrase.de \
        -output /n/rush_lab/jc/onmt/gen/$name.test.en \
        -beam_size 5
}

# cphraser has compositional source phrase + repeated after the encoder + original word embeddings
train_phrase_cphrasere_word_nodistill() {
    seed=$(od -A n -t d -N 1 /dev/urandom |tr -d ' ')
    name=phrase.cphrasere.word.nodistill.s$seed
    python /n/home13/jchiu/projects/OpenNMT-py/train.py \
        -encoder_type brnn \
        -data $PHRASEDATA_MACHINE_WORD_NODISTILL \
        -dataword $DATA \
        -src_phrase_mappings /n/rush_lab/data/iwslt14-de-en/data/iwslt14.tokenized.phrase.de-en/phrase.src.pkl \
        -unigram_vocab /n/rush_lab/data/iwslt14-de-en/data-onmt/iwslt14.tokenized.de-en.3-3.vocab.pt \
        -repeat_encoder_phrases \
        -add_word_vectors \
        -share_context_embeddings \
        -save_model $MODEL/$name/$name.lr1.clip5 \
        -seed $seed \
        -gpuid $1 \
        -learning_rate 1 \
        -max_grad_norm 5 \
        -epochs 25 \
        | tee ${LOG}/$name.lr1.clip5.log
}

train_phrase_cphrasere_word_nodistill_nosharectxtemb() {
    seed=$(od -A n -t d -N 1 /dev/urandom |tr -d ' ')
    name=phrase.cphrasere.word.nodistill.nosharectxtemb.s$seed
    python /n/home13/jchiu/projects/OpenNMT-py/train.py \
        -encoder_type brnn \
        -data $PHRASEDATA_MACHINE_WORD_NODISTILL \
        -dataword $DATA \
        -src_phrase_mappings /n/rush_lab/data/iwslt14-de-en/data/iwslt14.tokenized.phrase.de-en/phrase.src.pkl \
        -unigram_vocab /n/rush_lab/data/iwslt14-de-en/data-onmt/iwslt14.tokenized.de-en.3-3.vocab.pt \
        -repeat_encoder_phrases \
        -add_word_vectors \
        -save_model $MODEL/$name/$name.lr1.clip5 \
        -seed $seed \
        -gpuid $1 \
        -learning_rate 1 \
        -max_grad_norm 5 \
        -epochs 25 \
        | tee ${LOG}/$name.lr1.clip5.log
}

train_phrase_cphrasere_word_nodistill_moredropout() {
    # Note...this is added INTO the code, so it'll be hard to pull it out.
    seed=$(od -A n -t d -N 1 /dev/urandom |tr -d ' ')
    name=phrase.cphrasere.word.nodistill.moredropout.s$seed
    python /n/home13/jchiu/projects/OpenNMT-py/train.py \
        -encoder_type brnn \
        -data $PHRASEDATA_MACHINE_WORD_NODISTILL \
        -dataword $DATA \
        -src_phrase_mappings /n/rush_lab/data/iwslt14-de-en/data/iwslt14.tokenized.phrase.de-en/phrase.src.pkl \
        -unigram_vocab /n/rush_lab/data/iwslt14-de-en/data-onmt/iwslt14.tokenized.de-en.3-3.vocab.pt \
        -repeat_encoder_phrases \
        -add_word_vectors \
        -save_model $MODEL/$name/$name.lr1.clip5 \
        -seed $seed \
        -gpuid $1 \
        -learning_rate 1 \
        -max_grad_norm 5 \
        -epochs 25 \
        | tee ${LOG}/$name.lr1.clip5.log
}

train_phrase_cphrasere_word_nodistill_nosharectxtemb_moredropout() {
    # Note...this is added INTO the code, so it'll be hard to pull it out.
    seed=$(od -A n -t d -N 1 /dev/urandom |tr -d ' ')
    name=phrase.cphrasere.word.nodistill.nosharectxtemb.moredropout.s$seed
    python /n/home13/jchiu/projects/OpenNMT-py/train.py \
        -encoder_type brnn \
        -data $PHRASEDATA_MACHINE_WORD_NODISTILL \
        -dataword $DATA \
        -src_phrase_mappings /n/rush_lab/data/iwslt14-de-en/data/iwslt14.tokenized.phrase.de-en/phrase.src.pkl \
        -unigram_vocab /n/rush_lab/data/iwslt14-de-en/data-onmt/iwslt14.tokenized.de-en.3-3.vocab.pt \
        -repeat_encoder_phrases \
        -add_word_vectors \
        -save_model $MODEL/$name/$name.lr1.clip5 \
        -seed $seed \
        -gpuid $1 \
        -learning_rate 1 \
        -max_grad_norm 5 \
        -epochs 25 \
        | tee ${LOG}/$name.lr1.clip5.log
}

generate_phrase_cphrasere_word() {
    name=phrase.cphrasere.word.nodistill
    python /n/home13/jchiu/projects/OpenNMT-py/translate.py \
        -model $MODEL/$name/phrase.cphrasere.word.nodistill.lr1.clip5_acc_59.11_ppl_9.15_e10.pt \
        -gpu 0 \
        -src /n/holylfs/LABS/rush_lab/data/iwslt14-de-en/data/iwslt14.tokenized.phrase.de-en/test.phrase.de \
        -srcwords /n/holylfs/LABS/rush_lab/data/iwslt14-de-en/data/iwslt14.tokenized.phrase.de-en/test.de \
        -output /n/rush_lab/jc/onmt/gen/$name.test.en \
        -beam_size 5
}

train_phrase_cphrase_word_nodistill_plr() {
    seed=$(od -A n -t d -N 1 /dev/urandom |tr -d ' ')
    name=phrase.cphrase.word.nodistill.plr.s$seed
    plr=2
    python /n/home13/jchiu/projects/OpenNMT-py/train.py \
        -encoder_type brnn \
        -data $PHRASEDATA_MACHINE_WORD_NODISTILL \
        -src_phrase_mappings /n/rush_lab/data/iwslt14-de-en/data/iwslt14.tokenized.phrase.de-en/phrase.src.pkl \
        -unigram_vocab /n/rush_lab/data/iwslt14-de-en/data-onmt/iwslt14.tokenized.de-en.3-3.vocab.pt \
        -save_model $MODEL/$name/$name.lr1.clip5 \
        -seed $seed \
        -gpuid $1 \
        -learning_rate 1 \
        -phrase_lr $plr \
        -max_grad_norm 5 \
        -epochs 25 \
        | tee ${LOG}/$name.lr1.clip5.plr$plr.log
}
