ROOT=/n/holylfs/LABS/rush_lab/data/iwslt14-de-en
TEXT=${ROOT}/data/iwslt14.tokenized.de-en
PHRASE=${ROOT}/data/iwslt14.tokenized.phrase.de-en

DATA=${ROOT}/data-onmt/iwslt14.tokenized.de-en.3-3

PHRASEDATA_NATURAL_NATURAL_NODISTILL=${ROOT}/data-onmt/iwslt14.tokenized.phrase.de-en.3-3.natural.natural.nodistill
# Poorly named, but only one phrase distribution for source.
PHRASEDATA_NATURAL_WORD_NODISTILL=${ROOT}/data-onmt/iwslt14.tokenized.phrase.de-en.3-3.machine.word.nodistill
PHRASEDATA_WORD_NATURAL_NODISTILL=${ROOT}/data-onmt/iwslt14.tokenized.phrase.de-en.3-3.word.natural.nodistill

PHRASEDATA_NATURAL_WORD_NODISTILL_REPEAT=${ROOT}/data-onmt/iwslt14.tokenized.phrase.de-en.3-3.machine.word.nodistill.repeat

MODEL=/n/rush_lab/jc/onmt/models
SAVE=/n/rush_lab/jc/onmt/validation

validate_baseline () {
    name=baseline-brnn2
    mkdir -p $SAVE/$name
    python /n/home13/jchiu/projects/OpenNMT-py/validate.py \
        --data $DATA \
        --checkpoint_path $MODEL/baseline-brnn2/baseline-brnn2_acc_62.80_ppl_7.72_e13.pt \
        --modelname $name \
        --savepath $SAVE/$name \
        --devid $1
}

validate_phrase_cnatural_scnatural_nodistill () {
    name=phrase.cnatural.scnatural.nodistill
    mkdir -p $SAVE/$name
    python /n/home13/jchiu/projects/OpenNMT-py/validate.py \
        --data $PHRASEDATA_NATURAL_NATURAL_NODISTILL \
        --checkpoint_path /n/rush_lab/jc/onmt/models/$name/phrase.cnatural.scnatural.nodistill.lr1.clip5_acc_40.81_ppl_49.96_e14.pt \
        --src_phrase_mappings /n/rush_lab/data/iwslt14-de-en/data/iwslt14.tokenized.phrase.de-en/phrase.src.pkl \
        --tgt_phrase_mappings /n/rush_lab/data/iwslt14-de-en/data/iwslt14.tokenized.phrase.de-en/phrase.natural.natural.tgt.pkl \
        --unigram_vocab /n/rush_lab/data/iwslt14-de-en/data-onmt/iwslt14.tokenized.de-en.3-3.vocab.pt \
        --modelname $name \
        --savepath $SAVE/$name \
        --devid $1
}

validate_phrase_cnatural_word_nodistill () {
    name=phrase.cnatural.word.nodistill
    mkdir -p $SAVE/$name
    python /n/home13/jchiu/projects/OpenNMT-py/validate.py \
        --data $PHRASEDATA_NATURAL_WORD_NODISTILL \
        --checkpoint_path /n/rush_lab/jc/onmt/models/phrase.cmachine.word.nodistill/phrase.cmachine.word.nodistill.lr1.clip5_acc_61.80_ppl_8.08_e18.pt \
        --src_phrase_mappings /n/rush_lab/data/iwslt14-de-en/data/iwslt14.tokenized.phrase.de-en/phrase.src.pkl \
        --unigram_vocab /n/rush_lab/data/iwslt14-de-en/data-onmt/iwslt14.tokenized.de-en.3-3.vocab.pt \
        --modelname $name \
        --savepath $SAVE/$name \
        --devid $1
}

# bad repeat, on the embeddings.
validate_phrase_cnatural_word_nodistill_repeat () {
    name=phrase.cphrase.word.nodistill.repeat
    mkdir -p $SAVE/$name
    python /n/home13/jchiu/projects/OpenNMT-py/validate.py \
        --data $PHRASEDATA_NATURAL_WORD_NODISTILL_REPEAT \
        --checkpoint_path /n/rush_lab/jc/onmt/models/$name/phrase.cphrase.word.nodistill.repeat.lr1.clip5_acc_62.25_ppl_7.94_e25.pt \
        --src_phrase_mappings /n/rush_lab/data/iwslt14-de-en/data/iwslt14.tokenized.phrase.de-en/phrase.src.pkl \
        --unigram_vocab /n/rush_lab/data/iwslt14-de-en/data-onmt/iwslt14.tokenized.de-en.3-3.vocab.pt \
        --modelname $name \
        --savepath $SAVE/$name \
        --devid $1
}

# repeat output of encdoer
# we use datawords here as a hack during viz
validate_phrase_cphraser_word_nodistill() {
    name=phrase.cphraser.word.nodistill.s148
    mkdir -p $SAVE/$name
    python /n/home13/jchiu/projects/OpenNMT-py/validate.py \
        --data $PHRASEDATA_NATURAL_WORD_NODISTILL \
        --data $PHRASEDATA_NATURAL_WORD_NODISTILL_REPEAT \
        --checkpoint_path /n/rush_lab/jc/onmt/models/$name/$name.lr1.clip5_acc_61.87_ppl_8.06_e25.pt \
        --src_phrase_mappings /n/rush_lab/data/iwslt14-de-en/data/iwslt14.tokenized.phrase.de-en/phrase.src.pkl \
        --unigram_vocab /n/rush_lab/data/iwslt14-de-en/data-onmt/iwslt14.tokenized.de-en.3-3.vocab.pt \
        --modelname $name \
        --savepath $SAVE/$name \
        --devid $1
}

# repeat output of encdoer
validate_phrase_cphrasere_word_nodistill() {
    name=phrase.cphrasere.word.nodistill.s137
    mkdir -p $SAVE/$name
    python /n/home13/jchiu/projects/OpenNMT-py/validate.py \
        --data $PHRASEDATA_NATURAL_WORD_NODISTILL \
        --datawords $DATA \
        --checkpoint_path /n/rush_lab/jc/onmt/models/$name/$name.lr1.clip5_acc_60.00_ppl_8.97_e12.pt \
        --src_phrase_mappings /n/rush_lab/data/iwslt14-de-en/data/iwslt14.tokenized.phrase.de-en/phrase.src.pkl \
        --unigram_vocab /n/rush_lab/data/iwslt14-de-en/data-onmt/iwslt14.tokenized.de-en.3-3.vocab.pt \
        --modelname $name \
        --savepath $SAVE/$name \
        --devid $1
}

# mixed skip connection between encoding and embedding
validate_baseline_brnn_add_word_attn() {
    name=baseline.brnn.e.s218
    mkdir -p $SAVE/$name
    python /n/home13/jchiu/projects/OpenNMT-py/validate.py \
        --data $DATA \
        --checkpoint_path /n/rush_lab/jc/onmt/models/$name/$name.lr1.clip5_acc_61.34_ppl_8.39_e13.pt \
        --modelname $name \
        --savepath $SAVE/$name \
        --devid $1
}
