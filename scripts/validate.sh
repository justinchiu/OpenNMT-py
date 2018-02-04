ROOT=/n/holylfs/LABS/rush_lab/data/iwslt14-de-en
TEXT=${ROOT}/data/iwslt14.tokenized.de-en
PHRASE=${ROOT}/data/iwslt14.tokenized.phrase.de-en

DATA=${ROOT}/data-onmt/iwslt14.tokenized.de-en.3-3

PHRASEDATA_NATURAL_NATURAL_NODISTILL=${ROOT}/data-onmt/iwslt14.tokenized.phrase.de-en.3-3.natural.natural.nodistill
# Poorly named, but only one phrase distribution for source.
PHRASEDATA_NATURAL_WORD_NODISTILL=${ROOT}/data-onmt/iwslt14.tokenized.phrase.de-en.3-3.machine.word.nodistill
PHRASEDATA_WORD_NATURAL_NODISTILL=${ROOT}/data-onmt/iwslt14.tokenized.phrase.de-en.3-3.word.natural.nodistill

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
        --devid 0
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
        --devid 0
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
        --devid 0
}
