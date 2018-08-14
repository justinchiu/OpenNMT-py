# rotowire (smaller)
if [ -n "$odyssey" ]; then
    text=/n/rush_lab/jc/code/data2text/boxscore-data/rotowire
else
    text=/home/justinchiu/research/data2text/boxscore-data/rotowire
fi
# train.json valid.json test.json
data=data/rotowire/roto

preprocess_rw() {
    mkdir -p data/rotowire
    # TODO(justinchiu): need to figure out
    # workaround for dynamic_dict and share_vocab
    # since we don't want to share the vocab (do we?) and
    # the copying vocab is constant (and not dynamic)
    python preprocess.py \
        -train_src ${text}/train.json \
        -valid_src ${text}/valid.json \
        -train_tgt ${text}/train.json \
        -valid_tgt ${text}/valid.json \
        -data_type sam \
        -tgt_vocab_size 80000 \
        -tgt_words_min_frequency 0 \
        -tgt_seq_length 1000 \
        -save_data $data
}

train_rw_soft() {
    # No IE component
    seed=3435
    name=model_soft
    gpuid=0
    python train.py \
        -data $data \
        -save_model $name -gpuid $gpuid -seed $seed \
        -src_word_vec_size 512 \
        -tgt_word_vec_size 512 \
        -memory_size 1024 \
        -decoder_rnn_size 768 \
        -attention_size 512 \
        -encoder_type brnn -batch_size 6 \
        -accum_count 1 -valid_batch_size 32 \
        -epochs 30 -optim adam \
        -learning_rate 3e-4 \
        -adam_eps 1e-8 \
        -start_decay_at 2 \
        -global_attention mlp \
        -report_every 1000 | tee $name.log
}
