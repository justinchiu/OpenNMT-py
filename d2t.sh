# rotowire (smaller)
text=/n/rush_lab/jc/code/data2text/boxscore-data/rotowire
# train.json valid.json test.json
data=data/rotowire

preprocess_rw() {
    mkdir -p data/rotowire
    python preprocess.py \
        -train_json ${text}/train.json \
        -valid_json ${text}/valid.json \
        -data_type sam \
        -vocab_size 80000 \
        -min_frequency 0 \
        -seq_length 1000 \
        -save_data $DATA
}

train_rw_soft() {
    # No IE component
    seed=3435
    name=model_soft
    gpuid=0
    python train.py \
        -data $DATA \
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
