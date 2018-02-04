"""
This file is for models creation, which consults options
and creates each encoder and decoder accordingly.
"""
import torch.nn as nn

import onmt
import onmt.Models
import onmt.modules
from onmt.Models import NMTModel, MeanEncoder, RNNEncoder, \
                        StdRNNDecoder, InputFeedRNNDecoder
from onmt.modules import Embeddings, ImageEncoder, CopyGenerator, \
                         TransformerEncoder, TransformerDecoder, \
                         CNNEncoder, CNNDecoder, PhrasePolytope
from onmt.modules.Embeddings import PhraseEmbeddings


def make_embeddings(opt, word_dict, feature_dicts, for_encoder=True):
    """
    Make an Embeddings instance.
    Args:
        opt: the option in current environment.
        word_dict(Vocab): words dictionary.
        feature_dicts([Vocab], optional): a list of feature dictionary.
        for_encoder(bool): make Embeddings for encoder or decoder?
    """
    if for_encoder:
        embedding_dim = opt.src_word_vec_size
    else:
        embedding_dim = opt.tgt_word_vec_size

    word_padding_idx = word_dict.stoi[onmt.IO.PAD_WORD]
    num_word_embeddings = len(word_dict)

    feats_padding_idx = [feat_dict.stoi[onmt.IO.PAD_WORD]
                         for feat_dict in feature_dicts]
    num_feat_embeddings = [len(feat_dict) for feat_dict in
                           feature_dicts]

    if hasattr(word_dict, "unigram_size"):
        return PhraseEmbeddings(
            embedding_dim,
            opt.position_encoding,
            opt.dropout,
            word_padding_idx,
            word_dict.unigram_size,
            word_dict.idx_mapping,
            nn.LSTM(
                input_size=embedding_dim,
                hidden_size=int(embedding_dim/2),
                num_layers=1,
                bias=False,
                batch_first=False,
                bidirectional=True
            )
        )
    return Embeddings(embedding_dim,
                      opt.position_encoding,
                      opt.feat_merge,
                      opt.feat_vec_exponent,
                      opt.feat_vec_size,
                      opt.dropout,
                      word_padding_idx,
                      feats_padding_idx,
                      num_word_embeddings,
                      num_feat_embeddings)


def make_phrase_embeddings(opt, word_dict):
    """
    Make an Embeddings instance.
    Args:
        opt: the option in current environment.
        word_dict(Vocab): words dictionary.
    """
    embedding_dim = opt.src_word_vec_size

    word_padding_idx = word_dict.stoi[onmt.IO.PAD_WORD]
    num_word_embeddings = len(word_dict)

    return PhraseEmbeddings(
        embedding_dim,
        opt.position_encoding,
        opt.dropout,
        word_padding_idx,
        num_word_embeddings,
        opt.phrase_mappings,
        opt.comp_fn
    )


def make_encoder(opt, embeddings):
    """
    Various encoder dispatcher function.
    Args:
        opt: the option in current environment.
        embeddings (Embeddings): vocab embeddings for this encoder.
    """
    if opt.encoder_type == "transformer":
        return TransformerEncoder(opt.enc_layers, opt.rnn_size,
                                  opt.dropout, embeddings)
    elif opt.encoder_type == "cnn":
        return CNNEncoder(opt.enc_layers, opt.rnn_size,
                          opt.cnn_kernel_width,
                          opt.dropout, embeddings)
    elif opt.encoder_type == "mean":
        return MeanEncoder(opt.enc_layers, embeddings)
    else:
        # "rnn" or "brnn"
        return RNNEncoder(opt.rnn_type, opt.brnn, opt.enc_layers,
                          opt.rnn_size, opt.dropout, embeddings)


def make_decoder(opt, embeddings):
    """
    Various decoder dispatcher function.
    Args:
        opt: the option in current environment.
        embeddings (Embeddings): vocab embeddings for this decoder.
    """
    if opt.decoder_type == "transformer":
        return TransformerDecoder(opt.dec_layers, opt.rnn_size,
                                  opt.global_attention, opt.copy_attn,
                                  opt.dropout, embeddings)
    elif opt.decoder_type == "cnn":
        return CNNDecoder(opt.dec_layers, opt.rnn_size,
                          opt.global_attention, opt.copy_attn,
                          opt.cnn_kernel_width, opt.dropout,
                          embeddings)
    elif opt.input_feed:
        return InputFeedRNNDecoder(opt.rnn_type, opt.brnn,
                                   opt.dec_layers, opt.rnn_size,
                                   opt.global_attention,
                                   opt.coverage_attn,
                                   opt.context_gate,
                                   opt.copy_attn,
                                   opt.dropout,
                                   embeddings)
    else:
        return StdRNNDecoder(opt.rnn_type, opt.brnn,
                             opt.dec_layers, opt.rnn_size,
                             opt.global_attention,
                             opt.coverage_attn,
                             opt.context_gate,
                             opt.copy_attn,
                             opt.dropout,
                             embeddings)


def make_base_model(model_opt, fields, gpu, checkpoint=None):
    """
    Args:
        model_opt: the option loaded from checkpoint.
        fields: `Field` objects for the model.
        gpu(int): whether to use gpu.
        checkpoint: the model gnerated by train phase, or a resumed snapshot
                    model from a stopped training.
    Returns:
        the NMTModel.
    """
    assert model_opt.model_type in ["text", "img"], \
        ("Unsupported model type %s" % (model_opt.model_type))

    # Make encoder.
    if model_opt.model_type == "text":
        src_dict = fields["src"].vocab
        feature_dicts = onmt.IO.collect_feature_dicts(fields, 'src')
        src_embeddings = make_embeddings(model_opt, src_dict,
                                         feature_dicts)
        encoder = make_encoder(model_opt, src_embeddings)
    else:
        encoder = ImageEncoder(model_opt.layers,
                               model_opt.brnn,
                               model_opt.rnn_size,
                               model_opt.dropout)

    # Make decoder.
    tgt_dict = fields["tgt"].vocab
    # TODO: prepare for a future where tgt features are possible.
    feature_dicts = onmt.IO.collect_feature_dicts(fields, 'tgt')
    tgt_embeddings = make_embeddings(model_opt, tgt_dict,
                                     feature_dicts, for_encoder=False)

    # Share the embedding matrix - preprocess with share_vocab required
    if model_opt.share_embeddings:
        tgt_embeddings.word_lut.weight = src_embeddings.word_lut.weight

    decoder = make_decoder(model_opt, tgt_embeddings)

    # Make NMTModel(= encoder + decoder).
    model = NMTModel(encoder, decoder)

    # Make Generator.
    if not model_opt.copy_attn and not (hasattr(model_opt, "tgt_phrase_mappings") and model_opt.tgt_phrase_mappings):
        generator = nn.Sequential(
            nn.Linear(model_opt.rnn_size, len(fields["tgt"].vocab)),
            nn.LogSoftmax())
        if model_opt.share_decoder_embeddings:
            generator[0].weight = decoder.embeddings.word_lut.weight
    elif hasattr(model_opt, "tgt_phrase_mappings") and model_opt.tgt_phrase_mappings:
        dec_embeddings = make_embeddings(
            model_opt, tgt_dict, feature_dicts, for_encoder=False)
        generator = nn.Sequential(
            PhrasePolytope(dec_embeddings, model_opt.tgt_distractors),
            nn.LogSoftmax()
        )
    else:
        generator = CopyGenerator(model_opt, fields["src"].vocab,
                                  fields["tgt"].vocab)

    # Load the model states from checkpoint or initialize them.
    if checkpoint is not None:
        print('Loading model parameters.')
        if "encoder.embeddings._buf" in checkpoint["model"]:
            checkpoint["model"]["encoder.embeddings._buf"].resize_(0)
        if "decoder.embeddings._buf" in checkpoint["model"]:
            checkpoint["model"]["decoder.embeddings._buf"].resize_(0)
        if "0.phrase_lut._buf" in checkpoint["generator"]:
            checkpoint["generator"]["0.phrase_lut._buf"].resize_(0)
        model.load_state_dict(checkpoint['model'])
        generator.load_state_dict(checkpoint['generator'])
    else:
        if model_opt.param_init != 0.0:
            print('Intializing model parameters.')
            for p in model.parameters():
                p.data.uniform_(-model_opt.param_init, model_opt.param_init)
            for p in generator.parameters():
                p.data.uniform_(-model_opt.param_init, model_opt.param_init)
        model.encoder.embeddings.load_pretrained_vectors(
                model_opt.pre_word_vecs_enc, model_opt.fix_word_vecs_enc)
        model.decoder.embeddings.load_pretrained_vectors(
                model_opt.pre_word_vecs_dec, model_opt.fix_word_vecs_dec)

    # Add generator to model (this registers it as parameter of model).
    model.generator = generator

    # Make the whole model leverage GPU if indicated to do so.
    if gpu > -1:
        model.cuda(gpu)
    else:
        model.cpu()

    return model
