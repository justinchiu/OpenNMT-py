#!/usr/bin/env python

from __future__ import division

import argparse
import glob
import os
import sys
import random

import numpy as np

import torch
import torch.nn as nn
from torch import cuda
import torch.nn.functional as F

import onmt
import onmt.io
import onmt.Models
import onmt.ModelConstructor
import onmt.modules
from onmt.Utils import use_gpu
import opts


parser = argparse.ArgumentParser(
    description='prune.py',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# opts.py
opts.add_md_help_argument(parser)
opts.model_opts(parser)
opts.train_opts(parser)

parser.add_argument("--n-test-vectors", type=int, default=10000)

parser.add_argument('--qw_i', default=2, help='Maximum number of fraction bits for inputs', type=int)
parser.add_argument('--qw_f', default=8, help='Maximum number of fraction bits for inputs ', type=int)

parser.add_argument('--min_weight', default=-1.0, type=float, help='Clamp weight to minimum value (used for sparse quantization)')
parser.add_argument('--max_weight', default= 1.0, type=float, help='Clamp weight to maximum value (used for sparse quantization)')

parser.add_argument('--qh_i', default=8, help='Maximum number of integer bits for hidden state', type=int)
parser.add_argument('--qh_f', default=8, help='Maximum number of fraction bits for hidden state', type=int)

parser.add_argument('--qi_i', default=8, help='Maximum number of fraction bits for inputs', type=int)
parser.add_argument('--qi_f', default=8, help='Maximum number of fraction bits for inputs', type=int)

opt = parser.parse_args()
if opt.word_vec_size != -1:
    opt.src_word_vec_size = opt.word_vec_size
    opt.tgt_word_vec_size = opt.word_vec_size

if opt.layers != -1:
    opt.enc_layers = opt.layers
    opt.dec_layers = opt.layers

opt.brnn = (opt.encoder_type == "brnn")
if opt.seed > 0:
    random.seed(opt.seed)
    torch.manual_seed(opt.seed)

if opt.rnn_type == "SRU" and not opt.gpuid:
    raise AssertionError("Using SRU requires -gpuid set.")

if torch.cuda.is_available() and not opt.gpuid:
    print("WARNING: You have a CUDA device, should run with -gpuid 0")

if opt.gpuid:
    cuda.set_device(opt.gpuid[0])
    if opt.seed > 0:
        torch.cuda.manual_seed(opt.seed)

if len(opt.gpuid) > 1:
    sys.stderr.write("Sorry, multigpu isn't supported yet, coming soon!\n")
    sys.exit(1)

# Set up the Crayon logging server.
if opt.exp_host != "":
    from pycrayon import CrayonClient

    cc = CrayonClient(hostname=opt.exp_host)

    experiments = cc.get_experiment_names()
    print(experiments)
    if opt.exp in experiments:
        cc.remove_experiment(opt.exp)
    experiment = cc.create_experiment(opt.exp)

if opt.tensorboard:
    from tensorboardX import SummaryWriter
    writer = SummaryWriter(opt.tensorboard_log_dir, comment="Onmt")


# QUANTIZATION


# not relevant unless in-between layers
def relu_quantize(x, qi, qf):
    fmax = 1. - float(torch.pow(2., torch.FloatTensor([-1. * qf])).numpy()[0])
    imax = float((torch.pow(2., torch.FloatTensor([qi])) - 1).numpy()[0])
    fdiv = float(torch.pow(2., torch.FloatTensor([-qf])).numpy()[0])
    x = torch.floor ( x / fdiv) * fdiv
    return x.clamp(0, imax + fmax)

def quantize_(x, qi, qf):
    fmax = 1. - float(torch.pow(2., torch.FloatTensor([-1. * qf])).numpy()[0])
    imax =      float((torch.pow(2., torch.FloatTensor([qi-1])) - 1).numpy()[0])
    imin = -1 * float((torch.pow(2., torch.FloatTensor([qi-1]))).numpy()[0])
    fdiv = float(torch.pow(2., torch.FloatTensor([-qf])).numpy()[0])
    x = torch.floor ( x / fdiv) * fdiv
    return torch.clamp(x, imin, imax + fmax)

def quantize_sparse_weights(x, qi, qf):
    pos_weights_mask   = torch.gt(x, 0)
    pos_weights_values = torch.masked_select(x, pos_weights_mask)

    pos_min = pos_weights_values.min()
    pos_max = pos_weights_values.max()

    pos_weights = x * pos_weights_mask.float()
    pos_weights = pos_weights - pos_min
    new_max = pos_weights.max()
    pos_weights = torch.div ( pos_weights, new_max )
    pos_weights = pos_weights * pos_weights_mask.float()

    pos_weights = quantize_(pos_weights, qi, qf)
    pos_weights = torch.mul(pos_weights , new_max)
    pos_weights = pos_weights + pos_min
    pos_weights = pos_weights * pos_weights_mask.float()

    return pos_weights

def weight_quantize_(
    parameter,
    min_weight=opt.min_weight,
    max_weight=opt.max_weight,
    qw_i=opt.qw_i,
    qw_f=opt.qw_f
):
    weight = parameter.data
    weight.clamp_(min_weight, max_weight)

    # Positive quantization
    pos_weights = quantize_sparse_weights(weight, qw_i, qw_f)

    # Negative Quantization
    neg_weights_mask = torch.lt(weight, 0)
    neg_weights = (weight * neg_weights_mask.float()).abs()
    neg_weights = quantize_sparse_weights(neg_weights, qw_i, qw_f) * -1

    new_weights = pos_weights + neg_weights
    parameter.data = new_weights

def quantize_model_(model):
    for name, p in model.named_parameters():
        if "rnn.weight" in name or "lut" in name:
            weight_quantize_(p)

def forward(model, input, state=None):
    encoder_rnn = model.encoder.rnn

def report_func(epoch, batch, num_batches,
                start_time, lr, report_stats):
    """
    This is the user-defined batch-level traing progress
    report function.

    Args:
        epoch(int): current epoch count.
        batch(int): current batch count.
        num_batches(int): total number of batches.
        start_time(float): last report time.
        lr(float): current learning rate.
        report_stats(Statistics): old Statistics instance.
    Returns:
        report_stats(Statistics): updated Statistics instance.
    """
    if batch % opt.report_every == -1 % opt.report_every:
        report_stats.output(epoch, batch + 1, num_batches, start_time)
        if opt.exp_host:
            report_stats.log("progress", experiment, lr)
        if opt.tensorboard:
            # Log the progress using the number of batches on the x-axis.
            report_stats.log_tensorboard(
                "progress", writer, lr, epoch * num_batches + batch)
        report_stats = onmt.Statistics()

    return report_stats


class DatasetLazyIter(object):
    """ An Ordered Dataset Iterator, supporting multiple datasets,
        and lazy loading.

    Args:
        datsets (list): a list of datasets, which are lazily loaded.
        fields (dict): fields dict for the datasets.
        batch_size (int): batch size.
        batch_size_fn: custom batch process function.
        device: the GPU device.
        is_train (bool): train or valid?
    """

    def __init__(self, datasets, fields, batch_size, batch_size_fn,
                 device, is_train):
        self.datasets = datasets
        self.fields = fields
        self.batch_size = batch_size
        self.batch_size_fn = batch_size_fn
        self.device = device
        self.is_train = is_train

        self.cur_iter = self._next_dataset_iterator(datasets)
        # We have at least one dataset.
        assert self.cur_iter is not None

    def __iter__(self):
        dataset_iter = (d for d in self.datasets)
        while self.cur_iter is not None:
            for batch in self.cur_iter:
                yield batch
            self.cur_iter = self._next_dataset_iterator(dataset_iter)

    def __len__(self):
        # We return the len of cur_dataset, otherwise we need to load
        # all datasets to determine the real len, which loses the benefit
        # of lazy loading.
        assert self.cur_iter is not None
        return len(self.cur_iter)

    def get_cur_dataset(self):
        return self.cur_dataset

    def _next_dataset_iterator(self, dataset_iter):
        try:
            self.cur_dataset = next(dataset_iter)
        except StopIteration:
            return None

        # We clear `fields` when saving, restore when loading.
        self.cur_dataset.fields = self.fields

        # Sort batch by decreasing lengths of sentence required by pytorch.
        # sort=False means "Use dataset's sortkey instead of iterator's".
        return onmt.io.OrderedIterator(
            dataset=self.cur_dataset, batch_size=self.batch_size,
            batch_size_fn=self.batch_size_fn,
            device=self.device, train=self.is_train,
            sort=False, sort_within_batch=True,
            repeat=False)


def make_dataset_iter(datasets, fields, opt, is_train=True):
    """
    This returns user-defined train/validate data iterator for the trainer
    to iterate over during each train epoch. We implement simple
    ordered iterator strategy here, but more sophisticated strategy
    like curriculum learning is ok too.
    """
    batch_size = opt.batch_size if is_train else opt.valid_batch_size
    batch_size_fn = None
    if is_train and opt.batch_type == "tokens":
        global max_src_in_batch, max_tgt_in_batch

        def batch_size_fn(new, count, sofar):
            global max_src_in_batch, max_tgt_in_batch
            if count == 1:
                max_src_in_batch = 0
                max_tgt_in_batch = 0
            max_src_in_batch = max(max_src_in_batch,  len(new.src) + 2)
            max_tgt_in_batch = max(max_tgt_in_batch,  len(new.tgt) + 1)
            src_elements = count * max_src_in_batch
            tgt_elements = count * max_tgt_in_batch
            return max(src_elements, tgt_elements)

    device = opt.gpuid[0] if opt.gpuid else -1

    return DatasetLazyIter(datasets, fields, batch_size, batch_size_fn,
                           device, is_train)


def make_loss_compute(model, tgt_vocab, opt, train=True):
    """
    This returns user-defined LossCompute object, which is used to
    compute loss in train/validate process. You can implement your
    own *LossCompute class, by subclassing LossComputeBase.
    """
    if opt.copy_attn:
        compute = onmt.modules.CopyGeneratorLossCompute(
            model.generator, tgt_vocab, opt.copy_attn_force,
            opt.copy_loss_by_seqlength)
    else:
        compute = onmt.Loss.NMTLossCompute(
            model.generator, tgt_vocab,
            label_smoothing=opt.label_smoothing if train else 0.0)

    if use_gpu(opt):
        compute.cuda()

    return compute

def train_model(model, fields, optim, data_type, model_opt, prune_schedule):
    train_loss = make_loss_compute(model, fields["tgt"].vocab, opt)
    valid_loss = make_loss_compute(model, fields["tgt"].vocab, opt,
                                   train=False)

    trunc_size = opt.truncated_decoder  # Badly named...
    shard_size = opt.max_generator_batches
    norm_method = opt.normalization
    grad_accum_count = opt.accum_count

    trainer = onmt.Trainer(model, train_loss, valid_loss, optim,
                           trunc_size, shard_size, data_type,
                           norm_method, grad_accum_count)


    """
    print('\nStart training...')
    print(' * number of epochs: %d, starting from Epoch %d' %
          (opt.epochs + 1 - opt.start_epoch, opt.start_epoch))
    print(' * batch size: %d' % opt.batch_size)

    for epoch in range(opt.start_epoch, opt.epochs + 1):
        print('')

        # 1. Train for one epoch on the training set.
        train_iter = make_dataset_iter(lazily_load_dataset("train"),
                                       fields, opt)
        #train_stats = trainer.train_prune(
            #train_iter, epoch, prune_schedule[epoch], report_func)
        print('Train perplexity: %g' % train_stats.ppl())
        print('Train accuracy: %g' % train_stats.accuracy())
    """

    # 2. Validate on the validation set.
    valid_iter = make_dataset_iter(lazily_load_dataset("valid"),
                                   fields, opt,
                                   is_train=False)
    valid_iter = list(valid_iter)
    save_file = "/n/rush_lab/jc/hardware/iwslt14-de-en/vectors"
    

    # layer 0
    ib_file = "{}/{}".format(save_file, "input_bias_l0")
    np.savetxt(
        ib_file,
        model.encoder.rnn.bias_ih_l0.data.cpu().numpy()
    )
    hb_file = "{}/{}".format(save_file, "hidden_bias_l0")
    np.savetxt(
        hb_file,
        model.encoder.rnn.bias_hh_l0.data.cpu().numpy()
    )
    iw_file = "{}/{}".format(save_file, "input_weight_l0")
    np.savetxt(
        iw_file,
        model.encoder.rnn.weight_ih_l0.data.cpu().numpy()
    )
    hw_file = "{}/{}".format(save_file, "hidden_weight_l0")
    np.savetxt(
        hw_file,
        model.encoder.rnn.weight_hh_l0.data.cpu().numpy()
    )

    # layer 1
    ib_file = "{}/{}".format(save_file, "input_bias_l1")
    np.savetxt(
        ib_file,
        model.encoder.rnn.bias_ih_l0.data.cpu().numpy()
    )
    hb_file = "{}/{}".format(save_file, "hidden_bias_l1")
    np.savetxt(
        hb_file,
        model.encoder.rnn.bias_hh_l0.data.cpu().numpy()
    )
    iw_file = "{}/{}".format(save_file, "input_weight_l1")
    np.savetxt(
        iw_file,
        model.encoder.rnn.weight_ih_l0.data.cpu().numpy()
    )
    hw_file = "{}/{}".format(save_file, "hidden_weight_l1")
    np.savetxt(
        hw_file,
        model.encoder.rnn.weight_hh_l0.data.cpu().numpy()
    )

    for i, batch in enumerate(valid_iter):
        embs = model.encoder.embeddings(
            batch.src[0].view(-1, 1, 1)
        )
        embs_np = embs.data.squeeze().cpu().numpy()

        encoder_rnn = model.encoder.rnn

        # forward
        forward_input_int = []
        forward_hidden_int = []
        forward_output = []
        for t in range(batch.src[0].size(0)):
            forward_input_int.append(F.linear(
                embs[t], encoder_rnn.weight_ih_l0, encoder_rnn.bias_ih_l0))
            if t == 0:
                forward_hidden_int.append(encoder_rnn.bias_hh_l0)
            else:
                forward_hidden_int.append(F.linear(
                    forward_output[-1], encoder_rnn.weight_hh_l0, encoder_rnn.bias_hh_l0))
            forward_output.append(relu_quantize(
                F.relu(forward_input_int[t] + forward_hidden_int[t]), opt.qh_i, opt.qh_f))

        # backward
        backward_input_int = []
        backward_hidden_int = []
        backward_output = []
        for t in range(batch.src[0].size(0))[::-1]:
            backward_input_int.insert(0, F.linear(
                embs[t], encoder_rnn.weight_ih_l0, encoder_rnn.bias_ih_l0))
            if t == batch.src[0].size(0)-1:
                backward_hidden_int.insert(0, encoder_rnn.bias_hh_l0)
            else:
                backward_hidden_int.insert(0, F.linear(
                    backward_output[-1], encoder_rnn.weight_hh_l0, encoder_rnn.bias_hh_l0))
            backward_output.insert(0, relu_quantize(
                F.relu(backward_input_int[0] + backward_hidden_int[0]), opt.qh_i, opt.qh_f))

        for t in range(batch.src[0].size(0)):
            vec_file = "{}/{}_{}_{}_l0".format(save_file, "input", i, t)
            np.savetxt(vec_file, embs_np[t])
            vec_file = "{}/{}_{}_{}_l0".format(save_file, "input_int", i, t)
            np.savetxt(vec_file, forward_input_int[t].data.squeeze().cpu().numpy())
            vec_file = "{}/{}_{}_{}_l0".format(save_file, "hidden_int", i, t)
            np.savetxt(vec_file, forward_hidden_int[t].data.squeeze().cpu().numpy())
            vec_file = "{}/{}_{}_{}_l0".format(save_file, "output", i, t)
            np.savetxt(vec_file, forward_output[t].data.squeeze().cpu().numpy())

        if i > 100:
            break

    valid_stats = trainer.validate(valid_iter)
    print('Validation perplexity: %g' % valid_stats.ppl())
    print('Validation accuracy: %g' % valid_stats.accuracy())

    """
        # 3. Log to remote server.
        if opt.exp_host:
            train_stats.log("train", experiment, optim.lr)
            valid_stats.log("valid", experiment, optim.lr)
        if opt.tensorboard:
            train_stats.log_tensorboard("train", writer, optim.lr, epoch)
            train_stats.log_tensorboard("valid", writer, optim.lr, epoch)

        # 4. Update the learning rate
        trainer.epoch_step(valid_stats.ppl(), epoch)

        # 5. Drop a checkpoint if needed.
        if epoch >= opt.start_checkpoint_at:
            trainer.drop_checkpoint(model_opt, epoch, fields, valid_stats)
            """


def check_save_model_path():
    save_model_path = os.path.abspath(opt.save_model)
    model_dirname = os.path.dirname(save_model_path)
    if not os.path.exists(model_dirname):
        os.makedirs(model_dirname)


def tally_parameters(model):
    n_params = sum([p.nelement() for p in model.parameters()])
    print('* number of parameters: %d' % n_params)
    enc = 0
    dec = 0
    for name, param in model.named_parameters():
        if 'encoder' in name:
            enc += param.nelement()
        elif 'decoder' or 'generator' in name:
            dec += param.nelement()
    print('encoder: ', enc)
    print('decoder: ', dec)


def lazily_load_dataset(corpus_type):
    """
    Dataset generator. Don't do extra stuff here, like printing,
    because they will be postponed to the first loading time.

    Args:
        corpus_type: 'train' or 'valid'
    Returns:
        A list of dataset, the dataset(s) are lazily loaded.
    """
    assert corpus_type in ["train", "valid"]

    def lazy_dataset_loader(pt_file, corpus_type):
        dataset = torch.load(pt_file)
        print('Loading %s dataset from %s, number of examples: %d' %
              (corpus_type, pt_file, len(dataset)))
        return dataset

    # Sort the glob output by file name (by increasing indexes).
    pts = sorted(glob.glob(opt.data + '.' + corpus_type + '.[0-9]*.pt'))
    if pts:
        for pt in pts:
            yield lazy_dataset_loader(pt, corpus_type)
    else:
        # Only one onmt.io.*Dataset, simple!
        pt = opt.data + '.' + corpus_type + '.pt'
        yield lazy_dataset_loader(pt, corpus_type)


def load_fields(dataset, data_type, checkpoint):
    if checkpoint is not None:
        print('Loading vocab from checkpoint at %s.' % opt.train_from)
        fields = onmt.io.load_fields_from_vocab(
            checkpoint['vocab'], data_type)
    else:
        fields = onmt.io.load_fields_from_vocab(
            torch.load(opt.data + '.vocab.pt'), data_type)
    fields = dict([(k, f) for (k, f) in fields.items()
                   if k in dataset.examples[0].__dict__])

    if data_type == 'text':
        print(' * vocabulary size. source = %d; target = %d' %
              (len(fields['src'].vocab), len(fields['tgt'].vocab)))
    else:
        print(' * vocabulary size. target = %d' %
              (len(fields['tgt'].vocab)))

    return fields


def collect_report_features(fields):
    src_features = onmt.io.collect_features(fields, side='src')
    tgt_features = onmt.io.collect_features(fields, side='tgt')

    for j, feat in enumerate(src_features):
        print(' * src feature %d size = %d' % (j, len(fields[feat].vocab)))
    for j, feat in enumerate(tgt_features):
        print(' * tgt feature %d size = %d' % (j, len(fields[feat].vocab)))


def build_model(model_opt, opt, fields, checkpoint):
    print('Building model...')
    model = onmt.ModelConstructor.make_base_model(model_opt, fields,
                                                  use_gpu(opt), checkpoint)
    if len(opt.gpuid) > 1:
        print('Multi gpu training: ', opt.gpuid)
        model = nn.DataParallel(model, device_ids=opt.gpuid, dim=1)
    print(model)

    return model


def build_optim(model, checkpoint):
    if opt.train_from:
        print('Loading optimizer from checkpoint.')
        optim = checkpoint['optim']
        optim.optimizer.load_state_dict(
            checkpoint['optim'].optimizer.state_dict())
        optim._set_rate(checkpoint["opt"].learning_rate)
        optim.lr_decay = checkpoint["opt"].learning_rate_decay
    else:
        print('Making optimizer for training.')
        optim = onmt.Optim(
            opt.optim, opt.learning_rate, opt.max_grad_norm,
            lr_decay=opt.learning_rate_decay,
            start_decay_at=opt.start_decay_at,
            beta1=opt.adam_beta1,
            beta2=opt.adam_beta2,
            adagrad_accum=opt.adagrad_accumulator_init,
            decay_method=opt.decay_method,
            warmup_steps=opt.warmup_steps,
            model_size=opt.rnn_size)

    optim.set_parameters(model.named_parameters())

    return optim


def main():
    # Load checkpoint if we resume from a previous training.
    if opt.train_from:
        print('Loading checkpoint from %s' % opt.train_from)
        checkpoint = torch.load(opt.train_from,
                                map_location=lambda storage, loc: storage)
        model_opt = checkpoint['opt']
        # I don't like reassigning attributes of opt: it's not clear.
        opt.start_epoch = checkpoint['epoch'] + 1
        checkpoint["opt"].learning_rate = opt.learning_rate
        checkpoint["opt"].learning_rate_decay = opt.learning_rate_decay
        model_opt.save_model = opt.save_model
        model_opt.valid_batch_size = opt.valid_batch_size
    else:
        checkpoint = None
        model_opt = opt

    # Peek the fisrt dataset to determine the data_type.
    # (All datasets have the same data_type).
    first_dataset = next(lazily_load_dataset("train"))
    data_type = first_dataset.data_type

    # Load fields generated from preprocess phase.
    fields = load_fields(first_dataset, data_type, checkpoint)

    # Report src/tgt features.
    collect_report_features(fields)

    # Build model.
    model = build_model(model_opt, opt, fields, checkpoint)
    tally_parameters(model)
    check_save_model_path()

    # Build optimizer.
    optim = build_optim(model, checkpoint)

    # Don't retrain because of pruning?
    """
    end_prune_epoch   = int(0.8 * (opt.epochs - opt.start_epoch)) + opt.start_epoch
    ramp_epochs = end_prune_epoch+1 - opt.start_epoch

    prune_schedule = {
        e: float(i) / ramp_epochs * opt.prune_threshold
        for i, e in enumerate(range(opt.start_epoch-1, end_prune_epoch+1))
    }
    for e in range(end_prune_epoch, opt.epochs+1):
        prune_schedule[e] = opt.prune_threshold
    print(prune_schedule)
    """
    # Do training.
    quantize_model_(model)
    # Hacked to just output val perf, but intermediate steps aren't quantized
    train_model(model, fields, optim, data_type, model_opt, None)
    # If using tensorboard for logging, close the writer after training.
    if opt.tensorboard:
        writer.close()


if __name__ == "__main__":
    main()
