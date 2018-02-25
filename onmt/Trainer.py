from __future__ import division
"""
This is the loadable seq2seq trainer library that is
in charge of training details, loss compute, and statistics.
See train.py for a use case of this library.

Note!!! To make this a general library, we implement *only*
mechanism things here(i.e. what to do), and leave the strategy
things to users(i.e. how to do it). Also see train.py(one of the
users of this library) for the strategy things we do.
"""
import time
import sys
import math
import torch
import torch.nn as nn
from torch.autograd import Variable as V

import onmt
import onmt.modules


class Statistics(object):
    """
    Train/validate loss statistics.
    """
    def __init__(self, loss=0, n_words=0, n_correct=0, n_phrases=0):
        self.loss = loss
        self.n_words = n_words
        self.n_correct = n_correct
        self.n_phrases = n_phrases
        self.n_src_words = 0
        self.start_time = time.time()

    def update(self, stat):
        self.loss += stat.loss
        self.n_words += stat.n_words
        self.n_correct += stat.n_correct
        self.n_phrases += stat.n_phrases

    def accuracy(self):
        return (100 * (self.n_correct / self.n_words)
            if self.n_phrases == 0 else 100 * (self.n_correct / self.n_phrases))

    def ppl(self):
        return math.exp(min(self.loss / self.n_words, 100))

    def elapsed_time(self):
        return time.time() - self.start_time

    def output(self, epoch, batch, n_batches, start):
        t = self.elapsed_time()
        print(("Epoch %2d, %5d/%5d; acc: %6.2f; ppl: %6.2f; " +
               "%3.0f src tok/s; %3.0f tgt tok/s; %6.0f s elapsed") %
              (epoch, batch,  n_batches,
               self.accuracy(),
               self.ppl(),
               self.n_src_words / (t + 1e-5),
               self.n_words / (t + 1e-5),
               time.time() - start))
        sys.stdout.flush()

    def log(self, prefix, experiment, lr):
        t = self.elapsed_time()
        experiment.add_scalar_value(prefix + "_ppl", self.ppl())
        experiment.add_scalar_value(prefix + "_accuracy", self.accuracy())
        experiment.add_scalar_value(prefix + "_tgtper",  self.n_words / t)
        experiment.add_scalar_value(prefix + "_lr", lr)


class Trainer(object):
    def __init__(self, model, train_iter, valid_iter,
                 train_loss, valid_loss, optim,
                 trunc_size, shard_size, trainwords=None, validwords=None):
        """
        Args:
            model: the seq2seq model.
            train_iter: the train data iterator.
            valid_iter: the validate data iterator.
            train_loss: the train side LossCompute object for computing loss.
            valid_loss: the valid side LossCompute object for computing loss.
            optim: the optimizer responsible for lr update.
            trunc_size: a batch is divided by several truncs of this size.
            shard_size: compute loss in shards of this size for efficiency.
        """
        # Basic attributes.
        self.model = model
        self.train_iter = train_iter
        self.valid_iter = valid_iter
        self.train_loss = train_loss
        self.valid_loss = valid_loss
        self.optim = optim
        self.trunc_size = trunc_size
        self.shard_size = shard_size

        # Phrase word
        self.trainwords = trainwords
        self.validwords = validwords

        # Set model in training mode.
        self.model.train()

    def train(self, epoch, report_func=None):
        """ Called for each epoch to train. """
        total_stats = Statistics()
        report_stats = Statistics()

        attn_weights_cpu = None
        attn_weights_gpu = None
        if hasattr(self.model.decoder, "scale_phrases") and self.model.decoder.scale_phrases:
            attn_weights_cpu = torch.FloatTensor()
            if self.train_iter.device >= 0:
                attn_weights_gpu = torch.FloatTensor().cuda(self.train_iter.device)

        for i, batch in enumerate(self.train_iter):
            target_size = batch.tgt.size(0)
            # Truncated BPTT
            trunc_size = self.trunc_size if self.trunc_size else target_size

            dec_state = None
            _, src_lengths = batch.src

            src = onmt.IO.make_features(batch, 'src')
            tgt_outer = onmt.IO.make_features(batch, 'tgt')

            # Begin mods
            attn_weights = None
            if hasattr(self.model.decoder, "scale_phrases") and self.model.decoder.scale_phrases:
                bsz = batch.tgt.size(1)
                attn_weights_cpu.resize_(batch.src[0].size()).fill_(0)
                nwords = [[word.count("_")+1 for word in batch.dataset[idx].src] for idx in batch.indices.data.tolist()]
                # not sure how slow this is
                for x in range(bsz):
                    for y in range(len(nwords[x])):
                        attn_weights_cpu[y,x] = nwords[x][y]
                attn_weights_cpu.log_()
                if batch.tgt.is_cuda:
                    attn_weights_gpu.resize_(attn_weights_cpu.size())
                    attn_weights_gpu.copy_(attn_weights_cpu)
                    attn_weights = V(attn_weights_gpu)
                else:
                    attn_weights = V(attn_weights_cpu)
            if hasattr(self.model, "ctxt_fn") and self.model.ctxt_fn is not None:
                rles = [[word.count("_")+1 for word in batch.dataset[idx].src] for idx in batch.indices.data.tolist()]
                self.model.ctxt_fn.rle_to_idxs(rles)
                if hasattr(self.model.ctxt_fn, "lut") and self.model.ctxt_fn.lut is not None:
                    self.model.ctxt_fn.dataset = self.trainwords
                    self.model.ctxt_fn.get_words(batch.indices)
                if hasattr(self.model.ctxt_fn, "poslut") and self.model.ctxt_fn.poslut is not None:
                    # For the positions we just need the lengths of the source sentences
                    self.model.ctxt_fn.dataset = self.trainwords
                    self.model.ctxt_fn.get_positions(batch.indices)
            if hasattr(self.model.generator[0], "phrase_lut"):
                self.model.generator[0].reset_perm()
                ctgt_outer = self.model.generator[0].collapse_target(tgt_outer)
                self.train_loss.criterion.weight = self.train_loss.criterion.weight.new(
                    len(self.model.generator[0].vertices) + self.model.generator[0].phrase_lut.word_vocab_size)
                self.train_loss.criterion.weight.fill_(1)
                self.train_loss.criterion.weight[self.train_loss.padding_idx] = 0
                batch.tgt.copy_(ctgt_outer.squeeze(2)) # ? lol...
            # end mods

            report_stats.n_src_words += src_lengths.sum()

            for j in range(0, target_size-1, trunc_size):
                # 1. Create truncated target.
                tgt = tgt_outer[j: j + trunc_size]

                # 2. F-prop all but generator.
                self.model.zero_grad()
                outputs, attns, dec_state = \
                    self.model(src, tgt, src_lengths, dec_state, attn_weights=attn_weights)

                # 3. Compute loss in shards for memory efficiency.
                batch_stats = self.train_loss.sharded_compute_loss(
                        batch, outputs, attns, j,
                        trunc_size, self.shard_size)

                # 4. Update the parameters and statistics.
                self.optim.step()
                total_stats.update(batch_stats)
                report_stats.update(batch_stats)

                # If truncated, don't backprop fully.
                if dec_state is not None:
                    dec_state = dec_state.detach()

            if report_func is not None:
                report_stats = report_func(
                        epoch, i, len(self.train_iter),
                        total_stats.start_time, self.optim.lr, report_stats)

        return total_stats

    def validate(self):
        """ Called for each epoch to validate. """
        # Set model in validating mode.
        self.model.eval()

        stats = Statistics()

        attn_weights_cpu = None
        attn_weights_gpu = None
        if self.model.decoder.scale_phrases:
            attn_weights_cpu = torch.FloatTensor()
            if self.train_iter.device >= 0:
                attn_weights_gpu = torch.FloatTensor().cuda(self.train_iter.device)
        #with torch.no_grad():
        if True:
            for batch in self.valid_iter:
                _, src_lengths = batch.src
                src = onmt.IO.make_features(batch, 'src')
                tgt = onmt.IO.make_features(batch, 'tgt')

                attn_weights = None
                if hasattr(self.model.decoder, "scale_phrases") and self.model.decoder.scale_phrases:
                    bsz = batch.tgt.size(1)
                    attn_weights_cpu.resize_(batch.src[0].size()).fill_(0)
                    nwords = [[word.count("_")+1 for word in batch.dataset[idx].src] for idx in batch.indices.data.tolist()]
                    # not sure how slow this is
                    for x in range(bsz):
                        for y in range(len(nwords[x])):
                            attn_weights_cpu[y,x] = nwords[x][y]
                    attn_weights_cpu.log_()
                    if batch.tgt.is_cuda:
                        attn_weights_gpu.resize_(attn_weights_cpu.size())
                        attn_weights_gpu.copy_(attn_weights_cpu)
                        attn_weights = V(attn_weights_gpu)
                    else:
                        attn_weights = V(attn_weights_cpu)
                if hasattr(self.model, "ctxt_fn") and self.model.ctxt_fn is not None:
                    rles = [[word.count("_")+1 for word in batch.dataset[idx].src] for idx in batch.indices.data.tolist()]
                    self.model.ctxt_fn.rle_to_idxs(rles)
                    if hasattr(self.model.ctxt_fn, "lut") and self.model.ctxt_fn.lut is not None:
                        self.model.ctxt_fn.dataset = self.validwords
                        self.model.ctxt_fn.get_words(batch.indices)
                    if hasattr(self.model.ctxt_fn, "poslut") and self.model.ctxt_fn.poslut is not None:
                        # For the positions we just need the lengths of the source sentences
                        self.model.ctxt_fn.dataset = self.validwords
                        self.model.ctxt_fn.get_positions(batch.indices)

                # F-prop through the model.
                outputs, attns, _ = self.model(src, tgt, src_lengths, attn_weights=attn_weights)

                # Compute loss.
                batch_stats = self.valid_loss.monolithic_compute_loss(
                        batch, outputs, attns)

                # Update statistics.
                stats.update(batch_stats)

        # Set model back to training mode.
        self.model.train()

        return stats

    def epoch_step(self, ppl, epoch):
        """ Called for each epoch to update learning rate. """
        return self.optim.updateLearningRate(ppl, epoch)

    def drop_checkpoint(self, opt, epoch, fields, valid_stats):
        """ Called conditionally each epoch to save a snapshot. """
        real_model = (self.model.module
                      if isinstance(self.model, nn.DataParallel)
                      else self.model)
        real_generator = (real_model.generator.module
                          if isinstance(real_model.generator, nn.DataParallel)
                          else real_model.generator)

        model_state_dict = real_model.state_dict()
        model_state_dict = {k: v for k, v in model_state_dict.items()
                            if 'generator' not in k}
        generator_state_dict = real_generator.state_dict()
        checkpoint = {
            'model': model_state_dict,
            'generator': generator_state_dict,
            'vocab': onmt.IO.save_vocab(fields),
            'opt': opt,
            'epoch': epoch,
            'optim': self.optim
        }
        torch.save(checkpoint,
                   '%s_acc_%.2f_ppl_%.2f_e%d.pt'
                   % (opt.save_model, valid_stats.accuracy(),
                      valid_stats.ppl(), epoch))
