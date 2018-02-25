import torch
import pickle

def aeq(*args):
    """
    Assert all arguments have the same value
    """
    arguments = (arg for arg in args)
    first = next(arguments)
    assert all(arg == first for arg in arguments), \
        "Not all arguments have the same value: " + str(args)


def sequence_mask(lengths, max_len=None):
    """
    Creates a boolean mask from sequence lengths.
    """
    batch_size = lengths.numel()
    max_len = max_len or lengths.max()
    mask = torch.arange(0, max_len).type_as(lengths)
    if lengths.is_cuda:
        mask = mask.cuda(lengths.get_device())
    return (mask
            .repeat(batch_size, 1)
            .lt(lengths.unsqueeze(1)))


def use_gpu(opt):
    return (hasattr(opt, 'gpuid') and len(opt.gpuid) > 0) or \
        (hasattr(opt, 'gpu') and opt.gpu > -1)


def add_phrases(vocab, mapping_path):
    # We want to load the original unigram vocabulary and
    # add on the phrase ids to the vocab afterwards, so that
    # for all ids > N, they belong to phrases and ids < N are all unigrams
    unigram_size = len(vocab.itos)
    vocab.unigram_size = unigram_size
    def extend(vocab, word):
        if word not in vocab.stoi:
            vocab.stoi[word] = len(vocab.itos)
            vocab.itos.append(word)
    with open(mapping_path, "rb") as f:
        phrase_mappings = pickle.load(f)
        phrase_mapping = {"_".join(words): list(words) for mapping in phrase_mappings for words, _ in mapping.items()}
        for phrase, _ in phrase_mapping.items():
            extend(vocab, phrase)
        idx_mapping = {vocab.stoi[phrase]: list(map(lambda x: vocab.stoi[x], words)) for phrase, words in phrase_mapping.items()}
        vocab.phrase_mapping = phrase_mapping
        vocab.idx_mapping = idx_mapping
    return vocab
