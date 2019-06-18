# encoding: utf-8

import argparse
import json
import os.path

import numpy
import six
from tqdm import tqdm
import chainer
from chainer import cuda
from chainer.dataset import convert
from chainer import reporter
from chainer import training
from chainer.training import extensions
from torch.utils.data import DataLoader

from dataset import TrfmDataset
from trfm import Transformer
from subfuncs import VaswaniRule


def seq2seq_pad_concat_convert(xy_batch, device, eos_id=0, bos_id=2):
    """
    Args:
        xy_batch (list of tuple of two numpy.ndarray-s or cupy.ndarray-s):
            xy_batch[i][0] is an array
            of token ids of i-th input sentence in a minibatch.
            xy_batch[i][1] is an array
            of token ids of i-th target sentence in a minibatch.
            The shape of each array is `(sentence length, )`.
        device (int or None): Device ID to which an array is sent. If it is
            negative value, an array is sent to CPU. If it is positive, an
            array is sent to GPU with the given ID. If it is ``None``, an
            array is left in the original device.

    Returns:
        Tuple of Converted array.
            (input_sent_batch_array, target_sent_batch_input_array,
            target_sent_batch_output_array).
            The shape of each array is `(batchsize, max_sentence_length)`.
            All sentences are padded with -1 to reach max_sentence_length.
    """

    x_seqs, y_seqs = zip(*xy_batch)

    x_block = convert.concat_examples(x_seqs, device, padding=-1)
    y_block = convert.concat_examples(y_seqs, device, padding=-1)
    xp = cuda.get_array_module(x_block)

    # The paper did not mention eos
    # add eos
    x_block = xp.pad(x_block, ((0, 0), (0, 1)),
                     'constant', constant_values=-1)
    for i_batch, seq in enumerate(x_seqs):
        x_block[i_batch, len(seq)] = eos_id
    x_block = xp.pad(x_block, ((0, 0), (1, 0)),
                     'constant', constant_values=bos_id)

    y_out_block = xp.pad(y_block, ((0, 0), (0, 1)),
                         'constant', constant_values=-1)
    for i_batch, seq in enumerate(y_seqs):
        y_out_block[i_batch, len(seq)] = eos_id

    y_in_block = xp.pad(y_block, ((0, 0), (1, 0)),
                        'constant', constant_values=bos_id)
    return (x_block, y_in_block, y_out_block)


def source_pad_concat_convert(x_seqs, device, eos_id=0, bos_id=2):
    x_block = convert.concat_examples(x_seqs, device, padding=-1)
    xp = cuda.get_array_module(x_block)

    # add eos
    x_block = xp.pad(x_block, ((0, 0), (0, 1)),
                     'constant', constant_values=-1)
    for i_batch, seq in enumerate(x_seqs):
        x_block[i_batch, len(seq)] = eos_id
    x_block = xp.pad(x_block, ((0, 0), (1, 0)),
                     'constant', constant_values=bos_id)
    return x_block



def main():
    parser = argparse.ArgumentParser(
        description='Chainer example: convolutional seq2seq')
    parser.add_argument('--batchsize', '-b', type=int, default=32,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=100,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=0,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--unit', '-u', type=int, default=256,
                        help='Number of units')
    parser.add_argument('--layer', '-l', type=int, default=2,
                        help='Number of layers')
    parser.add_argument('--head', type=int, default=4,
                        help='Number of heads in attention mechanism')
    parser.add_argument('--dropout', '-d', type=float, default=0.1,
                        help='Dropout rate')
    parser.add_argument('--input', '-i', type=str, default='../../data/Enum2Enum',
                        help='Input directory')
    parser.add_argument('--source', '-s', type=str,
                        default='source.txt',
                        help='Filename of train data for source language')
    parser.add_argument('--target', '-t', type=str,
                        default='target.txt',
                        help='Filename of train data for target language')
    parser.add_argument('--source-valid', '-svalid', type=str,
                        default='sval.txt',
                        help='Filename of validation data for source language')
    parser.add_argument('--target-valid', '-tvalid', type=str,
                        default='tval.txt',
                        help='Filename of validation data for target language')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--source-vocab', type=int, default=50,
                        help='Vocabulary size of source language')
    parser.add_argument('--target-vocab', type=int, default=50,
                        help='Vocabulary size of target language')
    parser.add_argument('--use-label-smoothing', action='store_true',
                        help='Use label smoothing for cross entropy')
    parser.add_argument('--embed-position', action='store_true',
                        help='Use position embedding rather than sinusoid')
    parser.add_argument('--use-fixed-lr', action='store_true',
                        help='Use fixed learning rate rather than the ' +
                             'annealing proposed in the paper')
    args = parser.parse_args()
    print(json.dumps(args.__dict__, indent=4))

    print("Creating Dataloader")
    vocab = WordVocab.load_vocab(args.vocab)
    train_dataset = TrfmDataset(args.train_data, vocab, seq_len=args.seq_len)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.n_worker, shuffle=True) 
    val_dataset = TrfmDataset(args.val_data, vocab, seq_len=args.seq_len, is_train=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.n_worker, shuffle=False) 

    # Define Model
    model = Transformer(
        args.layer,
        min(len(source_ids), len(source_words)),
        min(len(target_ids), len(target_words)),
        args.unit,
        h=args.head,
        dropout=args.dropout,
        max_length=500,
        use_label_smoothing=args.use_label_smoothing,
        embed_position=args.embed_position)
    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()
        model.to_gpu(args.gpu)
    # Setup Optimizer
    optimizer = chainer.optimizers.Adam(
        alpha=5e-5,
        beta1=0.9,
        beta2=0.98,
        eps=1e-9
    )
    optimizer.setup(model)

    print("Training Start")
    for epoch in tqdm(range(args.n_epoch)):
        for i, data in tqdm(enumerate(train_loader):
            loss = model(data)
            loss = F.s

            loss, loss_tsm, loss_msm, acc_tsm, acc_msm, validity = trainer.iteration(epoch, iter, data, data_iter)
            if iter % trainer.log_freq == 0:
                with open(log_dir + '/' + args.name + '.csv', 'a') as f:
                    f.write('{:d},{:d},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f}\n'.format(epoch, iter, loss, loss_tsm, loss_msm, acc_tsm, acc_msm, validity))
                if iter % (trainer.log_freq*10) == 0:
                    trainer.save(epoch, iter, save_dir) # Save model


    

    

    # Setup Trainer
    train_iter = chainer.iterators.SerialIterator(train_data, args.batchsize)
    test_iter = chainer.iterators.SerialIterator(test_data, args.batchsize,
                                                 repeat=False, shuffle=False)
    iter_per_epoch = len(train_data) // args.batchsize
    print('Number of iter/epoch =', iter_per_epoch)

    updater = training.StandardUpdater(
        train_iter, optimizer,
        converter=seq2seq_pad_concat_convert, device=args.gpu)

    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    # If you want to change a logging interval, change this number
    log_trigger = (min(1000, iter_per_epoch // 10), 'iteration')

    def floor_step(trigger):
        floored = trigger[0] - trigger[0] % log_trigger[0]
        if floored <= 0:
            floored = trigger[0]
        return (floored, trigger[1])

    # Validation every half epoch
    # TAKES TOO MUCH TIME FOR VALIDATION
    eval_trigger = floor_step((100, 'epoch'))
    #record_trigger = training.triggers.MinValueTrigger(
    #    'val/main/perp', eval_trigger)

    evaluator = extensions.Evaluator(
        test_iter, model,
        converter=seq2seq_pad_concat_convert, device=args.gpu)
    evaluator.default_name = 'val'
    trainer.extend(evaluator, trigger=eval_trigger)

    # Use Vaswan's magical rule of learning rate(Eq. 3 in the paper)
    # But, the hyperparamter in the paper seems to work well
    # only with a large batchsize.
    # If you run on popular setup (e.g. size=48 on 1 GPU),
    # you may have to change the hyperparamter.
    # I scaled learning rate by 0.5 consistently.
    # ("scale" is always multiplied to learning rate.)

    # If you use a shallow layer network (<=2),
    # you may not have to change it from the paper setting.
    if not args.use_fixed_lr:
        trainer.extend(
            # VaswaniRule('alpha', d=args.unit, warmup_steps=4000, scale=1.),
            # VaswaniRule('alpha', d=args.unit, warmup_steps=32000, scale=1.),
            # VaswaniRule('alpha', d=args.unit, warmup_steps=4000, scale=0.5),
            # VaswaniRule('alpha', d=args.unit, warmup_steps=16000, scale=1.),
            VaswaniRule('alpha', d=args.unit, warmup_steps=64000, scale=1.),
            trigger=(1, 'iteration'))
    observe_alpha = extensions.observe_value(
        'alpha',
        lambda trainer: trainer.updater.get_optimizer('main').alpha)
    trainer.extend(
        observe_alpha,
        trigger=(1, 'iteration'))

    # Only if a model gets best validation score,
    # save (overwrite) the model
    trainer.extend(extensions.snapshot_object(
        model, 'model_iter_{.updater.iteration}.npz'),
        trigger=(1000, 'iteration'))



if __name__ == '__main__':
    main()
