import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm
from adabound import AdaBound

from bert import BERT, BERTMSM
from dataset import MSMDataset
from build_vocab import WordVocab
import numpy as np
import utils

PAD = 0

class MSMTrainer:
    def __init__(self, bert, vocab_size, 
                 lr=1e-4, betas=(0.9, 0.999), final_lr=0.1, lr_decay=2,
                 log_freq=100, gpu_ids=[], vocab=None):
        """
        :param bert: BERT model
        :param vocab_size: vocabに含まれるトータルの単語数
        :param train_dataloader: train dataset data loader
        :param test_dataloader: test dataset data loader [can be None]
        :param lr: 学習率
        :param betas: Adam optimizer betasm
        :param with_cuda: traning with cuda
        :param log_freq: logを表示するiterationの頻度
        """

        # GPU環境において、GPUを指定しているかのフラグ
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.bert = bert
        self.model = BERTMSM(bert, vocab_size).to(self.device)

        if self.device == 'cuda':
            self.model = nn.DataParallel(self.model, gpu_ids)

        self.optim = AdaBound(self.model.parameters(), lr=lr, final_lr=final_lr)
        self.scheduler = lr_scheduler.StepLR(self.optim, lr_decay, gamma=0.1) # multiply 0.1 by lr every 2 epochs
        self.criterion = nn.NLLLoss()
        self.log_freq = log_freq
        self.vocab = vocab
        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))

    def iteration(self, it, data, rate, train=True):
        """
        :param epoch: 現在のepoch
        :param data_loader: torch.utils.data.DataLoader
        :param train: trainかtestかのbool値
        """
        data = {key: value.to(self.device) for key, value in data.items()}
        msm = self.model.forward(data["bert_input"], data["segment_embd"])
        loss = self.criterion(msm.transpose(1, 2), data["bert_label"])
        filleds = utils.sample(msm)
        smiles = []
        for filled in filleds:
            s1, s2 = self.num2str(filled)
            smiles.append(s1)
            smiles.append(s2)
        validity = utils.validity(smiles) * 100
        if train:
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

        # TSM prediction accuracy
        n = data["bert_input"].nelement() # batch_size * 220
        acc_msm = filleds.eq(data['bert_label']).sum().item() / n * 100

        if it % self.log_freq == 0:
            print('Iter: {:d}, Rate: {:.3f},  Loss: {:.3f}, Acc: {:.3f}, Validity: {:.3f}'.format(it, rate, loss.item(), acc_msm, validity))
            print(''.join([self.vocab.itos[j] for j in data['bert_input'][0]]).replace('<pad>', ' ').replace('<mask>', '?').replace('<eos>', '!').replace('<sos>', '!'))
            print(''.join([self.vocab.itos[j] for j in data['bert_label'][0]]).replace('<pad>', ' ').replace('<eos>', '!').replace('<sos>', '!'))
            tmp = utils.sample(msm)[0]
            print(''.join([self.vocab.itos[j] for j in tmp]).replace('<pad>', ' ').replace('<eos>', '!').replace('<sos>', '!'))
            print('')
    
        return loss.item(), acc_msm, validity

    def save(self, it, save_dir):
        """
        Saving the current BERT model on file_path

        :param epoch: current epoch number
        :param file_path: model output path which gonna be file_path+"ep%d" % epoch
        :return: final_output_path
        """
        output_path = save_dir + '/it{:06}.pkl'.format(it)
        torch.save(self.bert.state_dict(), output_path)
        self.bert.to(self.device)

    def num2str(self, nums):
        s = [self.vocab.itos[num] for num in nums]
        s = ''.join(s).replace('<pad>', '')
        ss = s.split('<eos>')
        if len(ss)>=2:
            return ss[0], s[1]
        else:
            sep = len(s)//2
            return s[:sep], s[sep:]

def main():
    parser = argparse.ArgumentParser(description='Pretrain SMILES Transformer')
    parser.add_argument('--n_epoch', '-e', type=int, default=100, help='number of epochs')
    parser.add_argument('--vocab', '-v', type=str, default='data/vocab.pkl', help='vocabulary (.pkl)')
    parser.add_argument('--train_data', type=str, default='data/chembl24_bert_train.csv', help='train corpus (.csv)')
    parser.add_argument('--test_data', type=str, default='data/chembl24_bert_test.csv', help='test corpus (.csv)')
    parser.add_argument('--out-dir', '-o', type=str, default='../result', help='output directory')
    parser.add_argument('--name', '-n', type=str, default='ST', help='model name')
    parser.add_argument('--seq_len', type=int, default=220, help='maximum length of the paired seqence')
    parser.add_argument('--batch_size', '-b', type=int, default=16, help='batch size')
    parser.add_argument('--n_worker', '-w', type=int, default=16, help='number of workers')
    parser.add_argument('--hidden', type=int, default=256, help='length of hidden vector')
    parser.add_argument('--n_layer', '-l', type=int, default=8, help='number of layers')
    parser.add_argument('--n_head', type=int, default=8, help='number of attention heads')
    parser.add_argument('--dropout', '-d', type=float, default=0.1, help='dropout rate')
    parser.add_argument('--lr', type=float, default=1e-3, help='AdaBound learning rate')
    parser.add_argument('--beta1', type=float, default=0.9, help='AdaBound beta1')
    parser.add_argument('--beta2', type=float, default=0.999, help='AdaBound beta2')
    parser.add_argument('--final-lr', type=float, default=0.01, help='AdaBound final lr')
    parser.add_argument('--lr-decay', type=int, default=50000, help='lr decay step size')
    parser.add_argument('--log-freq', type=int, default=100, help='log frequency')
    parser.add_argument('--gpu', metavar='N', type=int, nargs='+', help='list of GPU IDs to use')
    parser.add_argument('--checkpoint', '-c', type=str, default=None, help='Parameter to load')
    args = parser.parse_args()

    print("Loading Vocab", args.vocab)
    vocab = WordVocab.load_vocab(args.vocab)
    print("Building BERT model")
    bert = BERT(len(vocab), hidden=args.hidden, n_layers=args.n_layer, attn_heads=args.n_head, dropout=args.dropout)
    if args.checkpoint:
        print('Load', args.checkpoint)
        bert.load_state_dict(torch.load(args.checkpoint))
    bert.cuda()
    print("Creating BERT Trainer")
    trainer = MSMTrainer(bert, len(vocab),
                        lr=args.lr, betas=(args.beta1, args.beta2), final_lr=args.final_lr, lr_decay=args.lr_decay,
                        log_freq=args.log_freq, gpu_ids=args.gpu, vocab=vocab)

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    save_dir = os.path.join(args.out_dir, args.name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    log_dir = os.path.join(args.out_dir, 'log')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    with open(log_dir + '/' + args.name + '.csv', 'a') as f:
        f.write('iter,loss,acc_msm,acc_val\n')

    print("Training Start")
    rate = 0.05
    train_dataset = MSMDataset(args.train_data, vocab, seq_len=args.seq_len, rate=rate)
    train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.n_worker, shuffle=True)
    s = 0
    thres = (1 - 0.4*rate) * 100
    it = 0
    max_iter = 1000000
    while (it<=max_iter and rate<=0.5):
        for data in train_data_loader:
            trainer.scheduler.step() # LR scheduling
            loss, acc_msm, validity = trainer.iteration(it, data,  rate)
            if it % trainer.log_freq == 0:
                with open(log_dir + '/' + args.name + '.csv', 'a') as f:
                    f.write('{:d},{:.3f},{:.3f},{:.3f}\n'.format(it, loss, acc_msm, validity))
                if it % (trainer.log_freq*10) == 0:
                    trainer.save(it, save_dir) # Save model
            it += 1

            s = s*0.9 + acc_msm*0.1
            if s > thres: # Mask rate update
                rate += 0.01
                thres = (1 - 0.4*rate) * 100
                train_dataset = MSMDataset(args.train_data, vocab, seq_len=args.seq_len, rate=rate)
                train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.n_worker, shuffle=True)
                s = 0
                print('Mask rate: {:.2f} ,thres: {:.3f}'.format(rate, thres))
                break
            if it>max_iter:
                    break
            

if __name__=='__main__':
    main()