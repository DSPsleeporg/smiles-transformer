import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm
import optuna
from adabound import AdaBound

from bert import BERT, BERTMSM
from dataset import MSMDataset
from build_vocab import WordVocab
import numpy as np
import utils

PAD = 0

class MSMTrainer:
    def __init__(self, optim, bert, vocab_size, gpu_ids=[], vocab=None):
        # GPU環境において、GPUを指定しているかのフラグ
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.bert = bert
        self.model = BERTMSM(bert, vocab_size).to(self.device)

        if self.device == 'cuda':
            self.model = nn.DataParallel(self.model, gpu_ids)

        self.optim = optim
        self.criterion = nn.NLLLoss()
        self.vocab = vocab

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
        if train:
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

        # TSM prediction accuracy
        n = data["bert_input"].nelement() # batch_size * 220
        acc_msm = filleds.eq(data['bert_label']).sum().item() / n * 100
    
        return loss.item(), acc_msm

    def num2str(self, nums):
        s = [self.vocab.itos[num] for num in nums]
        s = ''.join(s).replace('<pad>', '')
        ss = s.split('<eos>')
        if len(ss)>=2:
            return ss[0], s[1]
        else:
            sep = len(s)//2
            return s[:sep], s[sep:]

    
def get_trainer(trial, args, vocab):
    hiddens = [128, 256, 512, 1024]
    hidden = trial.suggest_categorical('hidden', hiddens)
    n_layers = [2, 3, 4, 6, 8]
    n_layer = trial.suggest_categorical('n_layer', n_layers)
    n_heads = [2, 4, 8]
    n_head = trial.suggest_categorical('n_head', n_heads)

    vocab_size = len(vocab)
    bert = BERT(vocab_size, hidden=hidden, n_layers=n_layer, attn_heads=n_head, dropout=args.dropout)
    bert.cuda()

    optims = ['Adam', 'AdaBound']
    optim_name = trial.suggest_categorical('optimizer', optims)
    if optim_name=='Adam':
        lr = trial.suggest_loguniform('lr', 1e-6, 1e-3)
        optim = Adam(BERTMSM(bert, vocab_size).parameters(), lr=lr, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay)
    else:
        lr = trial.suggest_loguniform('lr', 1e-6, 1e-3)
        optim = AdaBound(BERTMSM(bert, vocab_size).parameters(), lr=lr, final_lr=0.1)

    
    trainer = MSMTrainer(optim, bert, vocab_size, gpu_ids=args.gpu, vocab=vocab)
    return trainer




def main():
    parser = argparse.ArgumentParser(description='Pretrain SMILES Transformer')
    parser.add_argument('--n_trial', '-t', type=int, default=100, help='number of optuna trials')
    parser.add_argument('--vocab', '-v', type=str, default='data/vocab.pkl', help='vocabulary (.pkl)')
    parser.add_argument('--train_data', type=str, default='data/chembl24_bert_train.csv', help='train corpus (.csv)')
    parser.add_argument('--test_data', type=str, default='data/chembl24_bert_test.csv', help='test corpus (.csv)')
    parser.add_argument('--name', '-n', type=str, default='ST', help='model name')
    parser.add_argument('--seq_len', type=int, default=220, help='maximum length of the paired seqence')
    parser.add_argument('--batch_size', '-b', type=int, default=64, help='batch size')
    parser.add_argument('--n_worker', '-w', type=int, default=16, help='number of workers')
    parser.add_argument('--dropout', '-d', type=float, default=0.1, help='dropout rate')
    parser.add_argument('--beta1', type=float, default=0.9, help='Adam beta1')
    parser.add_argument('--beta2', type=float, default=0.999, help='Adam beta2')
    parser.add_argument('--weight-decay', type=float, default=0.01, help='dropout rate')
    parser.add_argument('--gpu', metavar='N', type=int, nargs='+', help='list of GPU IDs to use')
    args = parser.parse_args()

    vocab = WordVocab.load_vocab(args.vocab)
    
    def objective(trial):
        trainer = get_trainer(trial, args, vocab)
        rate = 0.05
        train_dataset = MSMDataset(args.train_data, vocab, seq_len=args.seq_len, rate=rate)
        train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.n_worker, shuffle=True)
        s = 0
        thres = (1 - 0.4*rate) * 100
        it = 0
        max_iter = 10000
        while (it<=max_iter and rate<=0.5):
            for data in train_data_loader:
                loss, acc_msm = trainer.iteration(it, data,  rate)
                print(it)
                it += 1
                s = s*0.9 + acc_msm*0.1
                if s > thres: # Mask rate update
                    rate += 0.01
                    thres = (1 - 0.4*rate) * 100
                    train_dataset = MSMDataset(args.train_data, vocab, seq_len=args.seq_len, rate=rate)
                    train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.n_worker, shuffle=True)
                    s = 0
                    break
                if it>max_iter:
                    break

        return -rate+loss/100
    
    study = optuna.create_study()
    study.optimize(objective, n_trials=args.n_trial)
    df = study.trials_dataframe()
    df.to_csv('../results/log/optuna_msm.csv')

if __name__=='__main__':
    main()