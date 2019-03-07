import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
import optuna
from adabound import AdaBound

from bert import BERT, BERTMSM
from dataset import MSMDataset
from build_vocab import WordVocab
import numpy as np
PAD = 0

class MyLoss(nn.Module):
    def __init__(self):
        super(MyLoss, self).__init__()
        self.nll = nn.NLLLoss()
        self.vocab_size = 45

    def forward(self, ys, ts):
        loss = 0
        for y,t in  zip(ys,ts):
            b = torch.masked_select(t, t==PAD)
            l = len(b)
            b = b.reshape(1,l)
            a = torch.masked_select(y, t==PAD).reshape(1,self.vocab_size,l)
            loss += self.nll(a, b)/2 # paddding loss
            b = torch.masked_select(t, t!=PAD)
            l = len(b)
            if l>0:
                b = b.reshape(1,l)
                a = torch.masked_select(y, t!=PAD).reshape(1,self.vocab_size,l)
                loss += self.nll(a, b) # Not padding loss

        return loss/len(ys)

class MSMTrainer:
    def __init__(self, optim, bert, vocab_size, train_dataloader, test_dataloader,
                 log_freq=10, gpu_ids=[], vocab=None):
        """
        :param bert: BERT model
        :param vocab_size: vocabに含まれるトータルの単語数
        :param train_dataloader: train dataset data loader
        :param test_dataloader: test dataset data loader [can be None]
        :param lr: 学習率
        :param betas: Adam optimizer betas
        :param weight_decay: Adam optimizer weight decay param
        :param with_cuda: traning with cuda
        :param log_freq: logを表示するiterationの頻度
        """

        # GPU環境において、GPUを指定しているかのフラグ
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.bert = bert
        self.model = BERTMSM(bert, vocab_size).to(self.device)

        if self.device == 'cuda':
            self.model = nn.DataParallel(self.model, gpu_ids)

        self.train_data = train_dataloader
        self.test_data = test_dataloader
        self.optim = optim
        #self.criterion = MyLoss()
        self.criterion = nn.NLLLoss()

        self.log_freq = log_freq
        self.vocab = vocab
        

    def train(self, epoch):
        loss = self.iteration(epoch, self.train_data)
        return loss

    def test(self, epoch):
        loss = self.iteration(epoch, self.test_data, train=False)
        return loss

    def iteration(self, epoch, data_loader, train=True):
        """
        :param epoch: 現在のepoch
        :param data_loader: torch.utils.data.DataLoader
        :param train: trainかtestかのbool値
        """
        str_code = "train" if train else "test"
        data_iter = tqdm(enumerate(data_loader), desc="EP_%s:%d" % (str_code, epoch), total=len(data_loader), bar_format="{l_bar}{r_bar}")

        avg_loss = 0.0

        for i, data in data_iter:
            data = {key: value.to(self.device) for key, value in data.items()}
            msm = self.model.forward(data["bert_input"], data["segment_embd"])
            loss = self.criterion(msm.transpose(1, 2), data["bert_label"]) 
            if train:
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

            avg_loss += loss.item()

            post_fix = {
                "epoch": epoch,
                "iter": i,
                "avg_loss": avg_loss / (i + 1),
                "loss": loss.item()
            }
        return  avg_loss/len(data_iter) # Total loss
    
def get_trainer(trial, args, vocab, train_data_loader, test_data_loader):
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
        lr = trial.suggest_loguniform('lr', 1e-6, 1e-1)
        optim = Adam(BERTMSM(bert, vocab_size).parameters(), lr=lr, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay)
    else:
        lr = trial.suggest_loguniform('lr', 1e-6, 1e-1)
        optim = AdaBound(BERTMSM(bert, vocab_size).parameters(), lr=lr, final_lr=0.1)

    
    trainer = MSMTrainer(optim, bert, vocab_size, train_dataloader=train_data_loader, test_dataloader=test_data_loader,
                        gpu_ids=args.gpu, vocab=vocab)
    return trainer




def main():
    parser = argparse.ArgumentParser(description='Pretrain SMILES Transformer')
    parser.add_argument('--n_epoch', '-e', type=int, default=300, help='number of epochs')
    parser.add_argument('--n_trial', '-t', type=int, default=100, help='number of optuna trials')
    parser.add_argument('--vocab', '-v', type=str, default='data/vocab.pkl', help='vocabulary (.pkl)')
    parser.add_argument('--train_data', type=str, default='data/chembl24_bert_train.csv', help='train corpus (.csv)')
    parser.add_argument('--test_data', type=str, default='data/chembl24_bert_test.csv', help='test corpus (.csv)')
    parser.add_argument('--name', '-n', type=str, default='ST', help='model name')
    parser.add_argument('--seq_len', type=int, default=203, help='maximum length of the paired seqence')
    parser.add_argument('--batch_size', '-b', type=int, default=16, help='batch size')
    parser.add_argument('--n_worker', '-w', type=int, default=16, help='number of workers')
    parser.add_argument('--dropout', '-d', type=float, default=0.1, help='dropout rate')
    parser.add_argument('--beta1', type=float, default=0.9, help='Adam beta1')
    parser.add_argument('--beta2', type=float, default=0.999, help='Adam beta2')
    parser.add_argument('--weight-decay', type=float, default=0.01, help='dropout rate')
    parser.add_argument('--gpu', metavar='N', type=int, nargs='+', help='list of GPU IDs to use')
    args = parser.parse_args()

    vocab = WordVocab.load_vocab(args.vocab)
    train_dataset = MSMDataset(args.train_data, vocab, seq_len=args.seq_len)
    test_dataset = MSMDataset(args.test_data, vocab, seq_len=args.seq_len)
    train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.n_worker)
    test_data_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.n_worker)

    def objective(trial):
        trainer = get_trainer(trial, args, vocab, train_data_loader, test_data_loader)
        for epoch in tqdm(range(args.n_epoch)):
            loss = trainer.train(epoch)            
            loss = trainer.test(epoch)
        return loss

    study = optuna.create_study()
    study.optimize(objective, n_trials=args.n_trial)

if __name__=='__main__':
    main()