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

from bert import BERT, BERTLM
from dataset import STDataset
from build_vocab import WordVocab
import numpy as np
import utils
PAD = 0

class STTrainer:
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
        self.model = BERTLM(bert, vocab_size).to(self.device)

        if self.device == 'cuda':
            self.model = nn.DataParallel(self.model, gpu_ids)

        self.train_data = train_dataloader
        self.test_data = test_dataloader
        self.optim = optim
        self.criterion = nn.NLLLoss()

        self.log_freq = log_freq
        self.vocab = vocab
        

    def train(self, epoch):
        ret = self.iteration(epoch, self.train_data)
        return ret

    def test(self, epoch):
        ret = self.iteration(epoch, self.test_data, train=False)
        return ret

    def iteration(self, epoch, data_loader, train=True):
        """
        :param epoch: 現在のepoch
        :param data_loader: torch.utils.data.DataLoader
        :param train: trainかtestかのbool値
        """
        str_code = "train" if train else "test"
        data_iter = tqdm(enumerate(data_loader), desc="EP_%s:%d" % (str_code, epoch), total=len(data_loader), bar_format="{l_bar}{r_bar}")

        avg_loss = 0.0
        total_correct_1 = 0
        total_correct_2 = 0
        total_element = 0

        for i, data in data_iter:
            data = {key: value.to(self.device) for key, value in data.items()}
            tsm, msm = self.model.forward(data["bert_input"], data["segment_embd"])
            loss_tsm = self.criterion(tsm, data["is_same"])
            loss_msm = self.criterion(msm.transpose(1, 2), data["bert_label"])
            filleds = utils.sample(msm)
            smiles = []
            for filled in filleds:
                s1, s2 = self.num2str(filled)
                smiles.append(s1)
                smiles.append(s2)
            loss_val = utils.loss_validity(smiles)
            loss = loss_tsm + loss_msm + loss_val
            if train:
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

            # TSM prediction accuracy
            avg_loss += loss.item()
            correct1 = tsm.argmax(dim=-1).eq(data["is_same"]).sum().item()
            total_correct_1 += correct1
            correct2 = filleds.eq(data['bert_label']).sum().item() / 220
            total_correct_2 += correct2
            total_element += data["is_same"].nelement()
        return  avg_loss/len(data_iter), total_correct_1*100.0/total_element, total_correct_2*100.0/total_element 
    
    def num2str(self, nums):
        s = [self.vocab.itos[num] for num in nums]
        s = ''.join(s).replace('<pad>', '')
        ss = s.split('<eos>')
        if len(ss)>=2:
            return ss[0], s[1]
        else:
            sep = len(s)//2
            return s[:sep], s[sep:]

    
def get_trainer(trial, args, vocab, train_data_loader, test_data_loader):
    hidden = 256
    n_layers = [2, 3, 4, 5, 6, 7, 8]
    n_layer = trial.suggest_categorical('n_layer', n_layers)
    n_heads = [2, 4, 8]
    n_head = trial.suggest_categorical('n_head', n_heads)

    vocab_size = len(vocab)
    bert = BERT(vocab_size, hidden=hidden, n_layers=n_layer, attn_heads=n_head, dropout=args.dropout)
    bert.cuda()

    lr = trial.suggest_loguniform('lr', 1e-6, 1e-3)
    final_lr = trial.suggest_loguniform('final_lr', 1e-4, 1e-1)
    optim = AdaBound(BERTLM(bert, vocab_size).parameters(), lr=lr, final_lr=final_lr)

    trainer = STTrainer(optim, bert, vocab_size, train_dataloader=train_data_loader, test_dataloader=test_data_loader,
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
    parser.add_argument('--seq_len', type=int, default=220, help='maximum length of the paired seqence')
    parser.add_argument('--batch_size', '-b', type=int, default=16, help='batch size')
    parser.add_argument('--n_worker', '-w', type=int, default=16, help='number of workers')
    parser.add_argument('--dropout', '-d', type=float, default=0.1, help='dropout rate')
    parser.add_argument('--beta1', type=float, default=0.9, help='Adam beta1')
    parser.add_argument('--beta2', type=float, default=0.999, help='Adam beta2')
    parser.add_argument('--weight-decay', type=float, default=0.01, help='dropout rate')
    parser.add_argument('--gpu', metavar='N', type=int, nargs='+', help='list of GPU IDs to use')
    args = parser.parse_args()

    vocab = WordVocab.load_vocab(args.vocab)
    train_dataset = STDataset(args.train_data, vocab, seq_len=args.seq_len)
    test_dataset = STDataset(args.test_data, vocab, seq_len=args.seq_len)
    train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.n_worker)
    test_data_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.n_worker)

    def objective(trial):
        trainer = get_trainer(trial, args, vocab, train_data_loader, test_data_loader)
        for epoch in tqdm(range(args.n_epoch)):
            loss, acc1, acc2 = trainer.train(epoch)            
            loss, acc1, acc2 = trainer.test(epoch)
        print('2SM, MSM:', acc1, acc2)
        return loss

    study = optuna.create_study()
    study.optimize(objective, n_trials=args.n_trial)

if __name__=='__main__':
    main()