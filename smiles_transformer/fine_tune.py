import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, lr_scheduler
from torch.utils.data import DataLoader
from sklearn.metrics import r2_score, mean_squared_error
from tqdm import tqdm
from adabound import AdaBound

from bert import BERT, BERTReg
from dataset import ESOLDataset
from build_vocab import WordVocab
import numpy as np
import utils


PAD = 0

class FTTrainer:
    def __init__(self, bert, vocab_size, train_dataloader, test_dataloader,
                 lr=1e-4, betas=(0.9, 0.999), final_lr=0.1,
                 gpu_ids=[], vocab=None):
        """
        :param bert: BERT model
        :param vocab_size: vocabに含まれるトータルの単語数
        :param train_dataloader: train dataset data loader
        :param test_dataloader: test dataset data loader [can be None]
        :param lr: 学習率
        :param betas: Adam optimizer betasm
        :param with_cuda: traning with cuda
        """

        # GPU環境において、GPUを指定しているかのフラグ
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.bert = bert
        self.model = BERTReg(bert).to(self.device)

        if self.device == 'cuda':
            self.model = nn.DataParallel(self.model, gpu_ids)

        self.train_data = train_dataloader
        self.test_data = test_dataloader

        self.optim = AdaBound(self.model.parameters(), lr=lr, final_lr=final_lr)
        self.criterion = nn.MSELoss()
        self.vocab = vocab
        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))

    def iteration(self, epoch, iter, data, data_iter, train=True):
        """
        :param epoch: 現在のepoch
        :param data_loader: torch.utils.data.DataLoader
        :param train: trainかtestかのbool値
        """
        data = {key: value.to(self.device) for key, value in data.items()}
        pred = self.model.forward(data["bert_input"], data["segment_embd"])
        loss = self.criterion(pred, data["target"])
        if train:
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
        return loss.item()

    def save(self, epoch, iter, save_dir):
        """
        Saving the current BERT model on file_path

        :param epoch: current epoch number
        :param file_path: model output path which gonna be file_path+"ep%d" % epoch
        :return: final_output_path
        """
        output_path = save_dir + '/ep{:02}_it{:06}.pkl'.format(epoch, iter)
        torch.save(self.bert.state_dict(), output_path)
        self.bert.to(self.device)


def main():
    parser = argparse.ArgumentParser(description='Pretrain SMILES Transformer')
    parser.add_argument('--n_epoch', '-e', type=int, default=10, help='number of epochs')
    parser.add_argument('--vocab', '-v', type=str, default='data/vocab.pkl', help='vocabulary (.pkl)')
    parser.add_argument('--train_data', type=str, default='data/ESOL_train.csv', help='train corpus (.csv)')
    parser.add_argument('--test_data', type=str, default='data/ESOL_test.csv', help='test corpus (.csv)')
    parser.add_argument('--out-dir', '-o', type=str, default='../result', help='output directory')
    parser.add_argument('--name', '-n', type=str, default='ST', help='model name')
    parser.add_argument('--seq_len', type=int, default=220, help='maximum length of the paired seqence')
    parser.add_argument('--batch_size', '-b', type=int, default=64, help='batch size')
    parser.add_argument('--n_worker', '-w', type=int, default=16, help='number of workers')
    parser.add_argument('--hidden', type=int, default=256, help='length of hidden vector')
    parser.add_argument('--n_layer', '-l', type=int, default=8, help='number of layers')
    parser.add_argument('--n_head', type=int, default=8, help='number of attention heads')
    parser.add_argument('--dropout', '-d', type=float, default=0.1, help='dropout rate')
    parser.add_argument('--lr', type=float, default=2e-5, help='AdaBound learning rate')
    parser.add_argument('--beta1', type=float, default=0.9, help='AdaBound beta1')
    parser.add_argument('--beta2', type=float, default=0.999, help='AdaBound beta2')
    parser.add_argument('--final-lr', type=float, default=2e-5, help='AdaBound final lr')
    parser.add_argument('--gpu', metavar='N', type=int, nargs='+', help='list of GPU IDs to use')
    parser.add_argument('--checkpoint', '-c', type=str, default='../result/chembl/ep09_it013000.pkl', help='Parameter to load')
    args = parser.parse_args()

    print("Loading Vocab", args.vocab)
    vocab = WordVocab.load_vocab(args.vocab)
    print("Loading Train Dataset", args.train_data)
    train_dataset = ESOLDataset(args.train_data, vocab, seq_len=args.seq_len)
    print("Creating Dataloader")
    train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.n_worker, shuffle=True) 
    print("Loading Test Dataset", args.test_data)
    test_dataset = ESOLDataset(args.test_data, vocab, seq_len=args.seq_len)
    print("Creating Dataloader")
    test_data_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.n_worker, shuffle=False) 
    print("Building BERT model")
    bert = BERT(len(vocab), hidden=args.hidden, n_layers=args.n_layer, attn_heads=args.n_head, dropout=args.dropout)
    if args.checkpoint:
        bert.load_state_dict(torch.load(args.checkpoint))
    bert.cuda()
    print("Creating BERT Trainer")
    trainer = FTTrainer(bert, len(vocab), train_dataloader=train_data_loader, test_dataloader=None,
                        lr=args.lr, betas=(args.beta1, args.beta2), final_lr=args.final_lr,
                        gpu_ids=args.gpu, vocab=vocab)

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    save_dir = os.path.join(args.out_dir, args.name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print("Training Start")
    for epoch in tqdm(range(args.n_epoch)):
        data_iter = tqdm(enumerate(train_data_loader), desc="EP_{:d}".format(epoch), total=len(train_data_loader), bar_format="{l_bar}{r_bar}")
        for iter, data in data_iter:
            loss = trainer.iteration(epoch, iter, data, data_iter)
            print("Iter: {:d}, MSE Loss: {:.4f}".format(iter, loss))

    se = 0
    for data in test_data_loader:
        data = {key: value.to(self.device) for key, value in data.items()}
        pred = self.model.forward(data["bert_input"], data["segment_embd"])
        se += mean_squared_error(pred, data['target'])*len(pred)
    print("RMSE: {:.4f}".format(se/len(test_dataset)**0.5))
            

if __name__=='__main__':
    main()