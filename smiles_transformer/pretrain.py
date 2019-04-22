import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm
from adabound import AdaBound

from bert import BERT, BERTLM
from dataset import STDataset
from build_vocab import WordVocab
import numpy as np
import utils


PAD = 0

class STTrainer:
    def __init__(self, bert, vocab_size, train_dataloader, test_dataloader,
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
        self.model = BERTLM(bert, vocab_size).to(self.device)

        if self.device == 'cuda':
            self.model = nn.DataParallel(self.model, gpu_ids)

        self.train_data = train_dataloader
        self.test_data = test_dataloader

        self.optim = AdaBound(self.model.parameters(), lr=lr, final_lr=final_lr)
        self.scheduler = lr_scheduler.StepLR(self.optim, lr_decay, gamma=0.1) # multiply 0.1 by lr every 2 epochs
        self.criterion = nn.NLLLoss()
        self.log_freq = log_freq
        self.vocab = vocab
        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))

    def iteration(self, epoch, iter, data, data_iter, train=True):
        """
        :param epoch: 現在のepoch
        :param data_loader: torch.utils.data.DataLoader
        :param train: trainかtestかのbool値
        """
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
        validity = utils.validity(smiles) * 100
        loss = loss_tsm + loss_msm
        if train:
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

        # TSM prediction accuracy
        n = data["is_same"].nelement()
        acc_tsm = tsm.argmax(dim=-1).eq(data["is_same"]).sum().item()  / n * 100
        acc_msm = filleds.eq(data['bert_label']).sum().item() / 220  / n * 100
        post_fix = {
                'Epoch': epoch,
                'Iter': iter,
                'Loss total': '{:.3f}'.format(loss.item()),
                'Acc 2SM': '{:.3f}'.format(acc_tsm),
                'Acc MSM': '{:.3f}'.format(acc_msm),
                'Validity': '{:.3f}'.format(validity)
            }

        if iter % self.log_freq == 0:
            data_iter.write(str(post_fix))
            print('')
            print(''.join([self.vocab.itos[j] for j in data['bert_input'][0]]).replace('<pad>', ' ').replace('<mask>', '?').replace('<eos>', '!').replace('<sos>', '!'))
            print(''.join([self.vocab.itos[j] for j in data['bert_label'][0]]).replace('<pad>', ' ').replace('<eos>', '!').replace('<sos>', '!'))
            tmp = utils.sample(msm)[0]
            print(''.join([self.vocab.itos[j] for j in tmp]).replace('<pad>', ' ').replace('<eos>', '!').replace('<sos>', '!'))
            print('')
    
        return loss.item(), loss_tsm.item(), loss_msm.item(), acc_tsm, acc_msm, validity

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
    print("Loading Train Dataset", args.train_data)
    train_dataset = STDataset(args.train_data, vocab, seq_len=args.seq_len)
    print("Creating Dataloader")
    train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.n_worker, shuffle=True) 
    print("Building BERT model")
    bert = BERT(len(vocab), hidden=args.hidden, n_layers=args.n_layer, attn_heads=args.n_head, dropout=args.dropout)
    if args.checkpoint:
        bert.load_state_dict(torch.load(args.checkpoint))
    bert.cuda()
    print("Creating BERT Trainer")
    trainer = STTrainer(bert, len(vocab), train_dataloader=train_data_loader, test_dataloader=None,
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
        f.write('epoch,iter,loss_tot,loss_2sm,loss_msm,acc_2sm,acc_msm,acc_val\n')

    print("Training Start")
    for epoch in tqdm(range(args.n_epoch)):
        data_iter = tqdm(enumerate(train_data_loader), desc="EP_{:d}".format(epoch), total=len(train_data_loader), bar_format="{l_bar}{r_bar}")
        for iter, data in data_iter:
            trainer.scheduler.step() # LR scheduling
            loss, loss_tsm, loss_msm, acc_tsm, acc_msm, validity = trainer.iteration(epoch, iter, data, data_iter)
            if iter % trainer.log_freq == 0:
                with open(log_dir + '/' + args.name + '.csv', 'a') as f:
                    f.write('{:d},{:d},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f}\n'.format(epoch, iter, loss, loss_tsm, loss_msm, acc_tsm, acc_msm, validity))
                if iter % (trainer.log_freq*10) == 0:
                    trainer.save(epoch, iter, save_dir) # Save model

if __name__=='__main__':
    main()