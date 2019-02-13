import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from bert import BERT, BERTMSM
from dataset import MSMDataset
from build_vocab import WordVocab
import numpy as np

PAD = 0

class MyLoss(nn.Module):
    def __init__(self):
        super(MyLoss, self).__init__()

    def forward(self, ys, ts):
        nll = nn.NLLLoss()

        loss = 0
        for y,t in  zip(ys,ts):
            #print(y.shape, t.shape)
            b = torch.masked_select(t, t==PAD)
            l = len(b)
            b = b.reshape(1,l)
            a = torch.masked_select(y, t==PAD).reshape(1,45,l)
            loss_1 = nll(a, b) # paddding loss
            b = torch.masked_select(t, t!=PAD)
            l = len(b)
            if l>0:
                b = b.reshape(1,l)
                a = torch.masked_select(y, t!=PAD).reshape(1,45,l)
                loss_2 = nll(a, b) # Not padding loss
                loss = loss + loss_1/2 + loss_2
            else:
                loss = loss + loss_1
        return loss

class MSMTrainer:
    def __init__(self, bert: BERT, vocab_size: int,
                 train_dataloader: DataLoader, test_dataloader: DataLoader = None,
                 lr: float = 1e-4, betas=(0.9, 0.999), weight_decay: float = 0.01,
                 log_freq: int = 10, gpu_ids=[], vocab=None):
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

        self.optim = Adam(self.model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
        self.criterion = MyLoss()
        self.log_freq = log_freq
        self.vocab = vocab
        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))
        

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
            # 0. batch_dataはGPU or CPUに載せる
            data = {key: value.to(self.device) for key, value in data.items()}
            msm = self.model.forward(data["bert_input"], data["segment_embd"])
            loss = self.criterion(msm.transpose(1, 2), data["bert_label"]) 
            if train:
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

            # TSM prediction accuracy
            avg_loss += loss.item()

            post_fix = {
                "epoch": epoch,
                "iter": i,
                "avg_loss": avg_loss / (i + 1),
                "loss": loss.item()
            }
            if i % self.log_freq == 0:
                data_iter.write(str(post_fix))
                print('*'*10)  
                print(''.join([self.vocab.itos[j] for j in data['bert_input'][0]]).replace('<pad>', ' ').replace('<mask>', '?').replace('<eos>', '!').replace('<sos>', '!'))
                print(''.join([self.vocab.itos[j] for j in data['bert_label'][0]]).replace('<pad>', ' ').replace('<eos>', '!').replace('<sos>', '!'))
                tmp = np.argmax(msm.transpose(1, 2)[0].detach().cpu().numpy(), axis=0) 
                print(''.join([self.vocab.itos[j] for j in tmp]).replace('<pad>', ' ').replace('<eos>', '!').replace('<sos>', '!'))
                print('*'*10)
        return  avg_loss/len(data_iter) # Total loss

    def save(self, epoch, save_dir):
        """
        Saving the current BERT model on file_path

        :param epoch: current epoch number
        :param file_path: model output path which gonna be file_path+"ep%d" % epoch
        :return: final_output_path
        """
        output_path = save_dir + '/ep_{:02}.pkl'.format(epoch)
        torch.save(self.bert.state_dict(), output_path)
        self.bert.to(self.device)
        print("EP:%d Model Saved on:" % epoch, output_path)

def main():
    parser = argparse.ArgumentParser(description='Pretrain SMILES Transformer')
    parser.add_argument('--n_epoch', '-e', type=int, default=100, help='number of epochs')
    parser.add_argument('--vocab', '-v', type=str, default='data/vocab.pkl', help='vocabulary (.pkl)')
    parser.add_argument('--train_data', type=str, default='data/chembl24_bert_train.csv', help='train corpus (.csv)')
    parser.add_argument('--test_data', type=str, default='data/chembl24_bert_test.csv', help='test corpus (.csv)')
    parser.add_argument('--out-dir', '-o', type=str, default='../result', help='output directory')
    parser.add_argument('--name', '-n', type=str, default='MSM', help='model name')
    parser.add_argument('--seq_len', type=int, default=203, help='maximum length of the paired seqence')
    parser.add_argument('--batch_size', '-b', type=int, default=256, help='batch size')
    parser.add_argument('--n_worker', '-w', type=int, default=16, help='number of workers')
    parser.add_argument('--hidden', type=int, default=256, help='length of hidden vector')
    parser.add_argument('--n_layer', '-l', type=int, default=4, help='number of layers')
    parser.add_argument('--n_head', type=int, default=4, help='number of attention heads')
    parser.add_argument('--dropout', '-d', type=float, default=0.1, help='dropout rate')
    parser.add_argument('--lr', type=float, default=1e-3, help='Adam learning rate')
    parser.add_argument('--beta1', type=float, default=0.9, help='Adam beta1')
    parser.add_argument('--beta2', type=float, default=0.999, help='Adam beta2')
    parser.add_argument('--weight-decay', type=float, default=0.01, help='dropout rate')
    parser.add_argument('--log-freq', type=int, default=100, help='log frequency')
    parser.add_argument('--gpu', metavar='N', type=int, nargs='+', help='list of GPU IDs to use')
    parser.add_argument('--checkpoint', '-c', type=str, default=None, help='Parameter to load')
    args = parser.parse_args()

    print("Loading Vocab", args.vocab)
    vocab = WordVocab.load_vocab(args.vocab)
    print("Loading Train Dataset", args.train_data)
    train_dataset = MSMDataset(args.train_data, vocab, seq_len=args.seq_len)
    print("Loading Test Dataset", args.test_data)
    test_dataset = MSMDataset(args.test_data, vocab, seq_len=args.seq_len)
    print("Creating Dataloader")
    train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.n_worker)
    test_data_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.n_worker) 
    print("Building BERT model")
    bert = BERT(len(vocab), hidden=args.hidden, n_layers=args.n_layer, attn_heads=args.n_head, dropout=args.dropout)
    if args.checkpoint:
        bert.load_state_dict(torch.load(args.checkpoint))
    bert.cuda()
    print("Creating BERT Trainer")
    trainer = MSMTrainer(bert, len(vocab), train_dataloader=train_data_loader, test_dataloader=test_data_loader,
                        lr=args.lr, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay,
                        log_freq=args.log_freq, gpu_ids=args.gpu, vocab=train_dataset.vocab)

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    save_dir = os.path.join(args.out_dir, args.name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    log_dir = os.path.join(args.out_dir, 'log')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    with open(log_dir + '/' + args.name + '.csv', 'a') as f:
        f.write('epoch,train_loss,test_loss\n')

    print("Training Start")
    for epoch in tqdm(range(args.n_epoch)):
        loss = trainer.train(epoch)
        print("EP%d Train, loss=" % (epoch), loss)
        with open(log_dir + '/' + args.name + '.csv', 'a') as f:
            f.write('%d,%f,' %(epoch, loss))
    
        trainer.save(epoch, save_dir) # Save model
        
        loss = trainer.test(epoch)
        print("EP%d Test, loss=" % (epoch), loss)
        with open(log_dir + '/' + args.name + '.csv', 'a') as f:
            f.write('%f\n' %(loss))

if __name__=='__main__':
    main()