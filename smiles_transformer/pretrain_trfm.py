import argparse
import numpy
from tqdm import tqdm
from torch.utils.data import DataLoader
import os
import math
import argparse
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.autograd import Variable
from build_vocab import WordVocab
from dataset import Seq2seqDataset

PAD = 0
UNK = 1
EOS = 2
SOS = 3
MASK = 4

class PositionalEncoding(nn.Module):
    "Implement the PE function. No batch support?"
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model) # (T,H)
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], 
                         requires_grad=False)
        return self.dropout(x)

class TrfmSeq2seq(nn.Module):
    def __init__(self, in_size, hidden_size, out_size, n_layers, dropout=0.1):
        super(TrfmSeq2seq, self).__init__()
        self.in_size = in_size
        self.hidden_size = hidden_size
        self.embed = nn.Embedding(in_size, hidden_size)
        self.pe = PositionalEncoding(hidden_size, dropout)
        self.trfm = nn.Transformer(d_model=hidden_size, nhead=4, num_encoder_layers=4, num_decoder_layers=4, dim_feedforward=256)
        self.out = nn.Linear(hidden_size, out_size)

    def forward(self, src, hidden=None):
        # src: (T,B)
        embedded = self.embed(src)  # (T,B,H)
        embedded = self.pe(embedded) # (T,B,H)
        hidden = self.trfm(embedded, embedded) # (T,B,H)
        out = self.out(hidden) # (T,B,V)
        out = F.log_softmax(out, dim=2) # (T,B,V)
        return out # (T,B, V)

def parse_arguments():
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--n_epoch', '-e', type=int, default=5, help='number of epochs')
    parser.add_argument('--vocab', '-v', type=str, default='data/vocab.pkl', help='vocabulary (.pkl)')
    parser.add_argument('--train_data', type=str, default='data/chembl24_bert_train.csv', help='train corpus (.csv)')
    parser.add_argument('--test_data', type=str, default='data/chembl24_bert_test.csv', help='test corpus (.csv)')
    parser.add_argument('--out-dir', '-o', type=str, default='../result', help='output directory')
    parser.add_argument('--name', '-n', type=str, default='ST', help='model name')
    parser.add_argument('--seq_len', type=int, default=220, help='maximum length of the paired seqence')
    parser.add_argument('--batch_size', '-b', type=int, default=32, help='batch size')
    parser.add_argument('--n_worker', '-w', type=int, default=16, help='number of workers')
    parser.add_argument('--hidden', type=int, default=256, help='length of hidden vector')
    parser.add_argument('--n_layer', '-l', type=int, default=4, help='number of layers')
    parser.add_argument('--n_head', type=int, default=8, help='number of attention heads')
    parser.add_argument('--dropout', '-d', type=float, default=0.1, help='dropout rate')
    parser.add_argument('--lr', type=float, default=1e-4, help='Adam learning rate')
    parser.add_argument('--gpu', metavar='N', type=int, nargs='+', help='list of GPU IDs to use')
    return parser.parse_args()


def evaluate(model, val_loader, vocab):
    model.eval()
    total_loss = 0
    for b, data in enumerate(val_loader):
        data = Variable(data.cuda())
        with torch.no_grad():
            output = model(data, data) # (T,B)
        loss = F.nll_loss(output[1:].view(-1, len(vocab)),
                               data[1:].contiguous().view(-1),
                               ignore_index=PAD)
        total_loss += loss.item()
    return total_loss / len(val_loader)

def main():
    args = parse_arguments()
    assert torch.cuda.is_available()

    vocab = WordVocab.load_vocab(args.vocab)
    print("[!] Instantiating models...")
    model = TrfmSeq2seq(len(vocab), args.hidden, len(vocab), args.n_layer).cuda()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    train_dataset = Seq2seqDataset(args.train_data, vocab)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.n_worker)
    val_dataset = Seq2seqDataset(args.test_data, vocab, is_train=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.n_worker)
    print(model)
    print('Total parameters:', sum(p.numel() for p in model.parameters()))

    best_loss = None
    for e in range(1, args.n_epoch):
        for b,data in tqdm(enumerate(train_loader)):
            data = torch.t(data.cuda()) # (T,B)
            optimizer.zero_grad()
            output = model(data, data) # (T,B,V)
            loss = F.nll_loss(output[1:].view(-1, len(vocab)),
                    data[1:].contiguous().view(-1), ignore_index=PAD)
            loss.backward()
            optimizer.step()
            if b%100==0:
                print('Train {:3d}: iter {:5d} | loss {:.3f} | ppl {:.3f}'.format(e, b, loss.item(), math.exp(loss.item())))
            if b%1000==0:
                loss = evaluate(model, val_loader, vocab)
                print('Val {:3d}: iter {:5d} | loss {:.3f} | ppl {:.3f}'.format(e, b, loss, math.exp(loss)))
                # Save the model if the validation loss is the best we've seen so far.
                if not best_loss or loss < best_loss:
                    print("[!] saving model...")
                    if not os.path.isdir(".save"):
                        os.makedirs(".save")
                    torch.save(model.state_dict(), './.save/trfm_%d_%d.pkl' % (e,b))
                    best_loss = loss


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt as e:
        print("[STOP]", e)
