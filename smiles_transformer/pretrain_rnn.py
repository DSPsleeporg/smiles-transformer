import argparse
import numpy
from tqdm import tqdm
from torch.utils.data import DataLoader
import os
import math
import argparse
import torch
from torch import optim
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm_
from torch.nn import functional as F
from rnn import Encoder, Decoder, Seq2Seq
from build_vocab import WordVocab
from dataset import Seq2seqDataset

PAD = 0
UNK = 1
EOS = 2
SOS = 3
MASK = 4

def parse_arguments():
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--n_epoch', '-e', type=int, default=10, help='number of epochs')
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
    parser.add_argument('--lr', type=float, default=1e-4, help='Adam learning rate')
    parser.add_argument('--lr-decay', type=int, default=50000, help='lr decay step size')
    parser.add_argument('--log-freq', type=int, default=100, help='log frequency')
    parser.add_argument('--gpu', metavar='N', type=int, nargs='+', help='list of GPU IDs to use')
    parser.add_argument('--checkpoint', '-c', type=str, default=None, help='Parameter to load')
    parser.add_argument('-grad_clip', type=float, default=10.0, help='in case of gradient explosion')
    return parser.parse_args()


def evaluate(model, val_loader, vocab):
    model.eval()
    total_loss = 0
    for b, data in tqdm(enumerate(val_loader)):
        data = Variable(data.cuda())
        with torch.no_grad():
            output = model(data, data, teacher_forcing_ratio=0.0)
        loss = F.nll_loss(output[1:].view(-1, len(vocab)),
                               data[1:].contiguous().view(-1),
                               ignore_index=PAD)
        total_loss += loss.item()
    return total_loss / len(val_loader)


def train(model, optimizer, train_loader, vocab, grad_clip):
    model.train()
    total_loss = 0
    for b,data in tqdm(enumerate(train_loader)):
        data = torch.t(data.cuda()) # (T,B)
        optimizer.zero_grad()
        output = model(data, data) # (T,B,V)
        loss = F.nll_loss(output[1:].view(-1, len(vocab)),
                data[1:].contiguous().view(-1), ignore_index=PAD)
        loss.backward()
        clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += loss.item()

        if b % 100 == 0 and b != 0:
            total_loss = total_loss / 100
            print("[%d][loss:%5.2f][pp:%5.2f]" %
                  (b, total_loss, math.exp(total_loss)))
            total_loss = 0


def main():
    args = parse_arguments()
    hidden_size = 256
    embed_size = 256
    assert torch.cuda.is_available()

    vocab = WordVocab.load_vocab(args.vocab)
    print("[!] Instantiating models...")
    encoder = Encoder(len(vocab), embed_size, hidden_size, n_layers=3, dropout=0.5)
    decoder = Decoder(embed_size, hidden_size, len(vocab), n_layers=3, dropout=0.5)
    model = Seq2Seq(encoder, decoder).cuda()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    train_dataset = Seq2seqDataset(args.train_data, vocab)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.n_worker)
    val_dataset = Seq2seqDataset(args.test_data, vocab, is_train=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.n_worker)
    print(model)

    best_val_loss = None
    for e in range(1, args.n_epoch):
        train(model, optimizer, train_loader, vocab, args.grad_clip)
        val_loss = evaluate(model, val_loader, vocab)
        print("[Epoch:%d] val_loss:%5.3f | val_pp:%5.2fS"
              % (e, val_loss, math.exp(val_loss)))

        # Save the model if the validation loss is the best we've seen so far.
        if not best_val_loss or val_loss < best_val_loss:
            print("[!] saving model...")
            if not os.path.isdir(".save"):
                os.makedirs(".save")
            torch.save(model.state_dict(), './.save/seq2seq_%d.pkl' % (e))
            best_val_loss = val_loss


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt as e:
        print("[STOP]", e)
