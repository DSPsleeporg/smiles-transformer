import argparse
import random
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
import os
import math
import argparse
import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm_
from torch.nn import functional as F
from build_vocab import WordVocab
from dataset import Seq2seqDataset

PAD = 0
UNK = 1
EOS = 2
SOS = 3
MASK = 4

class Encoder(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size,
                 n_layers=1, dropout=0.5):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.embed = nn.Embedding(input_size, embed_size)
        self.gru = nn.GRU(embed_size, hidden_size, n_layers,
                          dropout=dropout, bidirectional=True)

    def forward(self, src, hidden=None):
        # src: (T,B)
        embedded = self.embed(src)# (T,B,H)
        outputs, hidden = self.gru(embedded, hidden) # (T,B,2H), (2L,B,H) 
        # sum bidirectional outputs
        outputs = (outputs[:, :, :self.hidden_size] +
                   outputs[:, :, self.hidden_size:])
        return outputs, hidden # (T,B,H), (2L,B,H)


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))
        stdv = 1. / math.sqrt(self.v.size(0))
        self.v.data.uniform_(-stdv, stdv)

    def forward(self, hidden, encoder_outputs):
        timestep = encoder_outputs.size(0)
        h = hidden.repeat(timestep, 1, 1).transpose(0, 1)
        encoder_outputs = encoder_outputs.transpose(0, 1)  # [B*T*H]
        attn_energies = self.score(h, encoder_outputs)
        return F.relu(attn_energies).unsqueeze(1)

    def score(self, hidden, encoder_outputs):
        # [B*T*2H]->[B*T*H]
        energy = F.softmax(self.attn(torch.cat([hidden, encoder_outputs], 2)))
        energy = energy.transpose(1, 2)  # [B*H*T]
        v = self.v.repeat(encoder_outputs.size(0), 1).unsqueeze(1)  # [B*1*H]
        energy = torch.bmm(v, energy)  # [B*1*T]
        return energy.squeeze(1)  # [B*T]


class Decoder(nn.Module):
    def __init__(self, embed_size, hidden_size, output_size,
                 n_layers=1, dropout=0.2):
        super(Decoder, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers

        self.embed = nn.Embedding(output_size, embed_size)
        self.dropout = nn.Dropout(dropout, inplace=True)
        self.attention = Attention(hidden_size)
        self.gru = nn.GRU(hidden_size + embed_size, hidden_size,
                          n_layers, dropout=dropout)
        self.out = nn.Linear(hidden_size * 2, output_size)

    def forward(self, input, last_hidden, encoder_outputs):
        # Get the embedding of the current input word (last output word)
        embedded = self.embed(input).unsqueeze(0)  # (1,B,N)
        embedded = self.dropout(embedded)
        # Calculate attention weights and apply to encoder outputs
        attn_weights = self.attention(last_hidden[-1], encoder_outputs)
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))  # (B,1,N)
        context = context.transpose(0, 1)  # (1,B,N)
        # Combine embedded input word and attended context, run through RNN
        rnn_input = torch.cat([embedded, context], 2)
        output, hidden = self.gru(rnn_input, last_hidden)
        output = output.squeeze(0)  # (1,B,N) -> (B,N)
        context = context.squeeze(0)
        output = self.out(torch.cat([output, context], 1))
        output = F.log_softmax(output, dim=1) # log???
        return output, hidden, attn_weights


class RNNSeq2Seq(nn.Module):
    def __init__(self, in_size, hidden_size, out_size, n_layers):
        super(RNNSeq2Seq, self).__init__()
        self.encoder = Encoder(in_size, hidden_size, hidden_size, n_layers)
        self.decoder = Decoder(hidden_size, hidden_size, out_size, n_layers)

    def forward(self, src, trg, teacher_forcing_ratio=0.5): # (T,B)
        batch_size = src.size(1)
        max_len = trg.size(0)
        vocab_size = self.decoder.output_size
        outputs = Variable(torch.zeros(max_len, batch_size, vocab_size)).cuda() # (T,B,V)
        encoder_output, hidden = self.encoder(src) # (T,B,H), (2L,B,H)
        hidden = hidden[:self.decoder.n_layers] # (L,B,H)
        output = Variable(trg.data[0, :])  # sos
        for t in range(1, max_len):
            output, hidden, attn_weights = self.decoder(output, hidden, encoder_output) # (B,V), (L,B,H)
            outputs[t] = output
            is_teacher = random.random() < teacher_forcing_ratio
            top1 = output.data.max(dim=1)[1] # (B)
            output = Variable(trg.data[t] if is_teacher else top1).cuda()
        return outputs # (T,B,V)

    def _encode(self, src):
        # src: (T,B)
        embedded = self.encoder.embed(src)# (T,B,H)
        _, hidden = self.encoder.gru(embedded, None) # (T,B,2H), (2L,B,H)
        hidden = hidden.detach().numpy() 
        return np.hstack(hidden[2:]) #(B,4H)
        
    def encode(self, src):
        # src: (T,B)
        batch_size = src.shape[1]
        if batch_size<=100:
            return self._encode(src)
        else: # Batch is too large to load
            print('There are {:d} molecules. It will take a little time.'.format(batch_size))
            st,ed = 0,100
            out = self._encode(src[:,st:ed]) # (B,4H)
            while ed<batch_size:
                st += 100
                ed += 100
                out = np.concatenate([out, self._encode(src[:,st:ed])], axis=0)
            return out
        
def parse_arguments():
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--n_epoch', '-e', type=int, default=20, help='number of epochs')
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
    for b, data in enumerate(val_loader):
        sm1, sm2 = torch.t(data[0].cuda()), torch.t(data[1].cuda()) # (T,B)
        with torch.no_grad():
            output = model(sm1, sm2, teacher_forcing_ratio=0.0) # (T,B,V)
        loss = F.nll_loss(output[1:].view(-1, len(vocab)),
                               sm2[1:].contiguous().view(-1),
                               ignore_index=PAD)
        total_loss += loss.item()
    return total_loss / len(val_loader)

def main():
    args = parse_arguments()
    hidden_size = 256
    embed_size = 256
    assert torch.cuda.is_available()

    vocab = WordVocab.load_vocab(args.vocab)
    print("[!] Instantiating models...")
    encoder = Encoder(len(vocab), embed_size, hidden_size, n_layers=3, dropout=0.5)
    decoder = Decoder(embed_size, hidden_size, len(vocab), n_layers=3, dropout=0.5)
    model = RNNSeq2Seq(encoder, decoder).cuda()
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
            model.train()
            sm1, sm2 = torch.t(data[0].cuda()), torch.t(data[1].cuda()) # (T,B)
            optimizer.zero_grad()
            output = model(sm1, sm2, teacher_forcing_ratio=1.0) # (T,B,V)
            loss = F.nll_loss(output[1:].view(-1, len(vocab)),
                    sm2[1:].contiguous().view(-1), ignore_index=PAD)
            loss.backward()
            clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            if b%100==0:
                print('Train {:3d}: iter {:5d} | loss {:.3f} | ppl {:.3f}'.format(e, b, loss.item(), math.exp(loss.item())))
            if b%1000==0:
                loss = evaluate(model, val_loader, vocab)
                print('Val {:3d}: iter {:5d} | loss {:.3f} | ppl {:.3f}'.format(e, b, loss, math.exp(loss)))
                # Save the model if the validation loss is the best we've seen so far.
                
                print("[!] saving model...")
                if not os.path.isdir(".save"):
                    os.makedirs(".save")
                torch.save(model.state_dict(), './.save/rnnenum_%d_%d.pkl' % (e,b))
                best_loss = loss

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt as e:
        print("[STOP]", e)
