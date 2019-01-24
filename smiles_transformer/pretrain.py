import argparse
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from bert import BERT, BERTLM
from dataset import STDataset
from build_vocab import WordVocab

class STTrainer:
    def __init__(self, bert: BERT, vocab_size: int,
                 train_dataloader: DataLoader, test_dataloader: DataLoader = None,
                 lr: float = 1e-4, betas=(0.9, 0.999), weight_decay: float = 0.01,
                 with_cuda: bool = True, log_freq: int = 10):
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
        cuda_condition = torch.cuda.is_available() and with_cuda
        self.device = torch.device("cuda:0" if cuda_condition else "cpu")

        self.bert = bert
        self.model = BERTLM(bert, vocab_size).to(self.device)

        if torch.cuda.device_count() > 1:
            print("Using %d GPUS for BERT" % torch.cuda.device_count())
            self.model = nn.DataParallel(self.model)

        self.train_data = train_dataloader
        self.test_data = test_dataloader

        self.optim = Adam(self.model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
        self.criterion = nn.NLLLoss()
        self.log_freq = log_freq
        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))
        

    def train(self, epoch):
        self.iteration(epoch, self.train_data)

    def test(self, epoch):
        self.iteration(epoch, self.test_data, train=False)

    def iteration(self, epoch, data_loader, train=True):
        """
        :param epoch: 現在のepoch
        :param data_loader: torch.utils.data.DataLoader
        :param train: trainかtestかのbool値
        """
        str_code = "train" if train else "test"
        data_iter = tqdm(enumerate(data_loader), desc="EP_%s:%d" % (str_code, epoch), total=len(data_loader), bar_format="{l_bar}{r_bar}")

        avg_loss = 0.0
        total_correct = 0
        total_element = 0

        for i, data in data_iter:
            # 0. batch_dataはGPU or CPUに載せる
            data = {key: value.to(self.device) for key, value in data.items()}
            tsm, msm = self.model.forward(data["bert_input"], data["segment_embd"])
            loss_tsm = self.criterion(tsm, data["is_same"])
            loss_msm = self.criterion(msm.transpose(1, 2), data["bert_label"])
            loss = loss_tsm + loss_msm
            if train:
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

            # TSM prediction accuracy
            correct = tsm.argmax(dim=-1).eq(data["is_same"]).sum().item()
            avg_loss += loss.item()
            total_correct += correct
            total_element += data["is_same"].nelement()

            post_fix = {
                "epoch": epoch,
                "iter": i,
                "avg_loss": avg_loss / (i + 1),
                "avg_acc": total_correct / total_element * 100,
                "loss": loss.item()
            }
            if i % self.log_freq == 0:
                data_iter.write(str(post_fix))
        return  avg_loss/len(data_iter), total_correct*100.0/total_element # Total loss and TSM accuracy

    def save(self, epoch, save_dir):
        """
        Saving the current BERT model on file_path

        :param epoch: current epoch number
        :param file_path: model output path which gonna be file_path+"ep%d" % epoch
        :return: final_output_path
        """
        output_path = save_dir + '/ep_{:02}.pkl'.format(epoch)
        torch.save(self.bert.cpu(), output_path)
        self.bert.to(self.device)
        print("EP:%d Model Saved on:" % epoch, output_path)

def main():
    parser = argparse.ArgumentParser(description='Pretrain SMILES Transformer')
    parser.add_argument('--n_epoch', '-e', type=int, default=100, help='number of epochs')
    parser.add_argument('--vocab', '-v', type=str, default='data/vocab.pkl', help='vocabulary (.pkl)')
    parser.add_argument('--train_data', type=str, default='data/chembl24_bert_train.csv', help='train corpus (.csv)')
    parser.add_argument('--test_data', type=str, default='data/chembl24_bert_test.csv', help='test corpus (.csv)')
    parser.add_argument('--out-dir', '-o', type=str, default='../result', help='output directory')
    parser.add_argument('--name', '-n', type=str, default='ST', help='model name')
    parser.add_argument('--seq_len', type=int, default=203, help='maximum length of the paired seqence')
    parser.add_argument('--batch_size', '-b', type=int, default=1024, help='batch size')
    parser.add_argument('--n_worker', '-w', type=int, default=8, help='number of workers')
    parser.add_argument('--hidden', type=int, default=256, help='length of hidden vector')
    parser.add_argument('--n_layer', '-l', type=int, default=4, help='number of layers')
    parser.add_argument('--n_head', type=int, default=4, help='number of attention heads')
    parser.add_argument('--dropout', '-d', type=float, default=0.1, help='dropout rate')
    parser.add_argument('--lr', type=float, default=1e-3, help='Adam learning rate')
    parser.add_argument('--beta1', type=float, default=0.9, help='Adam beta1')
    parser.add_argument('--beta2', type=float, default=0.999, help='Adam beta2')
    parser.add_argument('--weight-decay', type=float, default=0.01, help='dropout rate')
    parser.add_argument('--with-cuda', action='store_true', help='use CUDA')
    parser.add_argument('--log-freq', type=int, default=100, help='log frequency')
    args = parser.parse_args()

    print("Loading Vocab", args.vocab)
    vocab = WordVocab.load_vocab(args.vocab)
    print("Loading Train Dataset", args.train_data)
    train_dataset = STDataset(args.train_data, vocab, seq_len=args.seq_len)
    print("Loading Test Dataset", args.test_data)
    test_dataset = STDataset(args.test_data, vocab, seq_len=args.seq_len)
    print("Creating Dataloader")
    train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.n_worker)
    test_data_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.n_worker) 
    print("Building BERT model")
    bert = BERT(len(vocab), hidden=args.hidden, n_layers=args.n_layer, attn_heads=args.n_head, dropout=args.dropout)
    print("Creating BERT Trainer")
    trainer = STTrainer(bert, len(vocab), train_dataloader=train_data_loader, test_dataloader=test_data_loader,
                        lr=args.lr, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay,
                        with_cuda=args.with_cuda, log_freq=args.log_freq)

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    save_dir = os.path.join(args.out_dir, args.name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    log_dir = os.path.join(args.out_dir, 'log')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    with open(log_dir + '/' + args.name + '.csv') as f:
        f.write('epoch,train_loss,train_acc,test_loss,test_acc\n')

    print("Training Start")
    for epoch in tqdm(range(args.n_epoch)):
        loss, acc = trainer.train(epoch)
        print("EP%d Train, loss=" % (epoch), loss, "accuracy=", acc)
        with open(log_dir + '/' + args.name + '.csv') as f:
            f.write('%d,%f,%f' %(epoch, loss, acc))
        
        trainer.save(epoch, save_dir) # Save model
        
        loss, acc = trainer.test(epoch)
        print("EP%d Test, loss=" % (epoch), loss, "accuracy=", acc)
        with open(log_dir + '/' + args.name + '.csv') as f:
            f.write('%f,%f\n' %(loss, acc))

if __name__=='__main__':
    main()