import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence


import os, sys
dirpath = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(dirpath)

from helper.vocab import *
from helper.utils import load_sentence_polarity

from tqdm.auto import tqdm

class LstmDataset(Dataset):

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]


def collate_fn(examples):
    lengths = torch.tensor([len(ex[0]) for ex in examples])
    inputs = [torch.tensor(ex[0]) for ex in examples]
    targets = torch.tensor([ex[1] for ex in examples], dtype=torch.long)
    inputs = pad_sequence(inputs, batch_first=True)
    return inputs, lengths, targets


class LSTM(nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_class):
        super(LSTM, self).__init__()

        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.output = nn.Linear(hidden_dim, num_class)

    def forward(self, inputs, lengths):
        embeddings = self.embeddings(inputs)
        # print(f'embeddings={embeddings}')
        x_pack = pack_padded_sequence(embeddings, lengths, batch_first=True, enforce_sorted=False)
        # print(f'x_pack={x_pack}')
        hidden, (hn, cn) = self.lstm(x_pack)
        # print(f'hidden={hidden}, hn={hn.size()}, cn={cn.size()}')
        outputs = self.output(hn[-1])
        log_probs = F.log_softmax(outputs, dim=-1)
        return log_probs


if __name__ == '__main__':
    # lengths = [6, 4, 5]
    # inpts = [torch.tensor([1, 23, 4, 5, 63, 2]), torch.tensor([1, 3, 45, 4]), torch.tensor([9, 3, 4, 2, 6])]
    # inpts = pad_sequence(inpts, batch_first=True)
    # print(inpts)

    # lstm = LSTM(1000, 4, 10, 2)
    # lstm(inpts, lengths)

    embedding_dim = 128
    hidden_dim = 256
    num_class = 2
    batch_size = 32
    num_epoch = 5

    #加载数据
    train_data, test_data, vocab = load_sentence_polarity()
    train_dataset = LstmDataset(train_data)
    test_dataset = LstmDataset(test_data)
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
    test_data_loader = DataLoader(test_dataset, batch_size=1, collate_fn=collate_fn, shuffle=False)

    #加载模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LSTM(len(vocab), embedding_dim, hidden_dim, num_class)
    model.to(device) #将模型加载到GPU中（如果已经正确安装）

    #训练过程
    nll_loss = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001) #使用Adam优化器

    model.train()
    for epoch in range(num_epoch):
        total_loss = 0
        for batch in tqdm(train_data_loader, desc=f"Training Epoch {epoch}"):
            inputs, lengths, targets = [x.to(device) for x in batch]
            log_probs = model(inputs, lengths)
            loss = nll_loss(log_probs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Loss: {total_loss:.2f}")

    #测试过程
    acc = 0
    for batch in tqdm(test_data_loader, desc=f"Testing"):
        inputs, lengths, targets = [x.to(device) for x in batch]
        with torch.no_grad():
            output = model(inputs, lengths)
            acc += (output.argmax(dim=1) == targets).sum().item()

    #输出在测试集上的准确率
    print(f"Acc: {acc / len(test_data_loader):.2f}")