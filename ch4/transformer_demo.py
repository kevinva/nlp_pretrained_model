import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

import math
from tqdm.auto import tqdm

import os, sys
dirpath = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(dirpath)

from helper.vocab import *
from helper.utils import load_sentence_polarity, length_to_mask

class TransformerDataset(Dataset):

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


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=512):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x

class Transformer(nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_class, dim_feedforward=512, \
                 num_head=2, num_layers=2, dropout=0.1, max_len=128, activation: str='relu'):
        super(Transformer, self).__init__()

        self.embedding_dim = embedding_dim
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.positional_encoding = PositionalEncoding(embedding_dim, dropout, max_len)

        encoder_layer = nn.TransformerEncoderLayer(hidden_dim, num_head, dim_feedforward, dropout, activation)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.output = nn.Linear(embedding_dim, num_class)

    def forward(self, inputs, lengths):
        inputs = torch.transpose(inputs, 0, 1)
        print(f'transpose inputs: {inputs}')
        hidden_states = self.embeddings(inputs)
        print(f'embeddings: {hidden_states}')
        hidden_states = self.positional_encoding(hidden_states)
        print(f'positional_encoding: {hidden_states}')
        attention_mask = length_to_mask(lengths) == False
        print(f'attention_mask: {attention_mask}')
        hidden_states = self.transformer(hidden_states, src_key_padding_mask=attention_mask)  # encoder这个mask可以将padding的部分忽略掉，让attention的注意力机制不再参与这一部分的运算
        print(f'transformer: {hidden_states}')
        hidden_states = hidden_states[0, :, :]  # hoho_todo: 为啥只要第一个维度？Bert?
        print(f'hidden_states: {hidden_states}')
        output = self.output(hidden_states)
        log_probs = F.log_softmax(output, dim=1)
        return log_probs


if __name__ == '__main__':
    # print(torch.arange(10).expand(3, 10))
    # print(length_to_mask(torch.tensor([3, 4, 2, 8])) == False)

    # lengths = torch.tensor([6, 4, 5])
    # inpts = [torch.tensor([1, 23, 4, 5, 63, 2]), torch.tensor([1, 3, 45, 4]), torch.tensor([9, 3, 4, 2, 6])]
    # inpts = pad_sequence(inpts, batch_first=True)
    # transformer = Transformer(1000, 4, 4, 2)
    # transformer(inpts, lengths)

    embedding_dim = 128
    hidden_dim = 128
    num_class = 2
    batch_size = 32
    num_epoch = 5

    # 加载数据
    train_data, test_data, vocab = load_sentence_polarity()
    train_dataset = TransformerDataset(train_data)
    test_dataset = TransformerDataset(test_data)
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
    test_data_loader = DataLoader(test_dataset, batch_size=1, collate_fn=collate_fn, shuffle=False)

    # 加载模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Transformer(len(vocab), embedding_dim, hidden_dim, num_class)
    model.to(device) # 将模型加载到GPU中（如果已经正确安装）

    # 训练过程
    nll_loss = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001) # 使用Adam优化器

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
        print(f"Loss: {total_loss:.3f}")

    # 测试过程
    acc = 0
    for batch in tqdm(test_data_loader, desc=f"Testing"):
        inputs, lengths, targets = [x.to(device) for x in batch]
        with torch.no_grad():
            output = model(inputs, lengths)
            acc += (output.argmax(dim=1) == targets).sum().item()

    # 输出在测试集上的准确率
    print(f"Acc: {acc / len(test_data_loader):.3f}")