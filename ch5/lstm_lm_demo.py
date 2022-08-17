import os, sys
dirpath = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(dirpath)

import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from tqdm.auto import tqdm

from helper.utils import *
from helper.vocab import *

class RnnlmDataset(Dataset):

    def __init__(self, corpus, vocab):
        self.data = []
        self.bos = vocab[BOS_TOKEN]
        self.eos = vocab[EOS_TOKEN]
        self.pad = vocab[PAD_TOKEN]
        for sentence in tqdm(corpus, desc='Dataset Construction'):
            inpt = [self.bos] + sentence
            target = sentence + [self.eos]
            self.data.append((inpt, target))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]

    def collate_fn(self, examples):
        inputs = [torch.tensor(ex[0]) for ex in examples]
        targets = [torch.tensor(ex[1]) for ex in examples]
        inputs = pad_sequence(inputs, batch_first=True, padding_value=self.pad)
        targets = pad_sequence(targets, batch_first=True, padding_value=self.pad)

        return (inputs, targets)

    
class RNNLM(nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(RNNLM, self).__init__()

        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.output = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, inputs):
        embeds = self.embeddings(inputs)
        hidden, _ = self.rnn(embeds)
        output = self.output(hidden)
        log_probs = F.log_softmax(output, dim=2)
        return log_probs


if __name__ == '__main__':
    embedding_dim = 64
    context_size = 2
    hidden_dim = 128
    batch_size = 32
    num_epoch = 5

    corpus, vocab = load_reuters()
    dataset = RnnlmDataset(corpus, vocab)
    data_loader = get_loader(dataset, batch_size)

    nll_loss = nn.NLLLoss(ignore_index=dataset.pad)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RNNLM(len(vocab), embedding_dim, hidden_dim).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model.train()
    for epoch in range(num_epoch):
        total_loss = 0

        for batch in tqdm(data_loader, desc=f'Training Epoch {epoch}'):
            inputs, targets = [x.to(device) for x in batch]
            optimizer.zero_grad()
            log_probs = model(inputs)

            loss = nll_loss(log_probs.view(-1, log_probs.shape[-1]), targets.view(-1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        print(f'Loss: {total_loss:.3f}')
    
    save_pretrained(vocab, model.embedding.weight.data, './output/rnnlm.vec')


