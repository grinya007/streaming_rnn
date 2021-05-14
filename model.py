import torch
from math import exp
from time import time

class RNN(torch.nn.Module):
    def __init__(
            self,
            num_embeddings: int,
            embedding_dim: int,
            hidden_dim: int,
            n_layers: int,
            dropout: float
        ):
        super(RNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.embed = torch.nn.Embedding(
            num_embeddings,
            embedding_dim
        )
        self.lstm = torch.nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            dropout=dropout,
            batch_first=True
        )
        self.fc = torch.nn.Linear(
            hidden_dim,
            num_embeddings
        )
        self.softmax = torch.nn.LogSoftmax(dim=1)

    def forward(self, x, h):
        embeds = self.embed(x)
        out, nh = self.lstm(embeds, h)
        logits = self.softmax(self.fc(out[:, -1]))
        return logits, nh

    def init_hidden(self, batch_size):
        hidden = (
            torch.zeros(self.n_layers, batch_size, self.hidden_dim),
            torch.zeros(self.n_layers, batch_size, self.hidden_dim)
        )
        return hidden


def make_rnn(vocab_size, embedding_dim, hidden_dim, num_layers, dropout):
    rnn = RNN(vocab_size, embedding_dim, hidden_dim, num_layers, dropout)
    criterion = torch.nn.NLLLoss()
    optimizer = torch.optim.Adam(rnn.parameters(), lr=0.001)

    return rnn, criterion, optimizer

def train_iteration(loader, rnn, criterion, optimizer, state, device):
    state_h, state_c = rnn.init_hidden(state.batch_size)
    state_h = state_h.to(device)
    state_c = state_c.to(device)
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        out, (state_h, state_c) = rnn(x, (state_h, state_c))
        loss = criterion(out, y.T)
        unknowns = (out.topk(1)[1] == 0).nonzero().shape[0]

        state_h = state_h.detach()
        state_c = state_c.detach()

        loss.backward()
        optimizer.step()

        state.next_batch(unknowns, loss.item())

    return state


class TrainState:
    def __init__(self, batch_size, record_every, print_log=False):
        self.timing = time()
        self.batch_counter = 0
        self.avg_loss = 0.
        self.unknowns = 0
        self.perplexity = []
        self.unknowns_ratio = []

        self.batch_size = batch_size
        self.record_every = record_every
        self.print_log = print_log

    def next_batch(self, unknowns, loss):
        self.batch_counter += 1
        self.unknowns += unknowns
        self.avg_loss += loss
        if self.batch_counter % self.record_every == 0:
            timing = time() - self.timing
            self.timing = time()

            unknowns_ratio = 100*self.unknowns/(self.record_every*self.batch_size)
            self.unknowns_ratio.append(unknowns_ratio)
            self.unknowns = 0

            avg_loss = self.avg_loss/self.record_every
            perplexity = exp(avg_loss)
            self.perplexity.append(perplexity)
            self.avg_loss = 0.

            if self.print_log:
                print((
                    "batch: {} (last {} batches in {:.2f} s), "
                    "unknown word prediction ratio: {:.4f}, "
                    "avg loss: {:.4f}, ppl: {:.4f}").format(
                    self.batch_counter, self.record_every,
                    timing, unknowns_ratio, avg_loss, perplexity
                ), flush=True)


class DataSet(torch.utils.data.IterableDataset):
    def __init__(self, word_index_generator, lookback):
        super(DataSet).__init__()
        self.word_index_generator = word_index_generator
        self.lookback = lookback

    def __iter__(self):
        X = []
        for idx in self.word_index_generator:
            if len(X) <= self.lookback:
                X.append(idx)
            else:
                yield torch.tensor(X[:-1]), torch.tensor(X[-1])
                X.pop(0)
                X.append(idx)


def make_data_loader(word_index_generator, batch_size, lookback):
    ds = DataSet(word_index_generator, lookback)
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, drop_last=True)
    return loader


def torch_device(device_name):
    return torch.device(device_name)

