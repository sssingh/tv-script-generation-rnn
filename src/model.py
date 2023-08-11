import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class RNN(nn.Module):
    def __init__(
        self, vocab_size, output_size, embedding_dim, hidden_dim, n_layers, dropout=0.5
    ):
        """
        Initialize the PyTorch RNN Module
        :param vocab_size: The number of input dimensions of the neural network (the size of the vocabulary)
        :param output_size: The number of output dimensions of the neural network
        :param embedding_dim: The size of embeddings, should you choose to use them
        :param hidden_dim: The size of the hidden layer outputs
        :param dropout: dropout to add in between LSTM/GRU layers
        """
        super(RNN, self).__init__()

        # set class variables
        self.vocab_size = vocab_size
        self.output_size = output_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        # define model layers
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=embedding_dim
        )
        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(in_features=hidden_dim, out_features=output_size)

    def forward(self, nn_input, hidden):
        """
        Forward propagation of the neural network
        :param nn_input: The input to the neural network
        :param hidden: The hidden state
        :return: Two Tensors, the output of the neural network and the latest hidden state
        """

        # nn_input shape = (batch_size, sequence_length, feature_dim) --> (64, 15, 1)
        # hidden shape = (n_layers, batch_size, hidden_dim) --> (2, 64, 512)

        # x embedding o/p shape = (batch_size, embedding_dim) --> (64, 300)
        x = self.embedding(nn_input)

        # x gru o/p shape = (batch_size, sequence_length, hidden_dim) --> (64, 15, 512)
        x, hidden = self.gru(x, hidden)
        x = self.dropout(x)

        # reshape GRU output such that 2nd dim = hidden_dim so that we can feed
        # it to fc layer --> (64 * 15, 512)
        x = x.reshape(-1, self.hidden_dim)

        # output fc o/p shape = (batch_size*sequence_length, output_size) --> (64 * 15, vocab_size)
        output = self.fc(x)

        # reshape fc output to make first dim = batch_size and last dim = output_size
        batch_size = nn_input.shape[0]
        sequence_length = nn_input.shape[1]
        output = output.reshape(batch_size, sequence_length, self.output_size)

        # extract all outputs for last time-step (-1 below) for all batches
        output = output[:, -1, :]

        # return one batch of output word scores and the hidden state
        return output, hidden

    def init_hidden(self, batch_size):
        """
        Initialize the hidden state of an LSTM/GRU
        :param batch_size: The batch_size of the hidden state
        :return: hidden state of dims (n_layers, batch_size, hidden_dim)
        """

        # initialize hidden state with zeros and move it to GPU if available
        train_on_gpu = torch.cuda.is_available()
        if train_on_gpu:
            hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim).cuda()
        else:
            hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim)

        return hidden


def generate_script(
    rnn, prime_id, int_to_vocab, token_dict, pad_value, predict_len=100
):
    """
    Generate text using the trained neural network
    :param decoder: The PyTorch Module that holds the trained neural network
    :param prime_id: The word id to start the first prediction
    :param int_to_vocab: Dict of word id keys to word values
    :param token_dict: Dict of punctuation tokens keys to punctuation values
    :param pad_value: The value used to pad a sequence
    :param predict_len: The length of text to generate
    :return: The generated text
    """
    rnn.eval()
    sequence_length = 15
    train_on_gpu = torch.cuda.is_available()

    # create a sequence (batch_size=1) with the prime_id
    current_seq = np.full((1, sequence_length), pad_value)
    current_seq[-1][-1] = prime_id
    predicted = [int_to_vocab[prime_id]]

    for _ in range(predict_len):
        if train_on_gpu:
            current_seq = torch.LongTensor(current_seq).cuda()
        else:
            current_seq = torch.LongTensor(current_seq)

        # initialize the hidden state
        hidden = rnn.init_hidden(current_seq.size(0))

        # get the output of the rnn
        output, _ = rnn(current_seq, hidden)

        # get the next word probabilities
        p = F.softmax(output, dim=1).data
        if train_on_gpu:
            p = p.cpu()  # move to cpu

        # use top_k sampling to get the index of the next word
        top_k = 5
        p, top_i = p.topk(top_k)
        top_i = top_i.numpy().squeeze()

        # select the likely next word index with some element of randomness
        p = p.numpy().squeeze()
        word_i = np.random.choice(top_i, p=p / p.sum())

        # retrieve that word from the dictionary
        word = int_to_vocab[word_i]
        predicted.append(word)

        if train_on_gpu:
            current_seq = current_seq.cpu()  # move to cpu
        # the generated word becomes the next "current sequence" and the cycle can continue
        if train_on_gpu:
            current_seq = current_seq.cpu()
        current_seq = np.roll(current_seq, -1, 1)
        current_seq[-1][-1] = word_i

    gen_sentences = " ".join(predicted)

    # Replace punctuation tokens
    for key, token in token_dict.items():
        ending = " " if key in ["\n", "(", '"'] else ""
        gen_sentences = gen_sentences.replace(" " + token.lower(), key)
    gen_sentences = gen_sentences.replace("\n ", "\n")
    gen_sentences = gen_sentences.replace("( ", "(")

    # return all the sentences
    return gen_sentences
