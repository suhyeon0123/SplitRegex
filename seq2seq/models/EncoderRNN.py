import torch.nn as nn
import torch.nn.functional as F
import torch
import seq2seq
from seq2seq.util.string_preprocess import preprocessing, get_set_lengths, get_mask, get_mask2

from .baseRNN import BaseRNN

class EncoderRNN(BaseRNN):
    r"""
    Applies a multi-layer RNN to an input sequence.

    Args:
        vocab_size (int): size of the vocabulary
        max_len (int): a maximum allowed length for the sequence to be processed
        hidden_size (int): the number of features in the hidden state `h`
        input_dropout_p (float, optional): dropout probability for the input sequence (default: 0)
        dropout_p (float, optional): dropout probability for the output sequence (default: 0)
        n_layers (int, optional): number of recurrent layers (default: 1)
        bidirectional (bool, optional): if True, becomes a bidirectional encodr (defulat False)
        rnn_cell (str, optional): type of RNN cell (default: gru)
        variable_lengths (bool, optional): if use variable length RNN (default: False)
        embedding (torch.Tensor, optional): Pre-trained embedding.  The size of the tensor has to match
            the size of the embedding parameter: (vocab_size, hidden_size).  The embedding layer would be initialized
            with the tensor if provided (default: None).
        update_embedding (bool, optional): If the embedding should be updated during training (default: False).

    Inputs: inputs, input_lengths
        - **inputs**: list of sequences, whose length is the batch size and within which each sequence is a list of token IDs.
        - **input_lengths** (list of int, optional): list that contains the lengths of sequences
            in the mini-batch, it must be provided when using variable length RNN (default: `None`)

    Outputs: output, hidden
        - **output** (batch, seq_len, hidden_size): tensor containing the encoded features of the input sequence
        - **hidden** (num_layers * num_directions, batch, hidden_size): tensor containing the features in the hidden state `h`

    """

    def __init__(self, vocab_size, max_len, hidden_size,
                 input_dropout_p=0, dropout_p=0,
                 n_layers=1, bidirectional=False, rnn_cell='LSTM', variable_lengths=False,
                 embedding=None, update_embedding=True, vocab = None):
        super(EncoderRNN, self).__init__(vocab_size, max_len, hidden_size,
                input_dropout_p, dropout_p, n_layers, rnn_cell)

        self.variable_lengths = variable_lengths
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        if embedding is not None:
            self.embedding.weight = nn.Parameter(embedding)
        self.embedding.weight.requires_grad = update_embedding
        self.vocab = vocab
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size
        self.input_dropout_p = input_dropout_p
        self.n_layers= n_layers
        self.rnn1 = self.rnn_cell(vocab_size, hidden_size, n_layers,
                                 batch_first=True, bidirectional=bidirectional, dropout=dropout_p)

        self.rnn2 = self.rnn_cell(hidden_size*2 if self.bidirectional else hidden_size, hidden_size, n_layers,
                                 batch_first=True, bidirectional=bidirectional, dropout=dropout_p)

    def forward(self, input_var, input_lengths=None):

        '''pos_input  = input_var[0] # batch, set_size, seq_len
        neg_input = input_var[1] # batch, set_size, seq_len
        pos_lengths = input_lengths[0] # batch, set_size
        neg_lengths = input_lengths[1] # batch, set_size'''

        batch_size = input_var.size(0)  #64
        set_size = input_var.size(1)    #10
        seq_len = input_var.size(2)     #10


        one_hot = F.one_hot(input_var)
        src_embedded = one_hot.view(batch_size*set_size,seq_len, -1).float()
        #src_embedded = self.embedding(input_var) # batch, set_size, seq_len, embedding_dim
        #src_embedded = src_embedded.view(batch_size*set_size,seq_len, -1) # batch x set_size, seq_len, embedding_dim
        masking = get_mask(input_var)  # batch, set_size, seq_len
        input_lengths = input_lengths.reshape(-1)  # batch x set_size


        #variable_lengths is True
        if self.variable_lengths:
            src_embedded = nn.utils.rnn.pack_padded_sequence(src_embedded, input_lengths.cpu(), batch_first=True, enforce_sorted=False)
            #neg_embedded = nn.utils.rnn.pack_padded_sequence(neg_embedded, neg_lengths.cpu(), batch_first=True, enforce_sorted=False)

        src_output, src_hidden = self.rnn1(src_embedded) # (batch x set_size, seq_len, hidden), # (num_layer x num_dir, batch*set_size, hidden)
        rnn1_hidden = src_hidden # (num_layer x num_dir, batch*set_size, hidden)


        if self.variable_lengths:
            src_output, _ = nn.utils.rnn.pad_packed_sequence(src_output, batch_first=True)
            #neg_output, _ = nn.utils.rnn.pad_packed_sequence(neg_output, batch_first=True)

        src_output = src_output.view(batch_size, set_size, src_output.size(1), -1) # batch, set_size, seq_len, hidden)
        #neg_output = neg_output.view(batch_size, set_size, neg_output.size(1), -1) # batch, set_size, neg_seq_len, hidden)
        set_embedded = src_hidden[0].view(self.n_layers, -1, batch_size*set_size, self.hidden_size) # num_layer(2), num_direction, batch x set_size, hidden
        #neg_set_embedded = neg_hidden[0].view(2, -1, batch_size*set_size, self.hidden_size) # num_layer(2), num_direction, batch x set_size, hidden
        # use hidden state of final_layer
        set_embedded = set_embedded[-1, :,:,:] # num_direction, batch x set_size, hidden
        #neg_set_embedded = neg_set_embedded[-1, :,:,:] # num_direction, batch x set_size, hidden

        if self.bidirectional:
            set_embedded = torch.cat((set_embedded[0], set_embedded[1]), dim=-1) # batch x set_size, num_direction x hidden
            #neg_set_embedded = torch.cat((neg_set_embedded[0], neg_set_embedded[1]), dim=-1) # batch x set_size, num_direction x hidden
        else:
            set_embedded = set_embedded.squeeze(0) # batch x set_size, hidden
            #neg_set_embedded = neg_set_embedded.squeeze(0) # batch x set_size, hidden

        set_embedded = set_embedded.view(batch_size, set_size, -1) # batch, set_size, hidden
        #neg_set_embedded = neg_set_embedded.view(batch_size, set_size, -1) # batch, set_size, hidden
        set_output, set_hidden = self.rnn2(set_embedded) # (batch, set_size, hidden), # (num_layer*num_dir, batch, hidden) 2개 tuple 구성
        #neg_set_output, neg_set_hidden = self.rnn2(neg_set_embedded) # (batch, set_size, hidden), # (num_later*num_dir, batch, hidden) 2개 tuple


        last_hidden = set_hidden[0] # num_layer x num_dir, batch, hidden
        #neg_set_last_hidden = neg_set_hidden[0] # num_later x num_dir, batch, hidden
        last_cell = set_hidden[1] # num_layer x num_dir, batch, hidden
        #neg_set_last_cell = neg_set_hidden[1] # num_layer x num_dir, batch, hidden
        #last_hidden = torch.cat((pos_set_last_hidden, neg_set_last_hidden), dim=-1) # num_layer x num_dir, batch, 2 x hidden
        #last_cell = torch.cat((pos_set_last_cell, neg_set_last_cell), dim=-1) # num_layer x num_dir, batch, 2 x hidden

        # hidden_size *2 -> hidden_size

        #last_hidden = self.linear(last_hidden.view(-1, self.hidden_size))
        #last_hidden = last_hidden.view(-1, batch_size, self.hidden_size) # 2, 64, 128
        #last_cell = self.linear(last_cell.view(-1, self.hidden_size))
        #last_cell = last_cell.view(-1, batch_size, self.hidden_size)

        #print(last_hidden.shape, last_cell.shape)


        hiddens = (last_hidden, last_cell)
        outputs = (src_output, set_output) #revised
        return outputs, hiddens, masking, rnn1_hidden
