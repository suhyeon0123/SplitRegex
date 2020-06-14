import torch.nn as nn
import torch

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
        
        self.rnn1 = self.rnn_cell(hidden_size, hidden_size, n_layers,
                                 batch_first=True, bidirectional=bidirectional, dropout=dropout_p)
        
        self.rnn2 = self.rnn_cell(n_layers * hidden_size*2 if self.bidirectional else n_layers * hidden_size, hidden_size,
                                 batch_first=True, bidirectional=True, dropout=dropout_p)
        
        
    def forward(self, input_var, input_lengths=None):
        """
        Applies a multi-layer RNN to an input sequence.

        Args:
            input_var (batch, seq_len): tensor containing the features of the input sequence.
            input_lengths (list of int, optional): A list that contains the lengths of sequences
              in the mini-batch

        Returns: output, hidden
            - output = (output1, output2)
            - **output1** (batch, set_size, seq_len, hidden_size): 
            - **output2** (batch, set_size, hidden_size*bidirectional): 
            - **hidden2** (num_layers * num_directions, batch, hidden_size): Tuple(final hidden state, final cell state)
        """

        batch_size = input_var.size(0)
        set_size = input_var.size(1)
        seq_len = input_var.size(2)
        
        embedded = self.embedding(input_var)
        embedded = embedded.view(batch_size*set_size, seq_len, -1)
        embedded = self.input_dropout(embedded)

        if self.variable_lengths:
            embedded = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths.view(-1).cpu(), batch_first=True, enforce_sorted=False)
            
        output1, hidden1 = self.rnn1(embedded)
        if self.variable_lengths:
            output1, _ = nn.utils.rnn.pad_packed_sequence(output1, batch_first=True)
        
        output1 = output1.view(batch_size, set_size, seq_len, -1)
        h_n1 = hidden1[0]
        c_n1 = hidden1[1]
                
        # bidirectional concat
        h_n1 = self._cat_directions(h_n1).squeeze(0)
        c_n1 = self._cat_directions(c_n1).squeeze(0)

        # layer concat 
        h_n1 = torch.cat([h_i for h_i in h_n1], dim = -1)
        c_n1 = torch.cat([c_i for c_i in c_n1], dim = -1)
        
        h_n1 = h_n1.view(batch_size, set_size, -1)
        c_n1 = c_n1.view(batch_size, set_size, -1)
        
        # feed final hidden state of set string into the new LSTM
        output2, hidden2 = self.rnn2(h_n1)
        output = (output1, output2)
        return output, hidden2
    
    
    def _cat_directions(self, h):
        """ If the encoder is bidirectional, do the following transformation.
            (#directions * #layers, #batch, hidden_size) -> (#layers, #batch, #directions * hidden_size)
        """
        if self.bidirectional:
            h = torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)
        return h
