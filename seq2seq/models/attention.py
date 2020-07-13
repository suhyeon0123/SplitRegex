import torch
import torch.nn as nn
import torch.nn.functional as F
from seq2seq.util.string_preprocess import pad_attention


class Attention(nn.Module):
    r"""
    Applies an attention mechanism on the output features from the decoder.

    .. math::
            \begin{array}{ll}
            x = context*output \\
            attn = exp(x_i) / sum_j exp(x_j) \\
            output = \tanh(w * (attn * context) + b * output)
            \end{array}

    Args:
        dim(int): The number of expected features in the output

    Inputs: output, context
        - **output** (batch, output_len, dimensions): tensor containing the output features from the decoder.
        - **context** (batch, input_len, dimensions): tensor containing features of the encoded input sequence.

    Outputs: output, attn
        - **output** (batch, output_len, dimensions): tensor containing the attended output features from the decoder.
        - **attn** (batch, output_len, input_len): tensor containing attention weights.

    Attributes:
        linear_out (torch.nn.Linear): applies a linear transformation to the incoming data: :math:`y = Ax + b`.
        mask (torch.Tensor, optional): applies a :math:`-inf` to the indices specified in the `Tensor`.

    Examples::

         >>> attention = seq2seq.models.Attention(256)
         >>> context = Variable(torch.randn(5, 3, 256))
         >>> output = Variable(torch.randn(5, 5, 256))
         >>> output, attn = attention(output, context)

    """
    def __init__(self, dim):
        super(Attention, self).__init__()
        self.linear_out = nn.Linear(dim*2, dim)
        self.mask = None
        self.mask1 = None
        self.mask2 = None
        
        
    def set_mask(self, mask):
        """
        Sets indices to be masked

        Args:
            mask (torch.Tensor): tensor containing indices to be masked
        """
        self.mask = mask
        self.mask1 = mask[0]
        self.mask2 = mask[1]


    def forward(self, dec_hidden, context):
        ''' 
        context -> context is encoder outputs that is composed of two LSTM encoder outputs with tuple.
        context[0] -> (#batch, #set_num, #enc_len, #hidden)
        context[1] -> (#batch, #set_num, #hidden)
        dec_hidden -> decoder hidden state (#batch, #dec_len, #hidden)
        '''

        batch_size = context[0].size(0) 
        set_size = context[0].size(1)
        enc_len = context[0].size(2)
        dec_len = dec_hidden.size(1)
        hidden_size = dec_hidden.size(-1)
        attn_set = []
        
        outputs1 = context[0]
        outputs2 = context[1]
        
        outputs1 = outputs1.view(outputs1.size(0), -1, outputs1.size(-1))
        attn = torch.bmm(dec_hidden, outputs1.transpose(1,2))
        attn = attn.view(batch_size, dec_len, set_size, enc_len)
        if self.mask1 is not None:
            self.mask1 = self.mask1.unsqueeze(1)
            attn = attn.data.masked_fill(self.mask1, -float('inf'))
        
        attn = torch.softmax(attn, dim=-1)
        attn[attn != attn] = 0
        attn_set.append(attn)
        
        outputs1 = context[0].reshape(batch_size*set_size, enc_len, -1)
        attn = attn.transpose(1,2)
        attn = attn.reshape(batch_size*set_size, -1, enc_len)
        
        c_t = torch.bmm(attn, outputs1)
        c_t = c_t.reshape(batch_size, set_size, dec_len, -1)
        c_t = c_t.transpose(1,2)
        
        attn2 = torch.bmm(dec_hidden, outputs2.transpose(1,2))
        attn2 = pad_attention(attn2, c_t.size(2))

        if self.mask2 is not None:
            self.mask2 = self.mask2.unsqueeze(1)
            attn2 = attn2.data.masked_fill(self.mask2, -float('inf'))

        attn2 = torch.softmax(attn2, dim=-1)
        attn_set.append(attn2)
        
        attn2 = attn2.unsqueeze(-1)
        attn2 = attn2.reshape(attn2.size(0)*attn2.size(1), -1, 1)
        attn2 = attn2.transpose(1,2)

        c_t = c_t.reshape(batch_size *dec_len, set_size, -1)
        c_t = torch.bmm(attn2, c_t)
        c_t = c_t.reshape(batch_size, dec_len, 1, -1)
        c_t = c_t.squeeze(2)

        combined = torch.cat((c_t, dec_hidden), dim=2)
        output = torch.tanh(self.linear_out(combined.view(-1,2*hidden_size))).view(batch_size, -1, hidden_size)
        return output, attn_set