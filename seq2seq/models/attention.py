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
    def __init__(self, dim, attn_mode):
        super(Attention, self).__init__()
        self.mask = None
        self.mask1 = None
        self.mask2 = None
        self.attn_mode = attn_mode # True (attention both pos and neg) # False (attention only pos samples)
        if self.attn_mode:
            self.linear_out = nn.Linear(dim*3, dim)
        else:
            self.linear_out = nn.Linear(dim*2, dim)

    def set_mask(self, mask):
        """
        Sets indices to be masked

        Args:
            mask (torch.Tensor): tensor containing indices to be masked
        """
        self.mask = mask
        self.mask1 = mask[0] # masking for positive samples
        self.mask2 = mask[1] # masking for negative samples

        
    def first_attention(self, dec_hidden, encoder_hiddens, mask):
        attn_output = None
        batch_size = dec_hidden.size(0)
        set_size = encoder_hiddens.size(1)
        enc_len = encoder_hiddens.size(2)
        dec_len = dec_hidden.size(1)
        encoder_hiddens_temp = encoder_hiddens.view(batch_size, set_size*enc_len, -1)
        encoder_hiddens_temp = encoder_hiddens_temp.transpose(1,2)
        attn = torch.bmm(dec_hidden, encoder_hiddens_temp)
        attn = attn.view(batch_size, -1, set_size, enc_len)
        if mask is not None:
            mask = mask.unsqueeze(1)
            attn = attn.data.masked_fill(mask, -float('inf'))
        attn = torch.softmax(attn, dim=-1)
        attn_output = attn
        encoder_hiddens = encoder_hiddens.reshape(batch_size*set_size, enc_len, -1)
        attn = attn.transpose(1,2)
        attn = attn.reshape(batch_size*set_size, -1, enc_len)
        c_t = torch.bmm(attn, encoder_hiddens)
        c_t = c_t.reshape(batch_size, set_size, dec_len, -1)
        c_t = c_t.transpose(1,2)
        return c_t, attn_output
    
    
    def second_attention(self, dec_hidden, encoder_hiddens):
        attn = torch.bmm(dec_hidden, encoder_hiddens.transpose(1,2))
        attn = torch.softmax(attn, dim=-1)
        return attn
    
    
    def forward(self, output, context):

        batch_size = output.size(0)  #640
        hidden_size = output.size(2) #128
        input_size = context.size(1)  #10

        # (batch, out_len, dim) * (batch, in_len, dim) -> (batch, out_len, in_len)
        attn = torch.bmm(output, context.transpose(1, 2))
        if self.mask is not None:
            attn.data.masked_fill_(self.mask, -float('inf'))
        attn = F.softmax(attn.view(-1, input_size), dim=1).view(batch_size, -1, input_size)

        # (batch, out_len, in_len) * (batch, in_len, dim) -> (batch, out_len, dim)
        mix = torch.bmm(attn, context)

        # concat -> (batch, out_len, 2*dim)
        combined = torch.cat((mix, output), dim=2)
        # output -> (batch, out_len, dim)
        output = F.tanh(self.linear_out(combined.view(-1, 2 * hidden_size))).view(batch_size, -1, hidden_size)
        
        return output, attn