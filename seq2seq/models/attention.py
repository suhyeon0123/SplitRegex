import torch
import torch.nn as nn
import torch.nn.functional as F


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

    def set_mask(self, mask):
        """
        Sets indices to be masked

        Args:
            mask (torch.Tensor): tensor containing indices to be masked
        """
        self.mask = mask

    def forward(self, dec_hidden, context):
        '''
        dec_hidden -> decoder hidden state -> (#batch, #dec_len, #hidden*2) if not teacher_forcing_ratio: dec_len=1 
        context -> context is encoder outputs that is composed of two LSTM encoder outputs with tuple.
        context[0] -> (#batch, #set_num, #enc_len, #hidden*2)
        context[1] -> (#batch, #set_num, #hidden*2)
        '''
        batch_size = dec_hidden.size(0)
        hidden_size = dec_hidden.size(2)
        set_size = context[0].size(1)
        enc_len = context[0].size(2)
        dec_len = dec_hidden.size(1)
        
        # one phase attention 
        output1 = context[0]
        output1 = output1.view(batch_size, set_size*enc_len,-1)
        attn = torch.bmm(dec_hidden, output1.transpose(1,2))        
        
        if self.mask is not None:
            attn.data.masked_fill_(self.mask, -float('inf'))
            
        attn = attn.view(batch_size, dec_len, set_size, enc_len) 
        attn = F.softmax(attn, dim=-1)

        output1 = output1.view(batch_size, set_size, enc_len, -1)
        output1= output1.reshape(batch_size*set_size, enc_len, -1)
        attn = attn.transpose(1,2)
        attn = attn.reshape(batch_size*set_size, dec_len, enc_len)
        c_t = torch.bmm(attn, output1)
                
        # two phase attetion 
        1/0
        
        return output, attn
