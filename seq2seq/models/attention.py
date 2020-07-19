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
        self.linear_out = nn.Linear(dim*2, dim)
        self.mask = None
        self.mask1 = None
        self.mask2 = None
        self.attn_mode = attn_mode # True (attention both pos and neg) # False (attention only pos samples)
        
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
        # context => ((pos_output, neg_output),(pos_set_output, neg_set_output))
        # pos_output => batch, set_size, pos_seq_len, hidden
        # neg_output => batch, set_size, neg_seq_len, hidden
        # pos_set_output => batch, set_size, hidden
        # neg_set_output => batch, set_size, hidden
        # dec_hidde => batch, dec_len, hidden
        1/0
        if self.attn_mode: # attention both pos and neg samples            
            pass
        else: # attention only pos samples
            pass
        return None