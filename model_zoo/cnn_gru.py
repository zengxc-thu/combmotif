
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import sys
sys.path.append("./")
class Permute_LN(nn.Module):
    """
    对channel通道进行layernorm
    输入是B*C*L的,所以要先permute
    然后对最后一个维度layernorm
    """
    def __init__(self, shape, eps=0):
        super(Permute_LN, self).__init__()
        self.ln = nn.LayerNorm(normalized_shape=shape, eps=eps)
    def forward(self, x):
        x = x.permute((0, 2, 1))
        x = self.ln(x)
        x = x.permute((0, 2, 1))
        return x

class Permute_GRU(nn.Module):

    def __init__(self, input_size,hidden_size,go_backwards=False,batch_first=True):
        super(Permute_GRU, self).__init__()
        # If batch_first is set to true, the expected input is [B, L (timestep), C]. Otherwise, the default second dimension is batch_ Size
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.go_backwards = go_backwards
    def forward(self, x):
        # The main issue with this gobackwards is to take the data from the last timestep instead of the first one
        x = x.permute((0, 2, 1))
        x, _= self.gru(x)
        x = x[:, -1, :] if self.go_backwards else x[:, 0, :]
        return x
    
    def weights_init(self):
        # pass
        for name, param in self.gru.named_parameters():
            if 'weight' in name:
                nn.init.kaiming_normal_(param)
            elif 'bias' in name:
                param.data.fill_(0)

class StochasticShift(nn.Module):
    def __init__(self, shift_max=0, symmetric=True, pad='uniform'):
        super(StochasticShift, self).__init__()
        self.shift_max = shift_max
        self.symmetric = symmetric
        
        
        if self.symmetric:
            self.augment_shifts = torch.arange(-self.shift_max, self.shift_max + 1)
        else:
            self.augment_shifts = torch.arange(0, self.shift_max + 1)
        
        self.pad = pad

    def forward(self, seq_1hot):
        if self.training:
            shift_i = torch.randint(0, len(self.augment_shifts), (1,), dtype=torch.int64).item()
            shift = self.augment_shifts[shift_i].item()
            
            if shift != 0:
                sseq_1hot = shift_sequence(seq_1hot, shift)
            else:
                sseq_1hot = seq_1hot
            return sseq_1hot
        else:
            return seq_1hot
    

def shift_sequence(seq, shift, pad_value=0):
    """
    Shift a sequence left or right by shift_amount.

    Args:
    seq: [batch_size, seq_length, seq_depth] sequence
    shift: signed shift value (int)
    pad_value: value to fill the padding

    Returns:
    sseq: Shifted sequence
    """
    if len(seq.shape) != 3:
        raise ValueError('input sequence should be rank 3')
    
    batch_size, seq_depth, seq_length = seq.shape
    
    pad = pad_value * torch.ones((batch_size, seq_depth, abs(shift)), dtype=seq.dtype)
    pad = pad.to(seq.device)
    def _shift_right(_seq):
        # shift is positive
        sliced_seq = _seq[:, :, :-shift]

        return torch.cat((pad, sliced_seq), dim=2)

    def _shift_left(_seq):
        # shift is negative
        sliced_seq = _seq[:, :, -shift:]
        return torch.cat((sliced_seq, pad), dim=2)

    sseq = _shift_right(seq) if shift > 0 else _shift_left(seq)
    sseq = sseq.view(batch_size, seq_depth, seq_length)

    return sseq



class Scale(nn.Module):
    def __init__(self, axis=-1, initializer='zeros'):
        super(Scale, self).__init__()
        if isinstance(axis, (list, tuple)):
            self.axis = list(axis)
        elif isinstance(axis, int):
            self.axis = [axis]
        else:
            raise TypeError('Expected an int or a list/tuple of ints for the '
                            'argument \'axis\', but received: %r' % axis)
        self.initializer = initializer

    def build(self, input_shape):
        # Convert axis to list and resolve negatives
        ndims = len(input_shape)
        for idx, x in enumerate(self.axis):
            if x < 0:
                self.axis[idx] = ndims + x

        # Validate axes
        for x in self.axis:
            if x < 0 or x >= ndims:
                raise ValueError('Invalid axis: %d' % x)
        if len(self.axis) != len(set(self.axis)):
            raise ValueError('Duplicate axis: {}'.format(tuple(self.axis)))

        param_shape = [input_shape[dim] for dim in self.axis]

        self.scale = nn.Parameter(torch.zeros(param_shape), requires_grad=True)

    def forward(self, x):
        expand_dims = [None] * len(x.shape)
        for axis in self.axis:
            expand_dims[axis] = slice(None)
        return x * self.scale[tuple(expand_dims)]

    def extra_repr(self):
        return f'axis={self.axis}, initializer={self.initializer}'




class saluki_torch(nn.Module):
    def __init__(self, seq_length=12288, seq_depth=6, augment_shift=3, filters=64, kernel_size=5,  padding = 0, l2_scale=0.001,
                 ln_epsilon=0.007, activation="relu", dropout=0.1, num_layers=6, go_backwards=True, rnn_type="gru", residual=False, bn_momentum=0.90, num_targets=1):
        super(saluki_torch, self).__init__()
        self.seq_length = seq_length
        self.seq_depth = seq_depth
        self.augment_shift = augment_shift
        self.filters = filters
        self.kernel_size = kernel_size
        self.l2_scale = l2_scale
        self.residual = residual
        self.ln_epsilon = ln_epsilon
        self.activation = activation
        self.dropout = dropout
        self.num_layers = num_layers
        self.go_backwards = go_backwards
        self.rnn_type = rnn_type
        self.bn_momentum = bn_momentum
        self.num_targets = num_targets



        # RNA convolution
        self.conv1 = nn.Conv1d(in_channels=seq_depth, out_channels=filters, kernel_size=kernel_size, padding=padding,
                                bias=False)
        if self.augment_shift != 0:
            self.shift_layer = StochasticShift(shift_max=self.augment_shift, symmetric=False, pad=0)
        
        
        self.Pln1 = Permute_LN(shape=(filters,), eps=self.ln_epsilon)

        # middle convolutions
        self.conv_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        self.ln_layers = nn.ModuleList()
        
        for _ in range(num_layers):
            self.ln_layers.append(Permute_LN(shape=(filters,), eps=self.ln_epsilon))
            self.conv_layers.append(nn.Conv1d(in_channels=filters, out_channels=filters, kernel_size=kernel_size,
                                              padding=padding, bias=True))


        
        # aggregate sequence

        self.bn1 = nn.BatchNorm1d(filters, momentum=bn_momentum)
        self.bn2 = nn.BatchNorm1d(filters, momentum=bn_momentum)
        self.fc1 = nn.Linear(filters, filters)
        self.head = nn.Linear(filters, self.num_targets)

        self.adapt_pool = nn.AdaptiveAvgPool1d(1)

        self.rnn_layer = Permute_GRU if rnn_type == 'gru' else nn.LSTM
        self.rnn = self.rnn_layer(input_size= filters, hidden_size=self.filters,batch_first=True, go_backwards=self.go_backwards)
        

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, Permute_GRU):
                m.weights_init()


       
    def forward(self, x):

        if self.training and self.augment_shift != 0:
            x = self.shift_layer(x)

        x = self.conv1(x)
        
        # middle convolutions
        for i in range(self.num_layers):
            x = self.ln_layers[i](x)
            x = F.relu(x)
            x = self.conv_layers[i](x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = F.max_pool1d(x,kernel_size=2)

        x = self.Pln1(x)
        x = F.relu(x)

        # x = self.adapt_pool(x)
        # x = x.squeeze(2)
        x = self.rnn(x)
        x = self.bn1(x)
        x = F.relu(x)
        # penultimate
        x = self.fc1(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.bn2(x)
        x = F.relu(x)

        # prediction head
        pred = self.head(x)

        return pred
    
if __name__=='__main__':
    # test model
    batch_size = 2
    input_shape = (batch_size, 6, 12288)
    input_data = torch.randn(input_shape)
    model = saluki_torch()
    with torch.no_grad():
        pred = model(input_data)
    print(pred)
