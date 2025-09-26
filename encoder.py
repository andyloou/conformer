import math
import torch
from torch import nn
from collections import OrderedDict
import math
import torch
from torch import nn

# Assume INF_VAL is defined somewhere, typically a large number like 10000
INF_VAL = 10000.0
import torch
from torch import nn
from torch.nn import LayerNorm
import torch.nn.functional as F

import math
import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional
INF_VAL = 1e4

def calculate_correct_fan(tensor, mode):
    """Calculate fan_in or fan_out for tensor"""
    mode = mode.lower()
    valid_modes = ['fan_in', 'fan_out']
    if mode not in valid_modes:
        raise ValueError("Mode {} not supported, please use one of {}.".format(mode, valid_modes))

    num_input_fmaps = tensor.size(1)
    num_output_fmaps = tensor.size(0)
    receptive_field_size = 1
    if tensor.dim() > 2:
        receptive_field_size = tensor[0][0].numel()
    fan_in = num_input_fmaps * receptive_field_size
    fan_out = num_output_fmaps * receptive_field_size

    return fan_in if mode == 'fan_in' else fan_out


def tds_uniform_(tensor, mode='fan_in'):
    """
    Uniform Initialization from the paper [Sequence-to-Sequence Speech Recognition 
    with Time-Depth Separable Convolutions]
    """
    fan = calculate_correct_fan(tensor, mode)
    gain = 2.0  # sqrt(4.0) = 2
    std = gain / math.sqrt(fan)  # sqrt(4.0 / fan_in)
    bound = std  # Calculate uniform bounds from standard deviation
    with torch.no_grad():
        return tensor.uniform_(-bound, bound)


def tds_normal_(tensor, mode='fan_in'):
    """
    Normal Initialization from the TDS paper
    """
    fan = calculate_correct_fan(tensor, mode)
    gain = 2.0
    std = gain / math.sqrt(fan)
    with torch.no_grad():
        return tensor.normal_(0, std)


def init_weights(m, mode: Optional[str] = 'xavier_uniform'):
    """Enhanced weight initialization matching the original NeMo version"""
    if isinstance(m, (nn.Conv1d, nn.Linear)):
        if mode is not None:
            if mode == 'xavier_uniform':
                nn.init.xavier_uniform_(m.weight, gain=1.0)
            elif mode == 'xavier_normal':
                nn.init.xavier_normal_(m.weight, gain=1.0)
            elif mode == 'kaiming_uniform':
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
            elif mode == 'kaiming_normal':
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            elif mode == 'tds_uniform':
                tds_uniform_(m.weight)
            elif mode == 'tds_normal':
                tds_normal_(m.weight)
            else:
                raise ValueError("Unknown Initialization mode: {0}".format(mode))
        
        # Initialize bias to zero if it exists
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.zeros_(m.bias)
            
    elif isinstance(m, nn.BatchNorm1d):
        if m.track_running_stats:
            m.running_mean.zero_()
            m.running_var.fill_(1)
            m.num_batches_tracked.zero_()
        if m.affine:
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

class avoid_float16_autocast_context:
    """Dummy context manager for autocast handling"""
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        pass

class ConvASRDecoder(nn.Module):
    """
    Simplified ASR Decoder for CTC-based models.
    
    Args:
        feat_in (int): Input feature dimension (176 in your config)
        num_classes (int): Number of vocabulary classes (1024 in your config) 
        init_mode (str): Weight initialization method
        vocabulary (list): Vocabulary list (optional, can infer from num_classes)
        add_blank (bool): Whether to add blank token for CTC (default True)
    """

    def __init__(self, feat_in=176, num_classes=1024, init_mode="xavier_uniform", vocabulary=None, add_blank=True):
        super().__init__()

        if vocabulary is None and num_classes < 0:
            raise ValueError("Neither vocabulary nor num_classes are set! At least one needs to be set.")

        # If vocabulary is provided, use its length
        if vocabulary is not None:
            if num_classes > 0 and num_classes != len(vocabulary):
                raise ValueError(
                    f"If vocabulary is specified, its length should equal num_classes. "
                    f"Got: num_classes={num_classes} and len(vocabulary)={len(vocabulary)}"
                )
            num_classes = len(vocabulary)
            self.vocabulary = vocabulary
        else:
            self.vocabulary = None

        if num_classes <= 0:
            raise ValueError("num_classes must be positive")

        self._feat_in = feat_in
        # Add 1 for blank token in CTC
        self._num_classes = num_classes + 1 if add_blank else num_classes

        # Simple 1x1 convolution to project from encoder features to vocabulary logits
        self.decoder_layers = nn.Sequential(
            nn.Conv1d(self._feat_in, self._num_classes, kernel_size=1, bias=True)
        )

        # Initialize weights
        self.apply(lambda x: init_weights(x, mode=init_mode))

        # Temperature for softmax (can be changed for inference)
        self.temperature = 1.0

    def forward(self, encoder_output):
        """
        Forward pass of the decoder.
        
        Args:
            encoder_output (torch.Tensor): Output from encoder [B, feat_in, T]
            
        Returns:
            torch.Tensor: Log probabilities [B, T, num_classes]
        """
        # Apply 1x1 conv: [B, feat_in, T] -> [B, num_classes, T]
        logits = self.decoder_layers(encoder_output)
        
        # Transpose to [B, T, num_classes] for CTC loss computation
        logits = logits.transpose(1, 2)
        
        # Apply temperature scaling if needed
        if self.temperature != 1.0:
            logits = logits / self.temperature
            
        # Return log probabilities
        return F.log_softmax(logits, dim=-1)

    @property
    def num_classes_with_blank(self):
        """Return total number of classes including blank token"""
        return self._num_classes

    def input_example(self, max_batch=1, max_dim=256):
        """Generate input examples for tracing/export"""
        device = next(self.parameters()).device
        input_example = torch.randn(max_batch, self._feat_in, max_dim, device=device)
        return (input_example,)
    
class RelPositionMultiHeadAttention(nn.Module):
    def __init__(
        self,
        n_head,
        n_feat,
        dropout_rate,
        pos_bias_u=None,
        pos_bias_v=None,
        max_cache_len=0,
        use_bias=True,
        use_pytorch_sdpa=False,  # Simplified - always False
        use_pytorch_sdpa_backends=None,
    ):
        super(RelPositionMultiHeadAttention, self).__init__()
        
        assert n_feat % n_head == 0
        self.d_k = n_feat // n_head
        self.s_d_k = math.sqrt(self.d_k)
        self.h = n_head
        self.dropout_rate = dropout_rate
        self.use_bias = use_bias
        
        # Linear layers for Q, K, V projections
        self.linear_q = nn.Linear(n_feat, n_feat, bias=use_bias)
        self.linear_k = nn.Linear(n_feat, n_feat, bias=use_bias)
        self.linear_v = nn.Linear(n_feat, n_feat, bias=use_bias)
        self.linear_out = nn.Linear(n_feat, n_feat, bias=use_bias)
        
        # Linear transformation for positional encoding
        self.linear_pos = nn.Linear(n_feat, n_feat, bias=False)
        
        # Learnable biases for relative attention computation
        if pos_bias_u is None or pos_bias_v is None:
            self.pos_bias_u = nn.Parameter(torch.FloatTensor(self.h, self.d_k))
            self.pos_bias_v = nn.Parameter(torch.FloatTensor(self.h, self.d_k))
            nn.init.zeros_(self.pos_bias_u)
            nn.init.zeros_(self.pos_bias_v)
        else:
            self.pos_bias_u = pos_bias_u
            self.pos_bias_v = pos_bias_v
            
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward_qkv(self, query, key, value):
        """Transform query, key and value."""
        n_batch = query.size(0)
        q = self.linear_q(query).view(n_batch, -1, self.h, self.d_k)
        k = self.linear_k(key).view(n_batch, -1, self.h, self.d_k)
        v = self.linear_v(value).view(n_batch, -1, self.h, self.d_k)
        q = q.transpose(1, 2)  # (batch, head, time, d_k)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        return q, k, v

    def rel_shift(self, x):
        """Compute relative positional encoding shift."""
        b, h, qlen, pos_len = x.size()
        # Add a column of zeros on the left side
        x = torch.nn.functional.pad(x, pad=(1, 0))  # (b, h, t1, t2+1)
        x = x.view(b, h, -1, qlen)  # (b, h, t2+1, t1)
        # Drop the first row
        x = x[:, :, 1:].view(b, h, qlen, pos_len)  # (b, h, t1, t2)
        return x

    def forward_attention(self, value, scores, mask):
        """Compute attention context vector."""
        n_batch = value.size(0)
        if mask is not None:
            mask = mask.unsqueeze(1)  # (batch, 1, time1, time2)
            scores = scores.masked_fill(mask, -INF_VAL)
            attn = torch.softmax(scores, dim=-1).masked_fill(mask, 0.0)
        else:
            attn = torch.softmax(scores, dim=-1)

        p_attn = self.dropout(attn)
        x = torch.matmul(p_attn, value)  # (batch, head, time1, d_k)
        x = x.transpose(1, 2).reshape(n_batch, -1, self.h * self.d_k)  # (batch, time1, d_model)
        return self.linear_out(x)

    def forward(self, query, key, value, mask, pos_emb, cache=None):
        """
        Compute scaled dot product attention with relative positional encoding.
        
        Args:
            query (torch.Tensor): (batch, time1, size)
            key (torch.Tensor): (batch, time2, size)
            value (torch.Tensor): (batch, time2, size)
            mask (torch.Tensor): (batch, time1, time2)
            pos_emb (torch.Tensor): (batch, time1, size)
            cache: Not used in simplified version
            
        Returns:
            output (torch.Tensor): (batch, time1, d_model)
        """
        # Handle autocast if enabled
        if torch.is_autocast_enabled():
            query, key, value = query.to(torch.float32), key.to(torch.float32), value.to(torch.float32)

        with avoid_float16_autocast_context():
            # Get Q, K, V projections
            q, k, v = self.forward_qkv(query, key, value)
            q = q.transpose(1, 2)  # (batch, time1, head, d_k)

            # Process positional embeddings
            n_batch_pos = pos_emb.size(0)
            n_batch = value.size(0)
            p = self.linear_pos(pos_emb).view(n_batch_pos, -1, self.h, self.d_k)
            p = p.transpose(1, 2)  # (batch, head, time1, d_k)

            # Add positional biases
            q_with_bias_u = (q + self.pos_bias_u).transpose(1, 2)  # (batch, head, time1, d_k)
            q_with_bias_v = (q + self.pos_bias_v).transpose(1, 2)

            # Compute attention scores
            # Matrix AC: content-content attention
            matrix_ac = torch.matmul(q_with_bias_u, k.transpose(-2, -1))
            
            # Matrix BD: content-position attention
            matrix_bd = torch.matmul(q_with_bias_v, p.transpose(-2, -1))
            matrix_bd = self.rel_shift(matrix_bd)
            
            # Ensure BD matrix matches AC matrix size
            matrix_bd = matrix_bd[:, :, :, :matrix_ac.size(-1)]
            
            # Combine both attention components
            scores = (matrix_ac + matrix_bd) / self.s_d_k
            
            # Apply attention
            output = self.forward_attention(v, scores, mask)

        return output
    
class Swish(nn.Module):
    """Swish activation function"""
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)


class ConformerFeedForward(nn.Module):
    """
    Simplified feed-forward module of Conformer model.
    """
    def __init__(self, d_model, d_ff, dropout, use_bias=True):
        super(ConformerFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff, bias=use_bias)
        self.activation = Swish()
        self.dropout = nn.Dropout(p=dropout)
        self.linear2 = nn.Linear(d_ff, d_model, bias=use_bias)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class ConformerConvolution(nn.Module):
    """
    Simplified convolution module for the Conformer model.
    Uses standard PyTorch Conv1d instead of CausalConv1D since your config doesn't need causal convolutions.
    """
    def __init__(self, d_model, kernel_size, norm_type='batch_norm', use_bias=True):
        super(ConformerConvolution, self).__init__()
        assert (kernel_size - 1) % 2 == 0
        
        self.d_model = d_model
        self.kernel_size = kernel_size
        self.use_bias = use_bias
        padding = (kernel_size - 1) // 2

        # Pointwise conv 1 (1x1 conv that doubles channels for GLU)
        self.pointwise_conv1 = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model * 2,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=use_bias,
        )

        # Depthwise conv (using standard Conv1d with groups=d_model)
        self.depthwise_conv = nn.Conv1d(
            in_channels=d_model,  # After GLU, channels are halved back to d_model
            out_channels=d_model,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            groups=d_model,  # Depthwise convolution
            bias=use_bias,
        )

        # Normalization
        if norm_type == 'batch_norm':
            self.batch_norm = nn.BatchNorm1d(d_model)
        elif norm_type == 'layer_norm':
            self.batch_norm = nn.LayerNorm(d_model)
        else:
            raise ValueError(f"conv_norm_type={norm_type} is not supported!")

        self.activation = Swish()
        
        # Pointwise conv 2 (1x1 conv back to d_model)
        self.pointwise_conv2 = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=use_bias,
        )

    def forward(self, x, pad_mask=None, cache=None):
        # x: [B, T, d_model] -> [B, d_model, T] for Conv1d
        x = x.transpose(1, 2)
        
        # First pointwise conv
        x = self.pointwise_conv1(x)  # [B, d_model*2, T]
        
        # GLU activation (splits channels and applies sigmoid gating)
        x = F.glu(x, dim=1)  # [B, d_model, T]
        
        # Apply padding mask if provided
        if pad_mask is not None:
            x = x.masked_fill(pad_mask.unsqueeze(1), 0.0)

        # Depthwise convolution
        x = self.depthwise_conv(x)  # [B, d_model, T]

        # Normalization
        if isinstance(self.batch_norm, nn.LayerNorm):
            # For LayerNorm, need to transpose back
            x = x.transpose(1, 2)  # [B, T, d_model]
            x = self.batch_norm(x)
            x = x.transpose(1, 2)  # [B, d_model, T]
        else:
            x = self.batch_norm(x)

        # Activation
        x = self.activation(x)
        
        # Second pointwise conv
        x = self.pointwise_conv2(x)  # [B, d_model, T]
        
        # Transpose back to [B, T, d_model]
        x = x.transpose(1, 2)
        
        return x


class ConformerLayer(nn.Module):
    def __init__(
        self,
        d_model,
        d_ff,
        self_attention_model='rel_pos',
        n_heads=4,
        conv_kernel_size=31,
        conv_norm_type='batch_norm',
        dropout=0.1,
        dropout_att=0.1,
        pos_bias_u=None,
        pos_bias_v=None,
        att_context_size=[-1, -1],
        use_bias=True,
    ):
        super(ConformerLayer, self).__init__()
        
        if self_attention_model != 'rel_pos':
            raise ValueError("This simplified version only supports 'rel_pos' attention model")

        self.self_attention_model = self_attention_model
        self.fc_factor = 0.5

        # First feed forward module
        self.norm_feed_forward1 = LayerNorm(d_model)
        self.feed_forward1 = ConformerFeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout, use_bias=use_bias)

        # Convolution module
        self.norm_conv = LayerNorm(d_model)
        self.conv = ConformerConvolution(
            d_model=d_model,
            kernel_size=conv_kernel_size,
            norm_type=conv_norm_type,
            use_bias=use_bias,
        )

        # Multi-headed self-attention module
        self.norm_self_att = LayerNorm(d_model)
        self.self_attn = RelPositionMultiHeadAttention(
            n_head=n_heads,
            n_feat=d_model,
            dropout_rate=dropout_att,
            pos_bias_u=pos_bias_u,
            pos_bias_v=pos_bias_v,
            max_cache_len=att_context_size[0],
            use_bias=use_bias,
        )

        # Second feed forward module
        self.norm_feed_forward2 = LayerNorm(d_model)
        self.feed_forward2 = ConformerFeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout, use_bias=use_bias)

        self.dropout = nn.Dropout(dropout)
        self.norm_out = LayerNorm(d_model)

    def forward(self, x, att_mask=None, pos_emb=None, pad_mask=None, cache_last_channel=None, cache_last_time=None):
        """
        Simplified forward pass without caching support.
        Args:
            x (torch.Tensor): input signals (B, T, d_model)
            att_mask (torch.Tensor): attention masks(B, T, T)
            pos_emb (torch.Tensor): (L, 1, d_model)
            pad_mask (torch.tensor): padding mask
        Returns:
            x (torch.Tensor): (B, T, d_model)
        """
        # First feed forward module (half residual connection)
        residual = x
        x = self.norm_feed_forward1(x)
        x = self.feed_forward1(x)
        residual = residual + self.dropout(x) * self.fc_factor

        # Multi-headed self-attention module
        x = self.norm_self_att(residual)
        x = self.self_attn(query=x, key=x, value=x, mask=att_mask, pos_emb=pos_emb)
        
        residual = residual + self.dropout(x)

        # Convolution module
        x = self.norm_conv(residual)
        x = self.conv(x, pad_mask=pad_mask)
        residual = residual + self.dropout(x)

        # Second feed forward module (half residual connection)
        x = self.norm_feed_forward2(residual)
        x = self.feed_forward2(x)
        residual = residual + self.dropout(x) * self.fc_factor

        # Final layer norm
        x = self.norm_out(residual)

        return x
class ConvSubsampling(nn.Module):
    def __init__(
        self,
        subsampling='striding',
        subsampling_factor=4,
        feat_in=80,
        feat_out=176,
        conv_channels=176,
        subsampling_conv_chunking_factor=1,
        activation=nn.ReLU(),
        is_causal=False,
    ):
        super(ConvSubsampling, self).__init__()
        
        if subsampling != 'striding':
            raise ValueError(f"This simplified version only supports 'striding', got: {subsampling}")
            
        if subsampling_factor % 2 != 0:
            raise ValueError("Sampling factor should be a multiply of 2!")
            
        self._subsampling = subsampling
        self._conv_channels = conv_channels
        self._feat_in = feat_in
        self._feat_out = feat_out
        self.subsampling_factor = subsampling_factor
        self.is_causal = is_causal
        self.subsampling_conv_chunking_factor = subsampling_conv_chunking_factor

        # Calculate number of sampling layers needed
        self._sampling_num = int(math.log(subsampling_factor, 2))
        
        # Striding parameters
        self._stride = 2
        self._kernel_size = 3
        self._ceil_mode = False

        # Padding setup (non-causal since is_causal=False in your config)
        self._left_padding = (self._kernel_size - 1) // 2
        self._right_padding = (self._kernel_size - 1) // 2

        # Build striding convolution layers
        layers = []
        in_channels = 1  # Start with 1 channel for 2D conv

        for i in range(self._sampling_num):
            layers.append(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=conv_channels,
                    kernel_size=self._kernel_size,
                    stride=self._stride,
                    padding=self._left_padding,
                )
            )
            layers.append(activation)
            in_channels = conv_channels

        self.conv = nn.Sequential(*layers)

        # Calculate output feature size after convolution
        in_length = torch.tensor(feat_in, dtype=torch.float)
        out_length = calc_length(
            lengths=in_length,
            all_paddings=self._left_padding + self._right_padding,
            kernel_size=self._kernel_size,
            stride=self._stride,
            ceil_mode=self._ceil_mode,
            repeat_num=self._sampling_num,
        )
        
        # Final linear projection
        self.out = nn.Linear(conv_channels * int(out_length), feat_out)

    def get_sampling_frames(self):
        """Return sampling frames for streaming support."""
        return [1, self.subsampling_factor]

    def get_streaming_cache_size(self):
        """Return cache size for streaming support."""
        return [0, self.subsampling_factor + 1]

    def forward(self, x, lengths):
        """
        Forward pass of ConvSubsampling.
        
        Args:
            x: Input tensor [B, T, feat_in]
            lengths: Length of each sequence [B]
            
        Returns:
            tuple: (output_tensor [B, T', feat_out], output_lengths [B])
        """
        # Calculate output lengths
        out_lengths = calc_length(
            lengths,
            all_paddings=self._left_padding + self._right_padding,
            kernel_size=self._kernel_size,
            stride=self._stride,
            ceil_mode=self._ceil_mode,
            repeat_num=self._sampling_num,
        )

        # Add channel dimension and transpose to [B, C, T, F] for 2D conv
        x = x.unsqueeze(1)  # [B, 1, T, feat_in]
        
        # Apply convolution layers
        x = self.conv(x)  # [B, conv_channels, T', F']

        # Flatten channel and frequency dimensions
        b, c, t, f = x.size()
        x = x.transpose(1, 2).reshape(b, t, -1)  # [B, T', C*F']
        
        # Apply final linear projection
        x = self.out(x)  # [B, T', feat_out]

        return x, out_lengths

    def change_subsampling_conv_chunking_factor(self, subsampling_conv_chunking_factor: int):
        """Change the chunking factor for convolution processing."""
        if (
            subsampling_conv_chunking_factor != -1
            and subsampling_conv_chunking_factor != 1
            and subsampling_conv_chunking_factor % 2 != 0
        ):
            raise ValueError("subsampling_conv_chunking_factor should be -1, 1, or a power of 2")
        self.subsampling_conv_chunking_factor = subsampling_conv_chunking_factor

def calc_length(lengths, all_paddings, kernel_size, stride, ceil_mode, repeat_num=1):
    """Calculates the output length of a Tensor passed through a convolution or max pooling layer"""
    add_pad: float = all_paddings - kernel_size
    one: float = 1.0
    for i in range(repeat_num):
        lengths = torch.div(lengths.to(dtype=torch.float) + add_pad, stride) + one
        if ceil_mode:
            lengths = torch.ceil(lengths)
        else:
            lengths = torch.floor(lengths)
    return lengths.to(dtype=torch.int)
class RelPositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout_rate, max_len=5000, xscale=None, dropout_rate_emb=0.0):
        """Construct a RelPositionalEncoding object."""
        super(RelPositionalEncoding, self).__init__()
        self.d_model = d_model
        self.xscale = xscale
        self.dropout = nn.Dropout(p=dropout_rate)
        self.max_len = max_len
        
        if dropout_rate_emb > 0:
            self.dropout_emb = nn.Dropout(dropout_rate_emb)
        else:
            self.dropout_emb = None

    def create_pe(self, positions, dtype):
        """Create positional encodings for given positions."""
        pos_length = positions.size(0)
        pe = torch.zeros(pos_length, self.d_model, device=positions.device)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32, device=positions.device)
            * -(math.log(INF_VAL) / self.d_model)
        )
        pe[:, 0::2] = torch.sin(positions * div_term)
        pe[:, 1::2] = torch.cos(positions * div_term)
        pe = pe.unsqueeze(0).to(dtype)
        
        if hasattr(self, 'pe'):
            self.pe = pe
        else:
            self.register_buffer('pe', pe, persistent=False)

    def extend_pe(self, length, device, dtype):
        """Reset and extend the positional encodings if needed for relative positioning."""
        needed_size = 2 * length - 1
        if hasattr(self, 'pe') and self.pe.size(1) >= needed_size:
            return
            
        # positions would be from negative numbers to positive
        # positive positions would be used for left positions and negative for right positions
        positions = torch.arange(length - 1, -length, -1, dtype=torch.float32, device=device).unsqueeze(1)
        self.create_pe(positions=positions, dtype=dtype)

    def forward(self, x, cache_len=0):
        """Compute relative positional encoding.
        Args:
            x (torch.Tensor): Input. Its shape is (batch, time, feature_size)
            cache_len (int): the size of the cache which is used to shift positions
        Returns:
            x (torch.Tensor): Its shape is (batch, time, feature_size)
            pos_emb (torch.Tensor): Its shape is (1, time, feature_size)
        """
        if self.xscale:
            x = x * self.xscale

        # center_pos would be the index of position 0
        # negative positions would be used for right and positive for left tokens
        # for input of length L, 2*L-1 positions are needed, positions from (L-1) to -(L-1)
        input_len = x.size(1) + cache_len
        center_pos = self.pe.size(1) // 2 + 1
        start_pos = center_pos - input_len
        end_pos = center_pos + input_len - 1
        pos_emb = self.pe[:, start_pos:end_pos]
        
        if self.dropout_emb:
            pos_emb = self.dropout_emb(pos_emb)
            
        return self.dropout(x), pos_emb
class FilterbankFeatures(nn.Module):
    def __init__(
        self,
        sample_rate=16000,
        n_window_size=400,
        n_window_stride=160,
        window="hann",
        normalize="per_feature",
        n_fft=512,
        preemph=0.97,
        nfilt=80,
        lowfreq=0,
        highfreq=None,
        log=True,
        log_zero_guard_type="add",
        log_zero_guard_value=2**-24,
        dither=1e-5,
        pad_to=0,
        frame_splicing=1,
        exact_pad=False,
        pad_value=0,
        mag_power=2.0,
        rng=None,
        nb_augmentation_prob=0.0,
        nb_max_freq=4000,
        mel_norm="slaney",
        stft_exact_pad=False,
        stft_conv=False,
    ):
        super().__init__()
        if rng is None:
            rng = random.Random()
        self.rng = rng
        if highfreq is None:
            highfreq = sample_rate / 2
        self.preemph = preemph
        self.n_fft = n_fft or n_window_size
        self.nfilt = nfilt
        self.normalize = normalize
        self.log = log
        self.dither = dither
        self.frame_splicing = frame_splicing
        self.n_window_size = n_window_size
        self.n_window_stride = n_window_stride
        self.pad_to = pad_to
        self.exact_pad = exact_pad
        self.pad_value = pad_value
        self.log_zero_guard_type = log_zero_guard_type
        self.log_zero_guard_value = log_zero_guard_value
        self.mag_power = mag_power
        self.nb_augmentation_prob = nb_augmentation_prob
        self.nb_max_freq = nb_max_freq
        self.mel_norm = mel_norm
        torch_windows = {
            'hann': torch.hann_window,
            'hamming': torch.hamming_window,
            'blackman': torch.blackman_window,
            'bartlett': torch.bartlett_window,
            'ones': torch.ones,
            None: torch.ones,
        }
        self.win_length = n_window_size
        self.hop_length = n_window_stride
        self.register_buffer("window", torch_windows[window](n_window_size, periodic=False))
        if exact_pad:
            self.stft_pad_amount = n_window_size // 2
        else:
            self.stft_pad_amount = None
        mel_basis = librosa.filters.mel(
            sr=sample_rate,
            n_fft=self.n_fft,
            n_mels=nfilt,
            fmin=lowfreq,
            fmax=highfreq,
            htk=False,
            norm=mel_norm if mel_norm else None
        )
        self.register_buffer("fb", torch.tensor(mel_basis.T).float())

    @torch.no_grad()
    def forward(self, audio, length):
        batch_size = audio.size(0)
        if self.dither > 0:
            audio += self.dither * torch.randn_like(audio)
        if self.preemph is not None:
            preemph_audio = audio.new_zeros(audio.shape)
            preemph_audio[:, 1:] = audio[:, 1:] - self.preemph * audio[:, :-1]
            preemph_audio[:, 0] = audio[:, 0]
            audio = preemph_audio
        if self.exact_pad:
            pad_amount = self.stft_pad_amount
            audio = F.pad(audio.unsqueeze(1), (pad_amount, pad_amount), mode="reflect").squeeze(1)
            length += 2 * pad_amount
        else:
            pad_amount = (self.n_window_size - self.n_window_stride) // 2
            needed_length = pad_amount + math.ceil((audio.size(1) - pad_amount) / self.n_window_stride) * self.n_window_stride
            if needed_length > audio.size(1):
                pad_right = needed_length - audio.size(1)
                audio = F.pad(audio, (0, pad_right), mode="reflect")
        fft = torch.fft.rfft(audio, n=self.n_fft)
        mag = torch.abs(fft)
        power = mag ** self.mag_power
        mel = torch.matmul(power, self.fb)
        if self.log:
            if self.log_zero_guard_type == "add":
                mel = torch.log(mel + self.log_zero_guard_value)
            elif self.log_zero_guard_type == "clamp":
                mel = torch.clamp(mel, min=self.log_zero_guard_value).log()
        if self.normalize == "per_feature":
            mean = mel.mean(dim=-1, keepdim=True)
            std = mel.std(dim=-1, keepdim=True) + 1e-5
            mel = (mel - mean) / std
        elif self.normalize == "all_features":
            mean = mel.mean(keepdim=True)
            std = mel.std(keepdim=True) + 1e-5
            mel = (mel - mean) / std
        if self.frame_splicing > 1:
            mel = mel.reshape(mel.size(0), mel.size(1) // self.frame_splicing, mel.size(1) * self.frame_splicing)
        if self.pad_to > 0:
            N = mel.size(-1)
            P = self.pad_to - N % self.pad_to
            if P > 0:
                mel = F.pad(mel, (0, P), value=self.pad_value)
        length = (length - self.n_window_size) // self.n_window_stride + 1
        return mel, length

# AudioToMelSpectrogramPreprocessor
class AudioToMelSpectrogramPreprocessor(nn.Module):
    def __init__(
        self,
        sample_rate=16000,
        window_size=0.025,
        window_stride=0.01,
        window="hann",
        normalize="per_feature",
        n_fft=512,
        log=True,
        frame_splicing=1,
        dither=1.0e-5,
        pad_to=0,
        pad_value=0.0,
        features=80,
        lowfreq=0,
        highfreq=None,
        log_zero_guard_type="add",
        log_zero_guard_value=2**-24,
        mag_power=2.0,
        preemph=0.97,
        exact_pad=False,
        mel_norm="slaney",
    ):
        super().__init__()
        n_window_size = int(window_size * sample_rate)
        n_window_stride = int(window_stride * sample_rate)
        self._sample_rate = sample_rate
        self.featurizer = FilterbankFeatures(
            sample_rate=sample_rate,
            n_window_size=n_window_size,
            n_window_stride=n_window_stride,
            window=window,
            normalize=normalize,
            n_fft=n_fft,
            preemph=preemph,
            nfilt=features,
            lowfreq=lowfreq,
            highfreq=highfreq,
            log=log,
            log_zero_guard_type=log_zero_guard_type,
            log_zero_guard_value=log_zero_guard_value,
            dither=dither,
            pad_to=pad_to,
            frame_splicing=frame_splicing,
            exact_pad=exact_pad,
            pad_value=pad_value,
            mag_power=mag_power,
            mel_norm=mel_norm,
        )
        self.register_buffer("dtype_sentinel_tensor", torch.tensor((), dtype=torch.float32), persistent=False)

    @typecheck()
    @torch.no_grad()
    def forward(self, input_signal, length):
        if input_signal.dtype != torch.float32:
            input_signal = input_signal.to(torch.float32)
        processed_signal, processed_length = self.featurizer(input_signal, length)
        processed_signal = processed_signal.to(self.dtype_sentinel_tensor.dtype)
        return processed_signal, processed_length

    @property
    def filter_banks(self):
        return self.featurizer.fb

# SpecAugment
class SpecAugment(nn.Module):
    def __init__(self, freq_masks=2, time_masks=5, freq_width=27, time_width=0.05, rng=None, mask_value=0.0, use_vectorized_code=True):
        super().__init__()
        self.freq_masks = freq_masks
        self.time_masks = time_masks
        self.freq_width = freq_width
        self.time_width = time_width
        self.mask_value = mask_value
        self.use_vectorized_code = use_vectorized_code
        if rng is None:
            rng = random.Random()
        self.rng = rng

    @torch.no_grad()
    def forward(self, input_spec, length=None):
        B, F, T = input_spec.size()
        if self.use_vectorized_code:
            if self.freq_masks > 0:
                for idx in range(B):
                    for _ in range(self.freq_masks):
                        x = self.rng.randint(0, F - self.freq_width)
                        input_spec[idx, x : x + self.freq_width, :] = self.mask_value
            if self.time_masks > 0:
                if length is None:
                    length = input_spec.new_full((B,), T, dtype=torch.int64)
                time_width = self.time_width
                if isinstance(time_width, float):
                    time_width = int(time_width * length.max().item())
                for idx in range(B):
                    l = length[idx].item()
                    for _ in range(self.time_masks):
                        x = self.rng.randint(0, l - time_width)
                        input_spec[idx, :, x : x + time_width] = self.mask_value
        else:
            if self.freq_masks > 0:
                for _ in range(self.freq_masks):
                    x = self.rng.randint(0, F - self.freq_width)
                    input_spec[:, x : x + self.freq_width, :] = self.mask_value
            if self.time_masks > 0:
                if length is None:
                    num_cols = T
                else:
                    num_cols = length.min().item()
                for _ in range(self.time_masks):
                    x = self.rng.randint(0, num_cols - self.time_width)
                    input_spec[:, :, x : x + self.time_width] = self.mask_value
        return input_spec

# SpectrogramAugmentation
class SpectrogramAugmentation(nn.Module):
    def __init__(
        self,
        freq_masks=2,
        time_masks=5,
        freq_width=27,
        time_width=0.05,
        rect_masks=0,
        rect_time=5,
        rect_freq=20,
        rng=None,
        mask_value=0.0,
        use_vectorized_spec_augment=True,
        use_numba_spec_augment=False,
    ):
        super().__init__()
        if rect_masks > 0:
            self.spec_cutout = SpecCutout(
                rect_masks=rect_masks,
                rect_time=rect_time,
                rect_freq=rect_freq,
                rng=rng,
            )
        else:
            self.spec_cutout = lambda x: x
        if freq_masks + time_masks > 0:
            self.spec_augment = SpecAugment(
                freq_masks=freq_masks,
                time_masks=time_masks,
                freq_width=freq_width,
                time_width=time_width,
                rng=rng,
                mask_value=mask_value,
                use_vectorized_code=use_vectorized_spec_augment,
            )
        else:
            self.spec_augment = lambda x, length=None: x
        self.spec_augment_numba = None

    @typecheck()
    def forward(self, input_spec, length):
        augmented_spec = self.spec_cutout(input_spec)
        if self.spec_augment_numba is not None and spec_augment_launch_heuristics(augmented_spec, length):
            augmented_spec = self.spec_augment_numba(input_spec=augmented_spec, length=length)
        else:
            augmented_spec = self.spec_augment(input_spec=augmented_spec, length=length)
        return augmented_spec

# SpecCutout
class SpecCutout(nn.Module):
    def __init__(self, rect_masks=0, rect_time=5, rect_freq=20, rng=None):
        super().__init__()
        self.rect_masks = rect_masks
        self.rect_time = rect_time
        self.rect_freq = rect_freq
        if rng is None:
            rng = random.Random()
        self.rng = rng

    @torch.no_grad()
    def forward(self, input_spec):
        sh = input_spec.shape
        for _ in range(self.rect_masks):
            t0 = self.rng.randint(0, sh[2] - self.rect_time)
            t1 = t0 + self.rect_time
            f0 = self.rng.randint(0, sh[1] - self.rect_freq)
            f1 = f0 + self.rect_freq
            input_spec[:, f0:f1, t0:t1] = 0
        return input_spec

# spec_augment_launch_heuristics
def spec_augment_launch_heuristics(input_spec, length):
    return input_spec.is_cuda and length is not None

class ConformerEncoder(nn.Module):
    # Removed input_types and output_types properties since they're just for NeMo type checking

    def __init__(
        self,
        feat_in=80,
        n_layers=16,
        d_model=176,
        feat_out=-1,
        subsampling='striding',
        subsampling_factor=4,
        subsampling_conv_channels=176,
        ff_expansion_factor=4,
        self_attention_model='rel_pos',
        n_heads=4,
        att_context_size=[-1, -1],
        xscaling=True,
        untie_biases=True,
        pos_emb_max_len=5000,
        conv_kernel_size=31,
        dropout=0.1,
        dropout_emb=0.0,
        dropout_att=0.1,
    ):
        super().__init__()
        
        # Store basic config
        self.d_model = d_model
        self.n_layers = n_layers
        self._feat_in = feat_in
        self.subsampling_factor = subsampling_factor
        self.att_context_size = att_context_size
        self.pos_emb_max_len = pos_emb_max_len
        
        d_ff = d_model * ff_expansion_factor

        # Set up xscaling
        if xscaling:
            self.xscale = math.sqrt(d_model)
        else:
            self.xscale = None

        # Subsampling layer
        self.pre_encode = ConvSubsampling(
            subsampling=subsampling,
            subsampling_factor=subsampling_factor,
            feat_in=feat_in,
            feat_out=d_model,
            conv_channels=subsampling_conv_channels,
            subsampling_conv_chunking_factor=1,  # Default value
            activation=nn.ReLU(True),
            is_causal=False,  # Default value
        )

        # Positional encoding (only rel_pos since that's what your config uses)
        self.pos_enc = RelPositionalEncoding(
            d_model=d_model,
            dropout_rate=dropout,  # dropout_pre_encoder equivalent
            max_len=pos_emb_max_len,
            xscale=self.xscale,
            dropout_rate_emb=dropout_emb,
        )

        # Biases for relative positional encoding (untied since untie_biases=True)
        # Each layer will have its own bias parameters
        pos_bias_u = None
        pos_bias_v = None

        # Conformer layers
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            layer = ConformerLayer(
                d_model=d_model,
                d_ff=d_ff,
                self_attention_model=self_attention_model,
                n_heads=n_heads,
                conv_kernel_size=conv_kernel_size,
                conv_norm_type='batch_norm',  # Default value
                dropout=dropout,
                dropout_att=dropout_att,
                pos_bias_u=pos_bias_u,
                pos_bias_v=pos_bias_v,
                att_context_size=att_context_size,
                use_bias=True,  # Default value
            )
            self.layers.append(layer)

        # Output projection (not used since feat_out=-1)
        if feat_out > 0 and feat_out != d_model:
            self.out_proj = nn.Linear(d_model, feat_out)
            self._feat_out = feat_out
        else:
            self.out_proj = None
            self._feat_out = d_model

        # Initialize max audio length
        self.max_audio_length = pos_emb_max_len
        self.set_max_audio_length(pos_emb_max_len)

    def set_max_audio_length(self, max_audio_length):
        """Sets maximum input length and extends positional encoding if needed."""
        self.max_audio_length = max_audio_length
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        self.pos_enc.extend_pe(max_audio_length, device, dtype)

    def update_max_seq_length(self, seq_length: int, device):
        """Updates the maximum sequence length for the model."""
        if seq_length > self.max_audio_length:
            self.set_max_audio_length(seq_length)

    def _create_masks(self, att_context_size, padding_length, max_audio_length, device):
        """Create attention and padding masks."""
        # Attention mask (unlimited context since att_context_size=[-1, -1])
        att_mask = torch.ones(1, max_audio_length, max_audio_length, dtype=torch.bool, device=device)
        
        # Since context is unlimited ([-1, -1]), no masking is applied to att_mask
        # att_mask remains all True
        
        # Padding mask
        pad_mask = torch.arange(0, max_audio_length, device=device).expand(
            padding_length.size(0), -1
        ) < padding_length.unsqueeze(-1)

        # Combine padding mask with attention mask
        pad_mask_for_att_mask = pad_mask.unsqueeze(1).repeat([1, max_audio_length, 1])
        pad_mask_for_att_mask = torch.logical_and(
            pad_mask_for_att_mask, pad_mask_for_att_mask.transpose(1, 2)
        )
        
        att_mask = att_mask[:, :max_audio_length, :max_audio_length]
        att_mask = torch.logical_and(pad_mask_for_att_mask, att_mask.to(pad_mask_for_att_mask.device))
        att_mask = ~att_mask  # Invert for masked attention

        pad_mask = ~pad_mask  # Invert for padding
        return pad_mask, att_mask

    def forward(self, audio_signal, length):
        if length is None:
            length = audio_signal.new_full(
                (audio_signal.size(0),), 
                audio_signal.size(-1), 
                dtype=torch.int64
            )

        # Update max sequence length if needed
        self.update_max_seq_length(seq_length=audio_signal.size(1), device=audio_signal.device)
        
        # Transpose to [B, T, feat_in]
        audio_signal = torch.transpose(audio_signal, 1, 2) if audio_signal.dim() == 3 and audio_signal.size(1) == self._feat_in else audio_signal

        # Apply subsampling
        audio_signal, length = self.pre_encode(x=audio_signal, lengths=length)
        length = length.to(torch.int64)

        max_audio_length = audio_signal.size(1)

        # Apply positional encoding
        audio_signal, pos_emb = self.pos_enc(x=audio_signal, cache_len=0)

        # Create masks
        pad_mask, att_mask = self._create_masks(
            att_context_size=self.att_context_size,
            padding_length=length,
            max_audio_length=max_audio_length,
            device=audio_signal.device,
        )

        # Pass through conformer layers
        for layer in self.layers:
            audio_signal = layer(
                x=audio_signal,
                att_mask=att_mask,
                pos_emb=pos_emb,
                pad_mask=pad_mask,
            )

        # Apply output projection if exists
        if self.out_proj is not None:
            audio_signal = self.out_proj(audio_signal)

        # Transpose back to [B, feat_out, T]
        audio_signal = torch.transpose(audio_signal, 1, 2)
        length = length.to(dtype=torch.int64)

        return audio_signal, length

class Featurizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.window = nn.Parameter(torch.zeros(400))
        self.fb = nn.Parameter(torch.zeros(80, 257))

class Preprocessor(nn.Module):
    def __init__(self, n_fft=512, win_length=400, hop_length=160):
        super().__init__()
        self.featurizer = Featurizer()
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length

    def forward(self, input_signal, length):
        # input_signal: [B, T_samples]
        # length: [B]
        spec = torch.stft(
            input_signal,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.featurizer.window.to(input_signal.device),
            center=True,
            pad_mode='reflect',
            onesided=True,
            return_complex=False
        )  # [B, F=257, T_frames, 2]

        amp = spec.pow(2).sum(-1)  # [B, 257, T_frames]

        mel = torch.matmul(self.featurizer.fb.to(amp.device), amp)  # [B, 80, T_frames]

        log_mel = torch.log(mel.clamp(min=1e-10))  # [B, 80, T_frames]

        # Transpose to [B, T_frames, 80]
        log_mel = log_mel.transpose(1, 2)

        # Compute feature lengths
        processed_length = ((length - self.win_length) // self.hop_length) + 1
        processed_length = processed_length.clamp(min=1)

        return log_mel, processed_length

class ConformerCTC(nn.Module):
    def __init__(self):
        super().__init__()
        self.preprocessor = Preprocessor()
        self.encoder = ConformerEncoder()
        self.decoder = ConvASRDecoder()

    def forward(self, input_signal, input_signal_length):
        processed_signal, processed_signal_length = self.preprocessor(input_signal, input_signal_length)
        encoded, encoded_length = self.encoder(processed_signal, processed_signal_length)
        log_probs = self.decoder(encoded)
        return log_probs, encoded_length

# To load the weights:
model = ConformerCTC()
for param in model.parameters():
    print(param.size())
# model.load_state_dict(checkpoint['state_dict'])
# Or if the checkpoint is directly the state_dict:
# model.load_state_dict(checkpoint)

vocabulary = [
    "<unk>", "s", "▁the", "▁a", "t", "▁to", "▁and", "▁i", "▁of", "''''", "ed", "▁in", "d", "ing", "n", "e",
    "▁it", "▁that", "▁you", "y", "er", "r", "▁for", "m", "▁is", "▁he", "re", "▁was", "▁be", "p", "ly", "▁so",
    "▁we", "a", "g", "o", "▁c", "b", "u", "▁on", "▁have", "▁but", "ll", "▁with", "▁re", "or", "▁s", "al",
    "▁do", "▁know", "ar", "▁they", "▁not", "▁as", "▁this", "in", "le", "▁e", "▁are", "▁like", "c", "▁uh",
    "ri", "▁me", "▁his", "▁at", "l", "es", "▁de", "▁yeah", "▁can", "k", "▁or", "▁my", "▁all", "▁had",
    "▁there", "▁will", "▁one", "il", "▁no", "▁what", "en", "ck", "▁b", "▁f", "ce", "ch", "i", "▁by", "▁she",
    "▁from", "▁an", "ic", "ur", "ve", "w", "ter", "la", "▁if", "▁just", "th", "li", "▁", "▁her", "▁um",
    "on", "ation", "▁w", "▁would", "f", "te", "▁st", "▁go", "ir", "it", "▁out", "ro", "▁pa", "▁were", "▁g",
    "▁t", "ion", "▁think", "an", "▁right", "▁about", "se", "lo", "ent", "▁up", "ment", "ate", "▁when", "h",
    "ne", "▁don", "▁has", "▁also", "▁more", "▁see", "▁okay", "▁their", "▁your", "ge", "▁who", "▁well",
    "▁co", "▁which", "▁some", "▁se", "▁time", "▁ba", "▁said", "▁con", "ers", "▁ra", "us", "de", "ra", "▁him",
    "▁our", "▁been", "▁fa", "▁po", "▁pro", "et", "x", "▁la", "id", "ver", "▁oh", "▁ma", "v", "▁now", "age",
    "▁two", "ld", "▁mo", "▁how", "tion", "▁people", "ive", "▁other", "ng", "ity", "z", "ist", "▁very",
    "▁get", "▁any", "▁un", "▁ro", "is", "▁work", "▁mean", "▁them", "▁lo", "vi", "▁because", "ies", "ul",
    "as", "ad", "mp", "▁bo", "-", "▁then", "▁good", "el", "nd", "▁li", "▁man", "▁dis", "▁could", "▁ho",
    "at", "ol", "▁bu", "▁te", "▁ha", "est", "me", "▁say", "ru", "ke", "▁sp", "▁k", "able", "▁su", "▁sa",
    "▁di", "▁fi", "ance", "▁really", "▁over", "▁even", "ry", "▁us", "▁ca", "ow", "ho", "▁into", "ence",
    "mo", "▁mi", "one", "qu", "ut", "lu", "▁o", "ty", "▁after", "▁want", "▁new", "▁take", "▁p", "▁look",
    "▁pre", "sh", "▁day", "▁should", "▁th", "▁need", "▁cha", "co", "▁much", "▁where", "▁d", "ant", "▁fe",
    "▁da", "▁make", "om", "▁did", "▁le", "un", "▁only", "im", "▁these", "ff", "ti", "ish", "▁ex", "ted",
    "▁first", "he", "ig", "▁vi", "▁ri", "▁en", "▁com", "ated", "▁than", "ma", "▁way", "um", "ct", "end",
    "ight", "▁here", "▁ta", "▁car", "▁part", "▁come", "ia", "▁off", "▁sc", "▁ah", "am", "▁tra", "▁yes",
    "▁back", "ture", "ful", "▁pri", "ction", "ine", "▁three", "ard", "▁let", "pe", "▁little", "▁down",
    "mb", "▁si", "▁dr", "▁mr", "▁going", "▁comp", "po", "▁m", "▁sta", "▁gra", "day", "▁many", "ian", "ta",
    "▁long", "▁pi", "▁too", "▁app", "▁kind", "ous", "ci", "▁ga", "ten", "nt", "▁before", "▁may", "▁got",
    "man", "tic", "ition", "cu", "ugh", "tra", "▁n", "ward", "▁give", "▁every", "▁hi", "ting", "▁exp",
    "▁those", "▁hu", "ot", "▁something", "▁lot", "▁still", "▁ne", "na", "ise", "pp", "▁most", "▁gu",
    "▁state", "▁actually", "▁such", "▁bi", "▁never", "tain", "▁great", "▁through", "▁al", "no", "▁mar",
    "▁year", "ach", "les", "▁school", "ally", "ial", "ha", "▁old", "▁made", "ary", "▁ar", "▁years",
    "▁help", "▁per", "ving", "ical", "ther", "▁does", "ac", "ca", "▁must", "di", "▁own", "▁ru", "▁things",
    "▁hand", "▁thing", "▁high", "▁last", "go", "▁sh", "▁under", "▁four", "▁place", "ations", "▁sure",
    "mi", "nce", "▁am", "for", "ness", "▁name", "▁five", "ound", "▁op", "▁cons", "▁ph", "▁same", "row",
    "ven", "ph", "ite", "▁pe", "j", "▁sha", "▁friend", "▁wi", "▁call", "▁european", "▁h", "ect", "ress",
    "▁live", "port", "▁mhm", "▁house", "ie", "ni", "▁plan", "▁jo", "▁play", "side", "▁va", "min", "ious",
    "▁life", "▁du", "▁ti", "▁six", "▁men", "▁again", "▁thank", "▁talk", "par", "▁home", "op", "▁both",
    "▁why", "▁put", "▁another", "nc", "▁being", "mit", "▁came", "led", "▁fo", "▁end", "▁member", "ative",
    "▁thought", "▁tri", "iv", "our", "red", "▁went", "lic", "▁find", "▁pu", "land", "▁start", "▁far",
    "▁eu", "▁imp", "▁always", "▁ju", "▁wa", "▁person", "▁singapore", "ap", "▁show", "▁chi", "▁ten",
    "▁eight", "▁while", "▁point", "▁y", "▁ja", "▁ya", "ling", "ctor", "▁use", "▁acc", "▁world", "▁pay",
    "▁read", "va", "vo", "▁change", "▁u", "▁pl", "▁sw", "▁war", "▁might", "nk", "ments", "and", "▁different",
    "▁dec", "cent", "▁ste", "▁better", "▁fun", "▁month", "ship", "ton", "▁tell", "▁twenty", "▁commission",
    "▁exc", "▁miss", "if", "▁love", "▁money", "▁found", "▁hundred", "gg", "▁add", "▁real", "ities", "▁na",
    "▁pass", "▁didn", "▁v", "▁feel", "▁week", "▁win", "ible", "▁try", "▁upon", "ba", "▁interest", "▁inter",
    "son", "line", "▁ob", "▁boy", "▁big", "▁used", "▁seven", "▁away", "▁family", "less", "▁ki", "ber",
    "▁around", "▁turn", "▁anything", "▁care", "▁young", "▁guess", "▁happen", "▁course", "▁agree",
    "▁support", "▁conf", "ual", "▁number", "▁trans", "ating", "▁mister", "▁hard", "▁watch", "ft",
    "▁next", "▁sea", "▁open", "▁without", "duc", "gra", "ak", "▁cap", "▁cre", "hi", "▁government",
    "▁vo", "▁between", "▁each", "▁ve", "▁though", "▁country", "▁few", "▁once", "'", "▁head", "▁free",
    "▁mu", "▁maybe", "▁act", "▁night", "▁thousand", "▁face", "▁uhhuh", "▁keep", "▁nine", "▁close",
    "▁case", "▁che", "▁against", "▁done", "▁ever", "▁law", "▁believe", "▁public", "▁room", "▁sub",
    "▁order", "▁important", "ient", "▁el", "▁children", "▁second", "▁bri", "▁business", "▁hope",
    "▁move", "fa", "▁however", "▁follow", "▁able", "▁word", "▁yet", "▁fla", "▁stand", "ize", "▁je",
    "▁service", "▁nothing", "▁report", "▁called", "▁grow", "▁continue", "▁issue", "▁since", "▁book",
    "▁lu", "▁qui", "▁develop", "▁gen", "▁certain", "light", "▁cor", "▁small", "▁took", "▁question",
    "▁whole", "▁problem", "▁side", "▁child", "▁full", "▁best", "▁mm", "▁probably", "fi", "▁qua",
    "▁sur", "▁market", "▁left", "▁everything", "▁during", "▁understand", "ook", "wa", "▁cent",
    "▁water", "▁quite", "▁leave", "▁himself", "ip", "▁near", "▁saw", "▁together", "▁large",
    "▁having", "▁already", "▁invest", "▁pretty", "▁direct", "▁hour", "▁fact", "way", "▁run", "▁bra",
    "▁clear", "▁fra", "▁area", "▁union", "▁enough", "▁consider", "▁lead", "▁remain", "▁president",
    "▁system", "▁def", "▁stuff", "▁food", "▁job", "▁heard", "▁err", "▁mind", "▁rest", "▁speak",
    "▁asked", "ator", "▁half", "▁father", "com", "▁less", "▁arm", "▁human", "ency", "▁matter",
    "▁group", "▁girl", "▁current", "▁main", "ttle", "▁later", "▁learn", "▁strong", "▁sign",
    "▁check", "▁light", "▁else", "▁true", "▁term", "qui", "▁minute", "▁spec", "▁return",
    "▁answer", "▁reason", "▁count", "▁shall", "▁communi", "▁travel", "▁wait", "▁provide",
    "▁low", "▁mother", "▁expect", "▁cause", "▁line", "▁general", "lf", "▁getting", "▁parliament",
    "▁bank", "▁company", "▁stop", "cause", "▁power", "▁gi", "▁europe", "▁moment", "▁among",
    "▁walk", "▁allow", "▁idea", "▁office", "▁town", "▁cannot", "▁countries", "▁become",
    "▁appear", "▁present", "▁bring", "▁least", "▁almost", "▁kids", "▁remember", "▁include",
    "▁short", "▁sometimes", "▁game", "▁level", "▁exactly", "▁particular", "▁social", "▁land",
    "▁woman", "▁north", "▁nice", "▁concern", "▁sort", "▁effect", "▁national", "▁several",
    "▁safe", "▁until", "▁further", "▁cost", "▁wonder", "▁whether", "▁either", "▁future",
    "▁pra", "▁council", "▁knew", "▁common", "▁south", "▁making", "▁morning", "▁process",
    "▁situation", "▁white", "▁result", "▁suppose", "▁employ", "▁political", "▁program",
    "▁along", "▁women", "▁ski", "▁court", "▁please", "▁shi", "▁possible", "▁protect",
    "▁experience", "▁definitely", "▁require", "▁account", "▁myself", "▁black", "▁example",
    "▁america", "▁thirty", "▁student", "▁view", "▁product", "▁wife", "▁health", "▁major",
    "▁difficult", "▁death", "▁visit", "▁across", "▁receive", "▁voice", "▁citizen", "▁regard",
    "▁author", "▁treat", "▁especially", "▁local", "▁europeans", "▁met", "▁single", "▁subject",
    "▁union's", "▁department", "▁instead", "▁paper", "▁ago", "▁policy", "▁music", "▁congress",
    "▁economic", "▁plant", "▁election", "▁guy", "▁series", "▁christmas", "▁age", "▁international",
    "▁period", "▁actu", "▁attention", "▁washington", "▁reporter", "▁film", "▁wh", "▁george",
    "▁decision", "▁ge", "▁third", "▁watching", "▁door", "▁art", "▁dr.", "▁yourself", "▁president's",
    "▁happy", "▁story", "▁brown", "▁certainly", "▁nearly", "▁clearly", "▁meet", "▁number",
    "▁teacher", "▁special", "▁industry", "▁james", "▁truth", "▁minutes", "▁towards", "▁beyond",
    "▁green", "▁led", "▁nature", "▁united", "▁history", "▁change", "▁self", "▁rather", "▁society",
    "▁tennessee", "▁weekend", "▁writing", "▁area's", "▁outside", "▁center", "▁perfect", "▁laura",
    "▁fight", "▁teacher's", "▁school's", "▁national's", "▁inside", "▁general's", "▁coming",
    "▁monday", "▁working", "▁sunday", "▁tuesday", "▁friday", "▁saturday", "▁thursday", "▁wednesday",
    "q"
]
