import torch
import torch.nn as nn
import torch.nn.utils.spectral_norm

class Identity(nn.Module):
    '''A workaround for identity layer as PyTorch doesn't have it.'''
    def __init__(self, *args, **kwargs):
        super().__init__()
    def forward(self, x):
        return x

block_types = {
    "id": Identity(),
    "ups": nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
    "avg": nn.Upsample(scale_factor=2, mode="bilinear")
}

############################
# BigGAN functions/classes #
# Taken from SynSin code   #
############################
# Spectral norm for linear layer (Suggested in BigGAN, used in SynSin)
def get_linear_layer(in_ch, out_ch, spectral_norm=True):
    linear = nn.Linear(in_ch, out_ch, bias=False)
    if spectral_norm:
        nn.utils.spectral_norm(linear)
    return linear

# Spectral norm for Conv2D layer (Suggested in BigGAN, used in SynSin)
def get_conv2D_layer(in_ch, out_ch, kernel_size, padding, stride, spectral_norm=True):
    conv2D = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding, stride=stride)
    if spectral_norm:
        conv2D = nn.utils.spectral_norm(conv2D)
    return conv2D

# Methods needed for class conditional BatchNorm
def fused_bn(x, mean, var, gain=None, bias=None, eps=1e-5):
    """
    Function used in manual_bn function.
    Based on the idea from BigGAN.
    """
    # Apply scale and shift--if gain and bias are provided, fuse them here
    # Prepare scale
    scale = torch.rsqrt(var + eps)
    # If a gain is provided, use it
    if gain is not None:
        scale = scale * gain
    # Prepare shift
    shift = mean * scale
    # If bias is provided, use it
    if bias is not None:
        shift = shift - bias
    return x * scale - shift

# Manual BN
# Calculate means and variances using mean-of-squares minus mean-squared
def manual_bn(x, gain=None, bias=None, return_mean_var=False, eps=1e-5):
    """
    Function used in bn class.
    Based on the idea from BigGAN.
    """
    # Cast x to float32 if necessary
    float_x = x.float()
    # Calculate expected value of x (m) and expected value of x**2 (m2)
    # Mean of x
    m = torch.mean(float_x, [0, 2, 3], keepdim=True)
    # Mean of x squared
    m2 = torch.mean(float_x ** 2, [0, 2, 3], keepdim=True)
    # Calculate variance as mean of squared minus mean squared.
    var = m2 - m ** 2
    # Cast back to float 16 if necessary
    var = var.type(x.type())
    m = m.type(x.type())
    # Return mean and variance for updating stored mean/var if requested
    if return_mean_var:
        return fused_bn(x, m, var, gain, bias, eps), m.squeeze(), var.squeeze()
    else:
        return fused_bn(x, m, var, gain, bias, eps)

class bn(nn.Module):
    """
    Class used in LinearNoiseLayer.
    Based on the idea from BigGAN.
    """
    def __init__(self, num_channels, eps=1e-5, momentum=0.1):
        super().__init__()

        # momentum for updating stats
        self.momentum = momentum
        self.eps = eps

        self.register_buffer("stored_mean", torch.zeros(num_channels))
        self.register_buffer("stored_var", torch.ones(num_channels))
        self.register_buffer("accumulation_counter", torch.zeros(1))
        # Accumulate running means and vars
        self.accumulate_standing = False

    # reset standing stats
    def reset_stats(self):
        self.stored_mean[:] = 0
        self.stored_var[:] = 0
        self.accumulation_counter[:] = 0

    def forward(self, x, gain, bias):
        if self.training:
            out, mean, var = manual_bn(
                x, gain, bias, return_mean_var=True, eps=self.eps
            )
            # If accumulating standing stats, increment them
            if self.accumulate_standing:
                self.stored_mean[:] = self.stored_mean + mean.data
                self.stored_var[:] = self.stored_var + var.data
                self.accumulation_counter += 1.0
            # If not accumulating standing stats, take running averages
            else:
                self.stored_mean[:] = (
                    self.stored_mean * (1 - self.momentum)
                    + mean * self.momentum
                )
                self.stored_var[:] = (
                    self.stored_var * (1 - self.momentum) + var * self.momentum
                )
            return out
        # If not in training mode, use the stored statistics
        else:
            mean = self.stored_mean.view(1, -1, 1, 1)
            var = self.stored_var.view(1, -1, 1, 1)
            # If using standing stats, divide them by the accumulation counter
            if self.accumulate_standing:
                mean = mean / self.accumulation_counter
                var = var / self.accumulation_counter
            return fused_bn(x, mean, var, gain, bias, self.eps)

# Class that actually implements class conditional BatchNorm. From BigGAN paper:
# The conditioning of each block is linearly projected to produce per-sample gains and biases 
# for the BatchNorm layers of the block. The bias projections are zero-centered, while the gain 
# projections are centered at 1.
class LinearNoiseLayer(nn.Module):
    def __init__(self, noise_sz=20, output_sz=32):
        """
        Class for adding in noise to the batch normalisation layer.
        Based on the idea from BigGAN.
        """
        super().__init__()
        self.noise_sz = noise_sz

        self.gain = get_linear_layer(noise_sz, output_sz)
        self.bias = get_linear_layer(noise_sz, output_sz)

        self.bn = bn(output_sz)

        self.noise_sz = noise_sz

    def forward(self, x):
        noise = torch.randn(x.size(0), self.noise_sz).to(x.device)

        # Predict biases/gains for this layer from the noise
        gain = (1 + self.gain(noise)).view(noise.size(0), -1, 1, 1)
        bias = self.bias(noise).view(noise.size(0), -1, 1, 1)

        xp = self.bn(x, gain=gain, bias=bias)
        return xp

############################
# SynSin functions/classes #
############################
class ResidualBlock(nn.Module):
    '''
    Single residual block with following variations:
    - ID: Residual block with an identity layer.
    - Ups: Residual block with an upsampling layer.
    - Avg: Residual block with an average pool layer.
    Based on ResNet_Block in blocks.py
    See Appendix B and fig. 14 in SynSin paper.
    '''
    def __init__(self, in_ch, out_ch, block_type):
        super().__init__()

        # variable_layer is the layer defining the type of the residual block.
        # It can be: Identity, nn.AvgPool2d, nn.Upsample.
        if block_type not in block_types.keys():
            raise "ResidualBlock: Wrong block type!"

        self.variable_layer = block_types[block_type]

        self.left_branch = nn.Sequential(
            get_conv2D_layer(in_ch, out_ch, kernel_size=(1,1), padding=0, stride=1),
            self.variable_layer
        )

        self.right_branch = nn.Sequential(
            LinearNoiseLayer(output_sz=in_ch),
            nn.ReLU(),
            get_conv2D_layer(in_ch, out_ch, kernel_size=(3,3), padding=1, stride=1),
            LinearNoiseLayer(output_sz=out_ch),
            nn.ReLU(),
            get_conv2D_layer(out_ch, out_ch, kernel_size=(3,3), padding=1, stride=1),
            self.variable_layer
        )

    def forward(self, x):
        return self.left_branch(x) + self.right_branch(x)

class FeatureNet(nn.Module):
    '''
    Spatial feature network that consists of ResidualBlocks of type "id" only.
    Based on ResNetEncoder in architectures.py
    See Appendix B and fig. 15(a) in SynSin paper.
    '''
    def __init__(self, res_block_dims=[]):
        '''
        res_block_dims include input and output dimensions for each ResNet block, 
        includes # of layers + 1 elements
        '''
        super().__init__()

        self.res_blocks = [] 
        for i in range(len(res_block_dims)-1):
            self.res_blocks.append( 
                ResidualBlock(
                    in_ch=res_block_dims[i],
                    out_ch=res_block_dims[i+1],
                    block_type="id"
                )
            )
        self.res_blocks = nn.Sequential(*self.res_blocks)

    def forward(self, x):
        return self.res_blocks(x)