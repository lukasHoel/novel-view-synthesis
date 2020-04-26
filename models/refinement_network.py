from residual_block import *

class RefineNet(nn.Module):
    '''
    Refinement network (RN) that consists of ResidualBlocks of type "id", "avg" and "ups".
    Based on ResNetDecoder in architectures.py
    See Appendix B and fig. 15(b) in SynSin paper.
    '''
    def __init__(self, res_block_dims=[], res_block_types=[], activate_out=nn.Sigmoid):
        '''
        Let n-many ResNet blocks, res_block_dims include n+1 elements 
        to specify input and output channels for each block.
        '''
        super().__init__()

        self.res_blocks = []
        for i in range(len(res_block_dims)-1):
            self.res_blocks.append( 
                ResidualBlock(
                    in_ch=res_block_dims[i],
                    out_ch=res_block_dims[i+1],
                    block_type=res_block_types[i]
                )
            )
        self.res_blocks = nn.Sequential(*self.res_blocks)
        # Final activation can be:
        # - nn.Sigmoid to force output to be in range [0,1]
        # - nn.Tanh to force output to be in range [-1,1]
        self.activate_out = activate_out

    def forward(self, x):
        x = self.res_blocks(x)

        if self.activate_out:
            x = self.activate_out(x)

        return x