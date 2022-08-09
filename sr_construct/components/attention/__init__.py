"""
base attention modules commonly used in SR archs
including channel-wise and spatial(pixel)-wise attention
"""

from .ch_attn import SE_Module, CCA_Module
from .pix_attn import PA_Module, PAConv_Module, SCPA_Module, ESA_Module