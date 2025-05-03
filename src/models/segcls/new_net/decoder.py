import torch
import torch.nn as nn
import torch.nn.functional as F

import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.models.components.lib.conv_layer import Conv
from src.models.components.lib.context_module import CFPModule
from src.models.components.lib.axial_atten import AA_kernel
from src.models.components.lib.partial_decoder import aggregation

class Decoder(nn.Module):
    def __init__(self, channel: int,  # divisible 2 and out_channel
                out_channel: int=1):
        # 
        super().__init__()

        # Partial Decoder
        self.agg1 = aggregation(channel, out_channel)
        
        self.CFP_1 = CFPModule(channel, d = 8)
        self.CFP_2 = CFPModule(channel, d = 8)
        self.CFP_3 = CFPModule(channel, d = 8)
        ###### dilation rate 4, 62.8

        self.ra1_conv1 = Conv(channel,channel,3,1,padding=1,bn_acti=True)
        self.ra1_conv2 = Conv(channel,channel,3,1,padding=1,bn_acti=True)
        self.ra1_conv3 = Conv(channel,out_channel,3,1,padding=1,bn_acti=True)
        
        self.ra2_conv1 = Conv(channel,channel,3,1,padding=1,bn_acti=True)
        self.ra2_conv2 = Conv(channel,channel,3,1,padding=1,bn_acti=True)
        self.ra2_conv3 = Conv(channel,out_channel,3,1,padding=1,bn_acti=True)
        
        self.ra3_conv1 = Conv(channel,channel,3,1,padding=1,bn_acti=True)
        self.ra3_conv2 = Conv(channel,channel,3,1,padding=1,bn_acti=True)
        self.ra3_conv3 = Conv(channel,out_channel,3,1,padding=1,bn_acti=True)
        
        self.aa_kernel_1 = AA_kernel(channel,channel)
        self.aa_kernel_2 = AA_kernel(channel,channel)
        self.aa_kernel_3 = AA_kernel(channel,channel)

    def forward(self, x2_rfb, x3_rfb, x4_rfb, return_decoder: bool = False, cond = None):
        if cond is not None:
            c1, c2, c3, c4 = cond

        decoder_1 = self.agg1(x4_rfb, x3_rfb, x2_rfb) # bs, out_channel, 44, 44
        if cond is not None:
            decoder_1 *= c1
        lateral_map_1 = F.interpolate(decoder_1, scale_factor=8, mode='bilinear') # bs, out_channel, 352, 352
        
        # ------------------- atten-one -----------------------
        decoder_2 = F.interpolate(decoder_1, scale_factor=0.25, mode='bilinear')
        if cond is not None:
            decoder_2 *= c2
        cfp_out_1 = self.CFP_3(x4_rfb) # bs, channel, 11, 11
        decoder_2_ra = -1*(torch.sigmoid(decoder_2)) + 1
        aa_atten_3 = self.aa_kernel_3(cfp_out_1)
        # aa_atten_3_o = decoder_2_ra.expand(-1, aa_atten_3.shape[1], -1, -1).mul(aa_atten_3)
        aa_atten_3_o = decoder_2_ra.repeat(1, aa_atten_3.shape[1] // decoder_2_ra.shape[1], 1, 1).mul(aa_atten_3)

        ra_3 = self.ra3_conv1(aa_atten_3_o) # channel - channel
        ra_3 = self.ra3_conv2(ra_3) # channel - channel
        ra_3 = self.ra3_conv3(ra_3) # channel - out_channel
        
        x_3 = ra_3 + decoder_2
        lateral_map_2 = F.interpolate(x_3,scale_factor=32,mode='bilinear') # bs, 1, 352, 352

        # ------------------- atten-two -----------------------      
        decoder_3 = F.interpolate(x_3, scale_factor=2, mode='bilinear')
        if cond is not None:
            decoder_3 *= c3
        cfp_out_2 = self.CFP_2(x3_rfb) # channel - channel
        decoder_3_ra = -1*(torch.sigmoid(decoder_3)) + 1
        aa_atten_2 = self.aa_kernel_2(cfp_out_2)
        # aa_atten_2_o = decoder_3_ra.expand(-1, aa_atten_2.shape[1], -1, -1).mul(aa_atten_2)
        aa_atten_2_o = decoder_3_ra.repeat(1, aa_atten_2.shape[1] // decoder_3_ra.shape[1], 1, 1).mul(aa_atten_2)

        ra_2 = self.ra2_conv1(aa_atten_2_o) # channel - channel
        ra_2 = self.ra2_conv2(ra_2) # channel - channel
        ra_2 = self.ra2_conv3(ra_2) # channel - out_channel
        
        x_2 = ra_2 + decoder_3
        lateral_map_3 = F.interpolate(x_2,scale_factor=16,mode='bilinear')  # bs, 1, 352, 352     
        
        # ------------------- atten-three -----------------------
        decoder_4 = F.interpolate(x_2, scale_factor=2, mode='bilinear')
        if cond is not None:
            decoder_4 *= c4
        cfp_out_3 = self.CFP_1(x2_rfb) # channel - channel
        decoder_4_ra = -1*(torch.sigmoid(decoder_4)) + 1
        aa_atten_1 = self.aa_kernel_1(cfp_out_3)
        # aa_atten_1_o = decoder_4_ra.expand(-1, aa_atten_1.shape[1], -1, -1).mul(aa_atten_1)
        aa_atten_1_o = decoder_4_ra.repeat(1, aa_atten_1.shape[1] // decoder_4_ra.shape[1], 1, 1).mul(aa_atten_1)

        ra_1 = self.ra1_conv1(aa_atten_1_o) # channel - channel
        ra_1 = self.ra1_conv2(ra_1) # channel - channel
        ra_1 = self.ra1_conv3(ra_1) # channel - out_channel
        
        x_1 = ra_1 + decoder_4
        lateral_map_5 = F.interpolate(x_1,scale_factor=8,mode='bilinear') 
        
        return lateral_map_5, lateral_map_3, lateral_map_2, lateral_map_1
