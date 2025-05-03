import torch.nn as nn

import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.models.components.pretrain.Res2Net_v1b import res2net101_v1b_26w_4s
from src.models.components.lib.conv_layer import Conv

class Encoder(nn.Module):
    def __init__(self, channel: int,  # divisible 2 and out_channel
                backbone = res2net101_v1b_26w_4s(pretrained=True)):
        super().__init__()

        # ---- ResNet Backbone ----
        self.resnet = backbone

        # Receptive Field Block
        self.rfb2_1 = Conv(512, channel,3,1,padding=1,bn_acti=True)
        self.rfb3_1 = Conv(1024, channel,3,1,padding=1,bn_acti=True)
        self.rfb4_1 = Conv(2048, channel,3,1,padding=1,bn_acti=True)

    def forward(self, x):
        """_summary_

        Args:
            x (_type_): required RGB images (b, 3, w, h)
        """

        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)      # bs, 64, 88, 88

        # ----------- low-level features -------------
        x1 = self.resnet.layer1(x)      # bs, 256, 88, 88
        x2 = self.resnet.layer2(x1)     # bs, 512, 44, 44
        
        x3 = self.resnet.layer3(x2)     # bs, 1024, 22, 22
        x4 = self.resnet.layer4(x3)     # bs, 2048, 11, 11

        x2_rfb = self.rfb2_1(x2) # bs, 32, 44, 44
        x3_rfb = self.rfb3_1(x3) # bs, 32, 22, 22
        x4_rfb = self.rfb4_1(x4) # bs, 32, 11, 11

        return x2_rfb, x3_rfb, x4_rfb, x3
