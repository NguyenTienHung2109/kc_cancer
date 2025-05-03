import copy
import torch
import torch.nn as nn

class CaraNet(nn.Module):
    def __init__(self, 
                slice_encoder: nn.Module,
                nodule_decoder: nn.Module,
                use_lung_loc: bool = False):
        super().__init__()
        
        self.encoder = slice_encoder
        self.decoder = nodule_decoder
        self.use_lung_loc = use_lung_loc

    def forward(self, x):
        
        x = torch.cat((x, x, x), dim = 1)
        
        x2_rfb, x3_rfb, x4_rfb, feature_map = self.encoder(x)
        res = {"fm": feature_map}
        
        logits = self.decoder(x2_rfb, x3_rfb, x4_rfb)
        res["seg_nodule"] = [logit[:,:1,...] for logit in logits]
        if self.use_lung_loc:
            res["seg_lung_loc"] = [logit[:,1:,...] for logit in logits]
        
        return res

if __name__ == '__main__':
    import rootutils
    rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
    from src.models.segcls.new_net import Encoder, Decoder

    use_lung_loc = True
    caranet = CaraNet(encoder=Encoder(channel=48),
                    decoder=Decoder(channel=48, out_channel=6),
                    use_lung_loc=use_lung_loc)
    input_tensor = torch.randn(1, 1, 352, 352)

    res = caranet(input_tensor)
    nodule_logits, feature_map = res["seg_nodule"], res["fm"]

    print("Feature map:", feature_map.shape)
    print("Nodule")
    for logit in nodule_logits:
        print(logit.shape)

    if use_lung_loc:
        lung_loc_logits = res["seg_lung_loc"]
        
        print("Lung Location")
        for logit in lung_loc_logits:
            print(logit.shape)
