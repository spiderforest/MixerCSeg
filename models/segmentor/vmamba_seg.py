import torch
from torch import nn
from fvcore.nn import FlopCountAnalysis, parameter_count
from models.decoder import SRFModule
from models.encoder import VSSEncoder
from ptflops import get_model_complexity_info


class Mymodel(nn.Module):
    def __init__(self, backbone, embed_dims, args=None):
        super().__init__()
        self.args = args
        self.backbone = backbone    
        self.decoder = SRFModule(embed_dims, mid_dim=8, size=(args.load_width, args.load_height))

    def forward(self, samples):
        outs = self.backbone(samples)
        out = self.decoder(outs)

        return out

class DiceLoss(nn.Module):
    def __init__(self, smooth=1., dims=(-2, -1)):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.dims = dims

    def forward(self, x, y):
        tp = (x * y).sum(self.dims)
        fp = (x * (1 - y)).sum(self.dims)
        fn = ((1 - x) * y).sum(self.dims)
        dc = (2 * tp + self.smooth) / (2 * tp + fp + fn + self.smooth)
        dc = dc.mean()

        return 1 - dc

class bce_dice(nn.Module):
    def __init__(self, args):
        super(bce_dice, self).__init__()
        self.bce_fn = nn.BCEWithLogitsLoss()
        self.dice_fn = DiceLoss()
        self.args = args

    def forward(self, y_pred, y_true):
        bce = self.bce_fn(y_pred, y_true)
        dice = self.dice_fn(y_pred.sigmoid(), y_true)
        return self.args.BCELoss_ratio * bce + self.args.DiceLoss_ratio * dice



def VMamba_seghead(args):
    device = torch.device(args.device)
    args.device = torch.device(args.device)

    embed_dim=[16,32,64,128]
    # embed_dim=  [16,32,64,128]   [32,64,128,256]    [64,128,256,512]
    
    
    depths = [1,1,1,1]
    state_dim=[8,8,16,16]

    backbone = VSSEncoder(
        in_dim=3,
        embed_dim=embed_dim,
        depths=depths,
        mlp_ratio=2.,
        state_dim=state_dim,
        )
    model = Mymodel(backbone, embed_dim, args).to(device)

    ## 计算 FLOPs
    # input_tensor = torch.rand([1,3,512,512]).to(device)
    # flop_analyzer = FlopCountAnalysis(model, input_tensor)
    # flops = flop_analyzer.total()
    # # 计算 Params
    # params = parameter_count(model)[""]
    # print("===============================")
    # print(f"FLOPs: {flops / 1e9:.2f} G")
    # print(f"Params: {params / 1e6:.2f} M")



    # flops, params = get_model_complexity_info(model, (3,512,512), as_strings=True,print_per_layer_stat=True)
    # print("%s |%s" % (flops,params))



    criterion = bce_dice(args)
    criterion.to(device)
    
    return model, criterion

