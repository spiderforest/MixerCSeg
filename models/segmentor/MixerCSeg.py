import torch
from torch import nn
from models.decoder import SRFModule
from models.encoder import VSSEncoder



class MixerCSeg(nn.Module):
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



def build_MixerCSeg(args):
    device = torch.device(args.device)
    args.device = torch.device(args.device)

    embed_dim=[16,32,64,128]

    depths = [1,1,1,1]
    state_dim=[8,8,16,16]

    backbone = VSSEncoder(
        in_dim=3,
        embed_dim=embed_dim,
        depths=depths,
        mlp_ratio=2.,
        state_dim=state_dim,
        )
    model = MixerCSeg(backbone, embed_dim, args).to(device)

    criterion = bce_dice(args)
    criterion.to(device)
    
    return model, criterion

