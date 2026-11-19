"""The same light U-Net as the wildfire detector (notebooks/train_unet.ipynb), with a 3-class head."""
import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    def __init__(self, in_channels=2, classes=3, base=16, depth=4, dropout=0.1):
        super().__init__()
        self.config = {"name": "UNet", "in_channels": in_channels, "classes": classes, "base": base, "depth": depth, "dropout": dropout}
        widths = [base * 2**level for level in range(depth)]
        self.down = nn.ModuleList()
        channels = in_channels
        for width in widths:
            self.down.append(conv_block(channels, width))
            channels = width
        self.bottom = conv_block(widths[-1], widths[-1] * 2)
        self.dropout = nn.Dropout2d(dropout)
        self.up = nn.ModuleList(nn.ConvTranspose2d(width * 2, width, 2, stride=2) for width in reversed(widths))
        self.merge = nn.ModuleList(conv_block(width * 2, width) for width in reversed(widths))
        self.head = nn.Conv2d(base, classes, 1)
        self.multiple = 2**depth  # input height and width must divide by this

    def forward(self, x):
        skips = []
        for block in self.down:
            x = block(x)
            skips.append(x)
            x = F.max_pool2d(x, 2)
        x = self.dropout(self.bottom(x))
        for up, merge, skip in zip(self.up, self.merge, reversed(skips)):
            x = merge(torch.cat([up(x), skip], dim=1))
        return self.head(x)


def build(config):
    cfg = {k: v for k, v in config.items() if k != "name"}
    return UNet(**cfg)


@torch.no_grad()
def predict_classes(model, z, device, window=1024):
    """Argmax classes [H, W] for a scaled image z [C, H, W], in padded windows so any size fits."""
    model.eval()
    c, h, w = z.shape
    m = model.multiple
    out = torch.zeros((h, w), dtype=torch.uint8)
    for r in range(0, h, window):
        for col in range(0, w, window):
            tile = torch.from_numpy(z[:, r:r + window, col:col + window].astype("float32"))
            th, tw = tile.shape[1:]
            ph, pw = (-th) % m, (-tw) % m
            tile = F.pad(tile, (0, pw, 0, ph))
            logits = model(tile[None].to(device))[0, :, :th, :tw]
            out[r:r + th, col:col + tw] = logits.argmax(0).to(torch.uint8).cpu()
    return out.numpy()
