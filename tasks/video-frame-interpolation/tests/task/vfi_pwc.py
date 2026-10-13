"""PWC-Net inference compatible with the FloLPIPS reference checkpoint.

Architecture/state names follow danier97/flolpips (pwcnet.py, adapted from
sniklaus/pytorch-pwc). See METRICS.md and LICENSE-metrics.txt. Correlation uses
ordinary PyTorch so the trusted CPU verifier needs no CuPy/CUDA.
No downloads, candidate imports, global CUDA state, or fallback flow estimator.
"""
import torch
from torch import nn
from torch.nn import functional as F

NAMES = ("One", "Two", "Thr", "Fou", "Fiv", "Six")
CHANNELS = (3, 16, 32, 64, 96, 128, 196)
VOLUME_CHANNELS = {2: 117, 3: 149, 4: 181, 5: 213, 6: 81}


def correlation(first, second):
    """81 channel-mean dot products, ordered dy then dx, each in [-4,4]."""
    height, width = first.shape[-2:]
    padded = F.pad(second, (4, 4, 4, 4))
    return torch.cat([(first * padded[:, :, y:y+height, x:x+width]).mean(1, keepdim=True)
                      for y in range(9) for x in range(9)], dim=1)


def backwarp(features, flow):
    height, width = flow.shape[-2:]
    y = torch.linspace(-1 + 1 / height, 1 - 1 / height, height, device=flow.device, dtype=flow.dtype)
    x = torch.linspace(-1 + 1 / width, 1 - 1 / width, width, device=flow.device, dtype=flow.dtype)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    grid = torch.stack((xx, yy), dim=0).unsqueeze(0)
    # Preserve reference scaling, including its align_corners=False convention.
    displacement = flow / flow.new_tensor([(width - 1) / 2, (height - 1) / 2])[None, :, None, None]
    padded = torch.cat((features, torch.ones_like(features[:, :1])), dim=1)
    sampled = F.grid_sample(padded, (grid + displacement).permute(0, 2, 3, 1),
                            mode="bilinear", padding_mode="zeros", align_corners=False)
    return sampled[:, :-1] * (sampled[:, -1:] > 0.999).to(features.dtype)


def convolution(cin, cout, *, stride=1, dilation=1):
    return [nn.Conv2d(cin, cout, 3, stride=stride, padding=dilation, dilation=dilation),
            nn.LeakyReLU(0.1, inplace=False)]


class Extractor(nn.Module):
    def __init__(self):
        super().__init__()
        for name, cin, cout in zip(NAMES, CHANNELS, CHANNELS[1:]):
            setattr(self, "net" + name, nn.Sequential(*convolution(cin, cout, stride=2),
                                                      *convolution(cout, cout), *convolution(cout, cout)))

    def forward(self, image):
        pyramid = []
        for name in NAMES:
            image = getattr(self, "net" + name)(image)
            pyramid.append(image)
        return pyramid


class Decoder(nn.Module):
    def __init__(self, level):
        super().__init__()
        self.level = level
        cin = VOLUME_CHANNELS[level]
        if level < 6:
            self.netUpflow = nn.ConvTranspose2d(2, 2, 4, stride=2, padding=1)
            self.netUpfeat = nn.ConvTranspose2d(VOLUME_CHANNELS[level + 1] + 448, 2, 4, stride=2, padding=1)
        for name, cout in zip(NAMES, (128, 128, 96, 64, 32)):
            setattr(self, "net" + name, nn.Sequential(*convolution(cin, cout)))
            cin += cout
        self.netSix = nn.Sequential(nn.Conv2d(cin, 2, 3, padding=1))

    def forward(self, first, second, previous=None):
        if previous is None:
            features = F.leaky_relu(correlation(first, second), 0.1)
        else:
            flow = self.netUpflow(previous[0])
            upfeatures = self.netUpfeat(previous[1])
            warped = backwarp(second, flow * (5 / 2 ** (self.level - 2)))
            volume = F.leaky_relu(correlation(first, warped), 0.1)
            features = torch.cat((volume, first, flow, upfeatures), dim=1)
        for name in NAMES[:5]:
            features = torch.cat((getattr(self, "net" + name)(features), features), dim=1)
        return self.netSix(features), features


class Refiner(nn.Module):
    def __init__(self):
        super().__init__()
        layers, cin = [], 565
        for cout, dilation in zip((128, 128, 128, 96, 64, 32), (1, 2, 4, 8, 16, 1)):
            layers.extend(convolution(cin, cout, dilation=dilation))
            cin = cout
        self.netMain = nn.Sequential(*layers, nn.Conv2d(cin, 2, 3, padding=1))

    def forward(self, features):
        return self.netMain(features)


class PWCNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.netExtractor = Extractor()
        for level in range(2, 7):
            setattr(self, "net" + NAMES[level - 1], Decoder(level))
        self.netRefiner = Refiner()

    def forward(self, first, second):
        height, width = first.shape[-2:]
        size = ((height + 63) // 64 * 64, (width + 63) // 64 * 64)
        pyramids = [self.netExtractor(F.interpolate(image, size=size, mode="bilinear", align_corners=False))
                    for image in (first, second)]
        previous = None
        for level in range(6, 1, -1):
            previous = getattr(self, "net" + NAMES[level - 1])(
                pyramids[0][level - 1], pyramids[1][level - 1], previous)
        flow = previous[0] + self.netRefiner(previous[1])
        flow = 20 * F.interpolate(flow, size=(height, width), mode="bilinear", align_corners=False)
        return flow * flow.new_tensor([width / size[1], height / size[0]])[None, :, None, None]
