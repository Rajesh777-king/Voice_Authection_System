
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ==========================================================
# BASIC BLOCKS
# ==========================================================

class Conv1dReluBn(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=1, stride=1,
                 padding=0, dilation=1, bias=False):

        super().__init__()

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            bias=bias
        )

        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        return self.bn(F.relu(self.conv(x)))


class Res2Conv1dReluBn(nn.Module):

    def __init__(self, channels,
                 kernel_size=1,
                 stride=1,
                 padding=0,
                 dilation=1,
                 bias=False,
                 scale=4):

        super().__init__()

        assert channels % scale == 0

        self.scale = scale
        self.width = channels // scale
        self.nums = scale - 1

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        for i in range(self.nums):

            self.convs.append(
                nn.Conv1d(
                    self.width,
                    self.width,
                    kernel_size,
                    stride,
                    padding,
                    dilation,
                    bias=bias
                )
            )

            self.bns.append(nn.BatchNorm1d(self.width))

    def forward(self, x):

        spx = torch.split(x, self.width, 1)
        out = []

        for i in range(self.nums):

            if i == 0:
                sp = spx[i]
            else:
                sp = sp + spx[i]

            sp = self.convs[i](sp)
            sp = self.bns[i](F.relu(sp))

            out.append(sp)

        out.append(spx[self.nums])

        return torch.cat(out, dim=1)


class SE_Connect(nn.Module):

    def __init__(self, channels, s=2):
        super().__init__()

        self.linear1 = nn.Linear(channels, channels // s)
        self.linear2 = nn.Linear(channels // s, channels)

    def forward(self, x):

        out = x.mean(dim=2)

        out = F.relu(self.linear1(out))
        out = torch.sigmoid(self.linear2(out))

        return x * out.unsqueeze(2)


def SE_Res2Block(channels,
                 kernel_size,
                 stride,
                 padding,
                 dilation,
                 scale):

    return nn.Sequential(

        Conv1dReluBn(
            channels,
            channels,
            kernel_size=1
        ),

        Res2Conv1dReluBn(
            channels,
            kernel_size,
            stride,
            padding,
            dilation,
            scale=scale
        ),

        Conv1dReluBn(
            channels,
            channels,
            kernel_size=1
        ),

        SE_Connect(channels)
    )


class AttentiveStatsPool(nn.Module):

    def __init__(self, in_dim, bottleneck_dim):

        super().__init__()

        self.linear1 = nn.Conv1d(
            in_dim,
            bottleneck_dim,
            kernel_size=1
        )

        self.linear2 = nn.Conv1d(
            bottleneck_dim,
            in_dim,
            kernel_size=1
        )

    def forward(self, x):

        alpha = torch.tanh(self.linear1(x))
        alpha = torch.softmax(self.linear2(alpha), dim=2)

        mean = torch.sum(alpha * x, dim=2)

        residuals = (
            torch.sum(alpha * x ** 2, dim=2)
            - mean ** 2
        )

        std = torch.sqrt(
            residuals.clamp(min=1e-9)
        )

        return torch.cat([mean, std], dim=1)


# ==========================================================
# ECAPA-TDNN MODEL
# ==========================================================

class ECAPA_TDNN(nn.Module):

    def __init__(self,
                 in_channels=80,
                 channels=512,
                 embd_dim=192):

        super().__init__()

        self.layer1 = Conv1dReluBn(
            in_channels,
            channels,
            kernel_size=5,
            padding=2
        )

        self.layer2 = SE_Res2Block(
            channels,
            kernel_size=3,
            stride=1,
            padding=2,
            dilation=2,
            scale=8
        )

        self.layer3 = SE_Res2Block(
            channels,
            kernel_size=3,
            stride=1,
            padding=3,
            dilation=3,
            scale=8
        )

        self.layer4 = SE_Res2Block(
            channels,
            kernel_size=3,
            stride=1,
            padding=4,
            dilation=4,
            scale=8
        )

        self.conv = nn.Conv1d(
            channels * 3,
            1536,
            kernel_size=1
        )

        self.pooling = AttentiveStatsPool(
            1536,
            128
        )

        self.bn1 = nn.BatchNorm1d(3072)

        self.linear = nn.Linear(
            3072,
            embd_dim
        )

        self.bn2 = nn.BatchNorm1d(embd_dim)

    def forward(self, x):

        x = x.transpose(1, 2)

        out1 = self.layer1(x)

        out2 = self.layer2(out1) + out1

        out3 = (
            self.layer3(out1 + out2)
            + out1
            + out2
        )

        out4 = (
            self.layer4(out1 + out2 + out3)
            + out1
            + out2
            + out3
        )

        out = torch.cat(
            [out2, out3, out4],
            dim=1
        )

        out = F.relu(self.conv(out))

        out = self.pooling(out)

        out = self.bn1(out)

        out = self.linear(out)

        out = self.bn2(out)

        return out


# ==========================================================
# LOAD CHECKPOINT
# ==========================================================

DEVICE = torch.device("cpu")

model = ECAPA_TDNN().to(DEVICE)


checkpoint = torch.load(
    "epoch_7.pt",
    map_location=DEVICE,
    weights_only=False
)



model.load_state_dict(
    checkpoint['model_state_dict']
)

model.eval()

print("Checkpoint loaded successfully!")


# ==========================================================
# DUMMY TEST INPUT
# ==========================================================

dummy_feature = np.random.rand(
    200,
    80
).astype(np.float32)

dummy_feature = torch.FloatTensor(
    dummy_feature
).unsqueeze(0).to(DEVICE)


# ==========================================================
# INFERENCE
# ==========================================================

with torch.no_grad():

    embedding = model(dummy_feature)

print("Embedding shape:", embedding.shape)

print("Inference successful!")

