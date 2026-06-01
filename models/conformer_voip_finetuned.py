import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================================================
# 1. BASE ECAPA BLOCKS
# =========================================================

class Res2Conv1dReluBn(nn.Module):

    def __init__(
        self,
        channels,
        kernel_size=1,
        stride=1,
        padding=0,
        dilation=1,
        bias=False,
        scale=4
    ):

        super().__init__()

        assert channels % scale == 0

        self.scale = scale

        self.width = channels // scale

        self.nums = scale if scale == 1 else scale - 1

        self.convs = nn.ModuleList([

            nn.Conv1d(
                self.width,
                self.width,
                kernel_size,
                stride,
                padding,
                dilation,
                bias=bias
            )

            for _ in range(self.nums)

        ])

        self.bns = nn.ModuleList([

            nn.BatchNorm1d(self.width)

            for _ in range(self.nums)

        ])

    def forward(self, x):

        out = []

        sp = None

        spx = torch.split(x, self.width, 1)

        for i in range(self.nums):

            sp = spx[i] if i == 0 else sp + spx[i]

            sp = self.bns[i](

                F.relu(
                    self.convs[i](sp)
                )
            )

            out.append(sp)

        if self.scale != 1:

            out.append(spx[self.nums])

        return torch.cat(out, dim=1)


class Conv1dReluBn(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=0,
        dilation=1,
        bias=False
    ):

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

        return self.bn(
            F.relu(
                self.conv(x)
            )
        )


class SE_Connect(nn.Module):

    def __init__(self, channels, s=2):

        super().__init__()

        assert channels % s == 0

        self.linear1 = nn.Linear(
            channels,
            channels // s
        )

        self.linear2 = nn.Linear(
            channels // s,
            channels
        )

    def forward(self, x):

        out = x.mean(dim=2)

        out = torch.sigmoid(

            self.linear2(
                F.relu(
                    self.linear1(out)
                )
            )
        )

        return x * out.unsqueeze(2)


def SE_Res2Block(
    channels,
    kernel_size,
    stride,
    padding,
    dilation,
    scale
):

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


# =========================================================
# 2. CONFORMER BLOCKS
# =========================================================

class ConformerFeedForward(nn.Module):

    def __init__(
        self,
        dim,
        expansion=4,
        dropout=0.1
    ):

        super().__init__()

        self.norm = nn.LayerNorm(dim)

        self.fc1 = nn.Linear(
            dim,
            dim * expansion
        )

        self.fc2 = nn.Linear(
            dim * expansion,
            dim
        )

        self.drop = nn.Dropout(dropout)

    def forward(self, x):

        r = x

        x = self.norm(x)

        x = self.drop(

            F.silu(
                self.fc1(x)
            )
        )

        x = self.drop(
            self.fc2(x)
        )

        return r + 0.5 * x


class ConformerConvModule(nn.Module):

    def __init__(
        self,
        dim,
        kernel_size=31,
        dropout=0.1
    ):

        super().__init__()

        self.norm = nn.LayerNorm(dim)

        self.pw_conv1 = nn.Conv1d(
            dim,
            dim * 2,
            kernel_size=1
        )

        self.dw_conv = nn.Conv1d(
            dim,
            dim,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            groups=dim
        )

        self.bn = nn.BatchNorm1d(dim)

        self.pw_conv2 = nn.Conv1d(
            dim,
            dim,
            kernel_size=1
        )

        self.drop = nn.Dropout(dropout)

    def forward(self, x):

        r = x

        x = self.norm(x).transpose(1, 2)

        x = self.pw_conv1(x)

        x, gate = x.chunk(2, dim=1)

        x = x * torch.sigmoid(gate)

        x = F.silu(
            self.bn(
                self.dw_conv(x)
            )
        )

        x = self.drop(
            self.pw_conv2(x)
        ).transpose(1, 2)

        return r + x


class ConformerBlock(nn.Module):

    def __init__(
        self,
        dim,
        num_heads=8,
        ff_expansion=4,
        conv_kernel=31,
        dropout=0.1
    ):

        super().__init__()

        self.ff1 = ConformerFeedForward(
            dim,
            ff_expansion,
            dropout
        )

        self.norm_a = nn.LayerNorm(dim)

        self.attn = nn.MultiheadAttention(
            dim,
            num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.drop_a = nn.Dropout(dropout)

        self.conv = ConformerConvModule(
            dim,
            conv_kernel,
            dropout
        )

        self.ff2 = ConformerFeedForward(
            dim,
            ff_expansion,
            dropout
        )

        self.norm_o = nn.LayerNorm(dim)

    def forward(self, x):

        x = x.transpose(1, 2)

        x = self.ff1(x)

        r = x

        x_n = self.norm_a(x)

        x_a, _ = self.attn(
            x_n,
            x_n,
            x_n
        )

        x = r + self.drop_a(x_a)

        x = self.conv(x)

        x = self.norm_o(
            self.ff2(x)
        )

        return x.transpose(1, 2)


# =========================================================
# 3. MULTI HEAD ATTENTIVE POOLING
# =========================================================

class MultiHeadAttentiveStatsPool(nn.Module):

    def __init__(
        self,
        in_dim,
        bottleneck_dim,
        num_heads=8
    ):

        super().__init__()

        self.num_heads = num_heads

        self.key_proj = nn.Conv1d(
            in_dim,
            bottleneck_dim * num_heads,
            kernel_size=1
        )

        self.val_proj = nn.Conv1d(
            bottleneck_dim * num_heads,
            num_heads,
            kernel_size=1
        )

    def forward(self, x):

        B, C, T = x.shape

        alpha = F.softmax(

            self.val_proj(
                torch.tanh(
                    self.key_proj(x)
                )
            ),

            dim=2
        )

        x_e = x.unsqueeze(1)

        alpha_e = alpha.unsqueeze(2)

        mean = (

            alpha_e * x_e

        ).sum(dim=3)

        std = (

            (
                alpha_e * x_e.pow(2)

            ).sum(dim=3) - mean.pow(2)

        ).clamp(min=1e-9).sqrt()

        return torch.cat([

            mean.reshape(B, -1),

            std.reshape(B, -1)

        ], dim=1)


# =========================================================
# 4. FULL MODEL
# =========================================================

class ECAPA_Conformer(nn.Module):

    def __init__(
        self,
        in_channels,
        channels,
        embd_dim,
        conformer_heads=8,
        conformer_ff_exp=4,
        conformer_kernel=31,
        pooling_heads=8,
        pooling_bottleneck=128,
        use_grad_checkpoint=False
    ):

        super().__init__()

        self.layer1 = Conv1dReluBn(
            in_channels,
            channels,
            kernel_size=5,
            padding=2
        )

        self.se_res2_2 = SE_Res2Block(
            channels,
            kernel_size=3,
            stride=1,
            padding=2,
            dilation=2,
            scale=8
        )

        self.conformer2 = ConformerBlock(
            channels,
            conformer_heads,
            conformer_ff_exp,
            conformer_kernel
        )

        self.se_res2_3 = SE_Res2Block(
            channels,
            kernel_size=3,
            stride=1,
            padding=3,
            dilation=3,
            scale=8
        )

        self.conformer3 = ConformerBlock(
            channels,
            conformer_heads,
            conformer_ff_exp,
            conformer_kernel
        )

        self.se_res2_4 = SE_Res2Block(
            channels,
            kernel_size=3,
            stride=1,
            padding=4,
            dilation=4,
            scale=8
        )

        self.conformer4 = ConformerBlock(
            channels,
            conformer_heads,
            conformer_ff_exp,
            conformer_kernel
        )

        self.agg_conv = nn.Conv1d(
            channels * 3,
            1536,
            kernel_size=1
        )

        self.pooling = MultiHeadAttentiveStatsPool(
            1536,
            pooling_bottleneck,
            pooling_heads
        )

        pool_out_dim = 1536 * 2 * pooling_heads

        self.bn1 = nn.BatchNorm1d(pool_out_dim)

        self.linear = nn.Linear(
            pool_out_dim,
            embd_dim
        )

        self.bn2 = nn.BatchNorm1d(embd_dim)

    def forward(self, x):

        x = x.transpose(1, 2)

        out1 = self.layer1(x)

        out2 = self.conformer2(

            self.se_res2_2(out1) + out1
        )

        out3 = self.conformer3(

            self.se_res2_3(out1 + out2)

            + out1 + out2
        )

        out4 = self.conformer4(

            self.se_res2_4(out1 + out2 + out3)

            + out1 + out2 + out3
        )

        agg = F.relu(

            self.agg_conv(

                torch.cat(
                    [out2, out3, out4],
                    dim=1
                )
            )
        )

        pooled = self.pooling(agg)

        out = self.bn1(pooled)

        out = self.linear(out)

        out = self.bn2(out)

        return out


# =========================================================
# CONFIG
# =========================================================

CONFIG = {

    'in_channels': 80,

    'embedding_dim': 192,

    'model_channels': 512,

    'conformer_num_heads': 8,

    'conformer_ff_expansion': 4,

    'conformer_conv_kernel': 31,

    'pooling_heads': 8,

    'pooling_bottleneck': 128,
}


# =========================================================
# LOAD MODEL
# =========================================================

ecapa_conformer_model = ECAPA_Conformer(

    in_channels=CONFIG['in_channels'],

    channels=CONFIG['model_channels'],

    embd_dim=CONFIG['embedding_dim'],

    conformer_heads=CONFIG['conformer_num_heads'],

    conformer_ff_exp=CONFIG['conformer_ff_expansion'],

    conformer_kernel=CONFIG['conformer_conv_kernel'],

    pooling_heads=CONFIG['pooling_heads'],

    pooling_bottleneck=CONFIG['pooling_bottleneck'],

    use_grad_checkpoint=False
)

checkpoint = torch.load(
    'conformer_voip_finetuned.pt',
    map_location=torch.device('cpu'),
    weights_only=False
)

if 'model_state_dict' in checkpoint:

    ecapa_conformer_model.load_state_dict(
        checkpoint['model_state_dict']
    )

else:

    ecapa_conformer_model.load_state_dict(checkpoint)

ecapa_conformer_model.eval()

print("ECAPA-Conformer loaded successfully!")