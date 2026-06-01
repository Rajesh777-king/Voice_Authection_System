import torch
import torch.nn as nn
import torch.nn.functional as F

# =========================================================
# MODULE A: Channel Temporal Attention
# =========================================================

class CTAModule(nn.Module):

    def __init__(
        self,
        in_channels,
        middle_channels=8,
        learnable_pooling=False
    ):
        super().__init__()

        self.learnable_pooling = learnable_pooling

        if learnable_pooling:
            self.register_parameter('freq_weights', None)

        self.conv1 = nn.Conv2d(
            in_channels,
            middle_channels,
            kernel_size=3,
            padding=1,
            bias=False
        )

        self.bn1 = nn.BatchNorm2d(middle_channels)

        self.conv2 = nn.Conv2d(
            middle_channels,
            in_channels,
            kernel_size=3,
            padding=1,
            bias=False
        )

    def _initialize_freq_weights(
        self,
        f_dim,
        device
    ):

        if (
            self.freq_weights is None or
            self.freq_weights.shape[0] != f_dim
        ):

            weights = torch.ones(
                f_dim,
                device=device
            ) / f_dim

            self.freq_weights = nn.Parameter(weights)

        return self.freq_weights

    def forward(self, x):

        B, C, T, F_dim = x.shape

        if self.learnable_pooling:

            w_raw = self._initialize_freq_weights(
                F_dim,
                x.device
            )

            w = torch.softmax(w_raw, dim=0)

            w = w.view(1, 1, 1, F_dim)

            mu = (x * w).sum(
                dim=-1,
                keepdim=True
            )

            var = (
                w * (x - mu) ** 2
            ).sum(
                dim=-1,
                keepdim=True
            )

            z = torch.sqrt(
                var.clamp(min=1e-9)
            ).squeeze(-1)

        else:

            mu = x.mean(
                dim=-1,
                keepdim=True
            )

            z = torch.sqrt(
                (
                    (x - mu) ** 2
                ).mean(dim=-1).clamp(min=1e-9)
            )

        z = z.unsqueeze(-1)

        d = F.relu(
            self.bn1(
                self.conv1(z)
            )
        )

        omega = torch.sigmoid(
            self.conv2(d)
        )

        return x * omega


# =========================================================
# MODULE B: E-Branchformer Block
# =========================================================

class EBranchformerBlock(nn.Module):

    def __init__(
        self,
        d_model,
        num_heads,
        dw_kernel_size=31,
        ffn_expansion=4,
        dropout=0.1,
        layer_scale=1e-5
    ):

        super().__init__()

        self.gamma_ffn1 = nn.Parameter(
            torch.ones(1) * layer_scale
        )

        self.gamma_mhsa = nn.Parameter(
            torch.ones(1) * layer_scale
        )

        self.gamma_conv = nn.Parameter(
            torch.ones(1) * layer_scale
        )

        self.gamma_ffn2 = nn.Parameter(
            torch.ones(1) * layer_scale
        )

        self.norm_ffn1 = nn.LayerNorm(d_model)

        self.ffn1 = self._make_ffn(
            d_model,
            ffn_expansion,
            dropout
        )

        self.norm_global = nn.LayerNorm(d_model)

        self.mhsa = nn.MultiheadAttention(
            d_model,
            num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.dropout_mhsa = nn.Dropout(dropout)

        self.norm_local = nn.LayerNorm(d_model)

        pad = (dw_kernel_size - 1) // 2

        self.pointwise_in = nn.Linear(
            d_model,
            d_model * 2
        )

        self.dw_conv = nn.Conv1d(
            d_model,
            d_model,
            dw_kernel_size,
            padding=pad,
            groups=d_model
        )

        self.dw_bn = nn.BatchNorm1d(d_model)

        self.pointwise_out = nn.Linear(
            d_model,
            d_model
        )

        self.dropout_conv = nn.Dropout(dropout)

        self.merge_proj = nn.Linear(
            d_model * 2,
            d_model
        )

        self.merge_dropout = nn.Dropout(dropout)

        self.norm_ffn2 = nn.LayerNorm(d_model)

        self.ffn2 = self._make_ffn(
            d_model,
            ffn_expansion,
            dropout
        )

        self.norm_out = nn.LayerNorm(d_model)

    @staticmethod
    def _make_ffn(
        d_model,
        expansion,
        dropout
    ):

        return nn.Sequential(
            nn.Linear(
                d_model,
                d_model * expansion
            ),

            nn.SiLU(),

            nn.Dropout(dropout),

            nn.Linear(
                d_model * expansion,
                d_model
            ),

            nn.Dropout(dropout),
        )

    def forward(self, x):

        x = x + (
            self.gamma_ffn1 *
            self.ffn1(
                self.norm_ffn1(x)
            )
        )

        residual = x

        x_norm = self.norm_global(x)

        global_out, _ = self.mhsa(
            x_norm,
            x_norm,
            x_norm
        )

        global_out = self.dropout_mhsa(global_out)

        x_norm2 = self.norm_local(x)

        gate = self.pointwise_in(x_norm2)

        g1, g2 = gate.chunk(2, dim=-1)

        g = g1 * torch.sigmoid(g2)

        g = g.transpose(1, 2)

        g = F.silu(
            self.dw_bn(
                self.dw_conv(g)
            )
        )

        g = g.transpose(1, 2)

        local_out = self.pointwise_out(g)

        local_out = self.dropout_conv(local_out)

        global_out = self.gamma_mhsa * global_out

        local_out = self.gamma_conv * local_out

        merged = torch.cat(
            [global_out, local_out],
            dim=-1
        )

        merged = self.merge_dropout(
            self.merge_proj(merged)
        )

        x = residual + merged

        x = x + (
            self.gamma_ffn2 *
            self.ffn2(
                self.norm_ffn2(x)
            )
        )

        return self.norm_out(x)


# =========================================================
# MODULE C: Weighted MFA
# =========================================================

class WeightedMFA(nn.Module):

    def __init__(self, num_layers):

        super().__init__()

        self.weights = nn.Parameter(
            torch.ones(num_layers)
        )

    def forward(self, layer_outputs):

        w = torch.softmax(
            self.weights,
            dim=0
        )

        weighted = [
            w[i] * layer_outputs[i]
            for i in range(len(layer_outputs))
        ]

        return torch.cat(
            weighted,
            dim=-1
        )


# =========================================================
# MODULE D: Multi Head Attentive Stats Pooling
# =========================================================

class MultiHeadAttentiveStatsPool(nn.Module):

    def __init__(
        self,
        in_dim,
        bottleneck_dim,
        num_heads=4
    ):

        super().__init__()

        self.attn = nn.ModuleList([

            nn.Sequential(

                nn.Conv1d(
                    in_dim,
                    bottleneck_dim,
                    kernel_size=1
                ),

                nn.Tanh(),

                nn.Conv1d(
                    bottleneck_dim,
                    in_dim,
                    kernel_size=1
                ),
            )

            for _ in range(num_heads)

        ])

        self.temperature = nn.Parameter(
            torch.ones(1)
        )

    def forward(self, x):

        means = []

        stds = []

        for head in self.attn:

            scores = head(x) / self.temperature

            alpha = torch.softmax(
                scores,
                dim=2
            )

            mean = (
                alpha * x
            ).sum(dim=2)

            var = (
                alpha * x ** 2
            ).sum(dim=2) - mean ** 2

            std = torch.sqrt(
                var.clamp(min=1e-9)
            )

            means.append(mean)

            stds.append(std)

        mean_agg = torch.stack(
            means,
            dim=0
        ).mean(dim=0)

        std_agg = torch.stack(
            stds,
            dim=0
        ).mean(dim=0)

        return torch.cat(
            [mean_agg, std_agg],
            dim=1
        )


# =========================================================
# FULL MODEL
# =========================================================

class CTAEBranchformer(nn.Module):

    def __init__(self, cfg):

        super().__init__()

        in_ch = cfg['in_channels']

        d_model = cfg['d_model']

        n_blocks = cfg['num_blocks']

        n_heads = cfg['num_heads']

        dw_k = cfg['dw_kernel_size']

        ffn_exp = cfg['ffn_expansion']

        dropout = cfg['dropout']

        c_sub = cfg['conv_subsample_channels']

        cta_m = cfg['cta_middle_channels']

        cta_lp = cfg['cta_learnable_pooling']

        pool_h = cfg['pooling_heads']

        pool_bottleneck = cfg['pooling_bottleneck']

        embd_dim = cfg['embedding_dim']

        self.conv_sub = nn.Sequential(

            nn.Conv2d(
                1,
                c_sub,
                kernel_size=3,
                stride=2,
                padding=1
            ),

            nn.ReLU(),

            nn.Conv2d(
                c_sub,
                c_sub,
                kernel_size=3,
                stride=2,
                padding=1
            ),

            nn.ReLU(),
        )

        f_prime = (in_ch + 3) // 4

        self.cta = CTAModule(
            c_sub,
            middle_channels=cta_m,
            learnable_pooling=cta_lp
        )

        self.flatten_proj = nn.Linear(
            c_sub * f_prime,
            d_model
        )

        self.input_dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([

            EBranchformerBlock(
                d_model,
                n_heads,
                dw_k,
                ffn_exp,
                dropout
            )

            for _ in range(n_blocks)

        ])

        self.mfa = WeightedMFA(n_blocks)

        mfa_out_dim = d_model * n_blocks

        self.pre_pool_norm = nn.LayerNorm(mfa_out_dim)

        self.pooling = MultiHeadAttentiveStatsPool(
            in_dim=mfa_out_dim,
            bottleneck_dim=pool_bottleneck,
            num_heads=pool_h
        )

        pool_out_dim = mfa_out_dim * 2

        self.bn1 = nn.BatchNorm1d(pool_out_dim)

        self.linear = nn.Linear(
            pool_out_dim,
            embd_dim
        )

        self.bn2 = nn.BatchNorm1d(embd_dim)

    def forward(self, x):

        x = x.unsqueeze(1)

        x = self.conv_sub(x)

        x = self.cta(x)

        B, C, T, F_dim = x.shape

        x = x.permute(0, 2, 1, 3)

        x = x.reshape(
            B,
            T,
            C * F_dim
        )

        x = self.input_dropout(
            self.flatten_proj(x)
        )

        block_outputs = []

        for block in self.blocks:

            x = block(x)

            block_outputs.append(x)

        mfa_out = self.mfa(block_outputs)

        mfa_out = self.pre_pool_norm(mfa_out)

        mfa_out = mfa_out.transpose(1, 2)

        pooled = self.pooling(mfa_out)

        out = self.bn1(pooled)

        out = self.linear(out)

        out = self.bn2(out)

        return out


# =========================================================
# CONFIG
# =========================================================

CONFIG = {

    'in_channels': 80,

    'd_model': 256,

    'num_heads': 4,

    'num_blocks': 6,

    'ffn_expansion': 4,

    'dw_kernel_size': 31,

    'embedding_dim': 192,

    'dropout': 0.1,

    'conv_subsample_channels': 64,

    'cta_learnable_pooling': False,

    'cta_middle_channels': 8,

    'pooling_heads': 4,

    'pooling_bottleneck': 128,
}


# =========================================================
# LOAD MODEL
# =========================================================

cta_model = CTAEBranchformer(CONFIG)

checkpoint = torch.load(
    'ebranchformer_scracth_kathbath.pt',
    map_location=torch.device('cpu')
)

cta_model.load_state_dict(
    checkpoint['model_state_dict']
)

cta_model.eval()

print("CTA-E-Branchformer loaded successfully!")