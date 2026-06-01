import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
import numpy as np

# ============================================================
# CONFIG
# ============================================================

CONFIG = {

    'in_channels': 80,

    'model_channels': 512,

    'embedding_dim': 192,
}

# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# ============================================================
# RES2 BLOCK
# ============================================================

class Res2Conv1dReluBn(nn.Module):

    def __init__(
        self,
        channels,
        kernel_size=1,
        stride=1,
        padding=0,
        dilation=1,
        bias=False,
        scale=8
    ):

        super().__init__()

        assert channels % scale == 0

        self.scale = scale

        self.width = channels // scale

        self.nums = scale - 1

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

        spx = torch.split(
            x,
            self.width,
            dim=1
        )

        outputs = []

        sp = None

        for i in range(self.nums):

            if i == 0:

                sp = spx[i]

            else:

                sp = sp + spx[i]

            sp = self.convs[i](sp)

            sp = F.relu(sp)

            sp = self.bns[i](sp)

            outputs.append(sp)

        outputs.append(
            spx[self.nums]
        )

        return torch.cat(
            outputs,
            dim=1
        )

# ============================================================
# BASIC CONV BLOCK
# ============================================================

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

        self.bn = nn.BatchNorm1d(
            out_channels
        )

    def forward(self, x):

        return self.bn(
            F.relu(
                self.conv(x)
            )
        )

# ============================================================
# SQUEEZE EXCITATION
# ============================================================

class SE_Connect(nn.Module):

    def __init__(
        self,
        channels,
        reduction=8
    ):

        super().__init__()

        self.linear1 = nn.Linear(

            channels,

            channels // reduction
        )

        self.linear2 = nn.Linear(

            channels // reduction,

            channels
        )

    def forward(self, x):

        out = x.mean(dim=2)

        out = F.relu(
            self.linear1(out)
        )

        out = torch.sigmoid(
            self.linear2(out)
        )

        out = out.unsqueeze(2)

        return x * out

# ============================================================
# SE RES2 BLOCK
# ============================================================

class SERes2Block(nn.Module):

    def __init__(
        self,
        channels,
        kernel_size,
        stride,
        padding,
        dilation,
        scale=8
    ):

        super().__init__()

        self.block = nn.Sequential(

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

    def forward(self, x):

        return self.block(x)

# ============================================================
# ATTENTIVE STATISTICS POOLING
# ============================================================

class AttentiveStatsPool(nn.Module):

    def __init__(
        self,
        in_dim,
        bottleneck_dim=128
    ):

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

        alpha = torch.tanh(
            self.linear1(x)
        )

        alpha = self.linear2(alpha)

        alpha = torch.softmax(
            alpha,
            dim=2
        )

        mean = torch.sum(
            alpha * x,
            dim=2
        )

        var = torch.sum(

            alpha * (x ** 2),

            dim=2

        ) - mean ** 2

        std = torch.sqrt(
            var.clamp(min=1e-9)
        )

        return torch.cat(
            [mean, std],
            dim=1
        )

# ============================================================
# ECAPA TDNN
# ============================================================

class ECAPA_TDNN(nn.Module):

    def __init__(
        self,
        in_channels=80,
        channels=512,
        embd_dim=192
    ):

        super().__init__()

        self.layer1 = Conv1dReluBn(

            in_channels,

            channels,

            kernel_size=5,

            padding=2
        )

        self.layer2 = SERes2Block(

            channels,

            kernel_size=3,

            stride=1,

            padding=2,

            dilation=2,

            scale=8
        )

        self.layer3 = SERes2Block(

            channels,

            kernel_size=3,

            stride=1,

            padding=3,

            dilation=3,

            scale=8
        )

        self.layer4 = SERes2Block(

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

        self.bn1 = nn.BatchNorm1d(
            3072
        )

        self.linear = nn.Linear(

            3072,

            embd_dim
        )

        self.bn2 = nn.BatchNorm1d(
            embd_dim
        )

    def forward(self, x):

        x = x.transpose(1, 2)

        out1 = self.layer1(x)

        out2 = self.layer2(out1) + out1

        out3 = self.layer3(
            out1 + out2
        ) + out1 + out2

        out4 = self.layer4(
            out1 + out2 + out3
        ) + out1 + out2 + out3

        out = torch.cat(

            [out2, out3, out4],

            dim=1
        )

        out = F.relu(
            self.conv(out)
        )

        out = self.pooling(out)

        out = self.bn1(out)

        out = self.linear(out)

        out = self.bn2(out)

        return out

# ============================================================
# LOAD MODEL
# ============================================================

model = ECAPA_TDNN(

    in_channels=CONFIG['in_channels'],

    channels=CONFIG['model_channels'],

    embd_dim=CONFIG['embedding_dim']

).to(device)

checkpoint = torch.load(

    "ecapa_zeroshot_kathpath.pt",

    map_location=device,

    weights_only=False
)

# ============================================================
# HANDLE CHECKPOINT FORMATS
# ============================================================

if isinstance(checkpoint, dict):

    if 'model_state_dict' in checkpoint:

        model.load_state_dict(

            checkpoint['model_state_dict'],

            strict=False
        )

    elif 'state_dict' in checkpoint:

        model.load_state_dict(

            checkpoint['state_dict'],

            strict=False
        )

    else:

        model.load_state_dict(

            checkpoint,

            strict=False
        )

else:

    model.load_state_dict(

        checkpoint,

        strict=False
    )

model.eval()

print("ECAPA Zeroshot Kathbath loaded successfully!")

# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_ecapa_kathbath_features(audio_path):

    signal, sr = librosa.load(

        audio_path,

        sr=16000
    )

    mel = librosa.feature.melspectrogram(

        y=signal,

        sr=sr,

        n_mels=80,

        n_fft=512,

        hop_length=160,

        win_length=400
    )

    mel = librosa.power_to_db(

        mel,

        ref=np.max
    )

    mel = (

        mel - np.mean(mel)

    ) / (

        np.std(mel) + 1e-9
    )

    mel = mel.T

    features = torch.FloatTensor(

        mel

    ).unsqueeze(0).to(device)

    return features

# ============================================================
# GENERATE EMBEDDING
# ============================================================

def get_ecapa_kathbath_embedding(audio_path):

    features = extract_ecapa_kathbath_features(
        audio_path
    )

    with torch.no_grad():

        embedding = model(features)

    embedding = embedding.squeeze()

    embedding = embedding.cpu().numpy()

    return embedding.tolist()

# ============================================================
# COSINE SIMILARITY
# ============================================================

def compute_similarity(
    embedding1,
    embedding2
):

    embedding1 = np.array(embedding1)

    embedding2 = np.array(embedding2)

    denominator = (

        np.linalg.norm(embedding1)

        *

        np.linalg.norm(embedding2)
    )

    if denominator == 0:

        return 0.0

    similarity = np.dot(

        embedding1,

        embedding2

    ) / denominator

    return float(similarity)