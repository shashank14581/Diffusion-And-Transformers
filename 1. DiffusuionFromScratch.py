# pip install lightning

import math, torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision, torchvision.transforms as T
import lightning as L
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# -----------------------
# Fast Config
# -----------------------
IMG = 32
BATCH = 64
T_STEPS = 50
EMB = 64
LR = 2e-4

# -----------------------
# CIFAR prompts
# -----------------------
classes = torchvision.datasets.CIFAR100(root="./data", train=True, download=True).classes
prompts = [f"a photo of a {c.replace('_',' ')}" for c in classes]

words = sorted(set(" ".join(prompts).split()))
vocab = {w: i + 1 for i, w in enumerate(words)}
vocab["<pad>"] = 0

def tok(prompt, max_len=8):
    ids = [vocab.get(w, 0) for w in prompt.lower().split()][:max_len]
    return torch.tensor(ids + [0] * (max_len - len(ids)))

prompt_tokens = torch.stack([tok(p) for p in prompts])

# -----------------------
# Data
# -----------------------
class CIFARData(L.LightningDataModule):
    def train_dataloader(self):
        tfm = T.Compose([
            T.ToTensor(),
            T.Normalize((0.5,) * 3, (0.5,) * 3)
        ])

        ds = torchvision.datasets.CIFAR100(
            "./data",
            train=True,
            download=True,
            transform=tfm
        )

        return DataLoader(
            ds,
            batch_size=BATCH,
            shuffle=True,
            num_workers=2,
            pin_memory=True
        )

# -----------------------
# Model
# -----------------------
class TimeEmb(nn.Module):
    def forward(self, t):
        half = EMB // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device) / half
        )
        x = t[:, None] * freqs[None]
        return torch.cat([x.sin(), x.cos()], dim=1)

class Block(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.c1 = nn.Conv2d(c1, c2, 3, padding=1)
        self.c2 = nn.Conv2d(c2, c2, 3, padding=1)
        self.e = nn.Linear(EMB, c2)
        self.skip = nn.Conv2d(c1, c2, 1) if c1 != c2 else nn.Identity()

    def forward(self, x, emb):
        h = F.silu(self.c1(x))
        h = h + self.e(emb)[:, :, None, None]
        h = F.silu(self.c2(h))
        return h + self.skip(x)

class TinyUNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.time = nn.Sequential(
            TimeEmb(),
            nn.Linear(EMB, EMB),
            nn.SiLU()
        )

        self.text = nn.Embedding(len(vocab), EMB, padding_idx=0)

        self.inp = nn.Conv2d(3, 32, 3, padding=1)

        self.d1 = Block(32, 64)
        self.d2 = Block(64, 128)
        self.mid = Block(128, 128)

        self.u1 = Block(128 + 64, 64)
        self.u2 = Block(64 + 32, 32)

        self.out = nn.Conv2d(32, 3, 1)

    def text_emb(self, tokens):
        x = self.text(tokens)
        mask = (tokens != 0).float()[:, :, None]
        return (x * mask).sum(1) / mask.sum(1).clamp(min=1)

    def forward(self, x, t, tokens):
        emb = self.time(t) + self.text_emb(tokens)

        x0 = self.inp(x)

        x1 = self.d1(F.avg_pool2d(x0, 2), emb)
        x2 = self.d2(F.avg_pool2d(x1, 2), emb)

        x = self.mid(x2, emb)

        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.u1(torch.cat([x, x1], dim=1), emb)

        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.u2(torch.cat([x, x0], dim=1), emb)

        return self.out(x)

# -----------------------
# Lightning DDPM
# -----------------------
class DDPM(L.LightningModule):
    def __init__(self):
        super().__init__()

        self.net = TinyUNet()

        betas = torch.linspace(1e-4, 0.02, T_STEPS)
        alphas = 1 - betas
        ahat = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("ahat", ahat)

    def noise_image(self, x, t):
        noise = torch.randn_like(x)
        a = self.ahat[t][:, None, None, None]
        return a.sqrt() * x + (1 - a).sqrt() * noise, noise

    def training_step(self, batch, batch_idx):
        x, y = batch

        tokens = prompt_tokens[y].to(self.device)
        t = torch.randint(0, T_STEPS, (x.size(0),), device=self.device)

        x_noisy, noise = self.noise_image(x, t)
        pred = self.net(x_noisy, t, tokens)

        loss = F.mse_loss(pred, noise)

        self.log("loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=LR)

    @torch.no_grad()
    def sample(self, prompt, n=4):
        self.eval()

        tokens = tok(prompt).unsqueeze(0).repeat(n, 1).to(self.device)
        x = torch.randn(n, 3, IMG, IMG, device=self.device)

        for i in reversed(range(T_STEPS)):
            t = torch.full((n,), i, device=self.device)

            pred = self.net(x, t, tokens)

            a = self.alphas[i]
            ah = self.ahat[i]
            b = self.betas[i]

            z = torch.randn_like(x) if i > 0 else torch.zeros_like(x)

            x = (1 / a.sqrt()) * (
                x - ((1 - a) / (1 - ah).sqrt()) * pred
            ) + b.sqrt() * z

        return (x.clamp(-1, 1) + 1) / 2

# -----------------------
# Train fast
# -----------------------
dm = CIFARData()
model = DDPM()

trainer = L.Trainer(
    max_epochs=3,
    accelerator="auto",
    devices=1,
    precision="16-mixed" if torch.cuda.is_available() else "32",
    limit_train_batches=0.25,
    log_every_n_steps=10
)

trainer.fit(model, dm)

# -----------------------
# Generate
# -----------------------
samples = model.sample("a photo of a motorcycle", n=4)

grid = torchvision.utils.make_grid(samples.cpu(), nrow=2)

plt.figure(figsize=(5, 5))
plt.imshow(grid.permute(1, 2, 0))
plt.axis("off")
plt.show()
