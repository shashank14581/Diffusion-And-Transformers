#!pip install -q lightning

import re
import math
import torch
import lightning as L
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from collections import Counter
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ---------------- DATA ----------------
texts = [
    "apple banana smoothie with honey",
    "mango orange fruit salad",
    "strawberry blueberry yogurt bowl",
    "grapes pineapple fresh juice",
    "banana apple cinnamon oatmeal",
    "carrot potato vegetable soup",
    "spinach tomato onion curry",
    "broccoli peas garlic stir fry",
    "cabbage carrot beans salad",
    "potato cauliflower vegetable roast",
]

labels = [0,0,0,0,0, 1,1,1,1,1]

train_texts, val_texts, train_labels, val_labels = train_test_split(
    texts,
    labels,
    test_size=0.2,
    stratify=labels,
    random_state=42
)

# ---------------- PREPROCESS ----------------
def preprocess(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text.split()

train_tokens = [preprocess(t) for t in train_texts]
val_tokens = [preprocess(t) for t in val_texts]

# ---------------- VOCAB ----------------
word_counts = Counter(w for sent in train_tokens for w in sent)

vocab = {"<pad>": 0, "<unk>": 1}

for word, count in word_counts.items():
    vocab[word] = len(vocab)

def encode(tokens):
    return [vocab.get(w, vocab["<unk>"]) for w in tokens]

train_encoded = [encode(t) for t in train_tokens]
val_encoded = [encode(t) for t in val_tokens]

# ---------------- DATASET ----------------
class RecipeDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return torch.tensor(self.texts[idx]), torch.tensor(self.labels[idx])

def collate_fn(batch):
    texts, labels = zip(*batch)

    max_len = max(len(x) for x in texts)
    padded = torch.zeros(len(texts), max_len, dtype=torch.long)

    for i, x in enumerate(texts):
        padded[i, :len(x)] = x

    return padded, torch.tensor(labels)

train_loader = DataLoader(
    RecipeDataset(train_encoded, train_labels),
    batch_size=4,
    shuffle=True,
    collate_fn=collate_fn
)

val_loader = DataLoader(
    RecipeDataset(val_encoded, val_labels),
    batch_size=4,
    shuffle=False,
    collate_fn=collate_fn
)

# ---------------- POSITIONAL ENCODING ----------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super().__init__()

        pe = torch.zeros(max_len, d_model)

        position = torch.arange(0, max_len).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe)

    def forward(self, x):
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]

# ---------------- FULL TRANSFORMER MODEL ----------------
class LitRecipeTransformer(L.LightningModule):
    def __init__(
        self,
        vocab_size,
        class_weights,
        d_model=64,
        nhead=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        dim_feedforward=128,
        dropout=0.1,
        num_classes=2
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            d_model,
            padding_idx=0
        )

        self.pos_encoder = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers
        )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )

        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_decoder_layers
        )

        # Learned decoder query token
        self.query_token = nn.Parameter(torch.randn(1, 1, d_model))

        self.fc = nn.Linear(d_model, num_classes)

        self.register_buffer("class_weights", class_weights)

    def forward(self, x):
        # x shape: (batch, src_seq_len)

        src_key_padding_mask = x == 0
        # True means ignore padding token

        src = self.embedding(x)
        src = self.pos_encoder(src)

        memory = self.encoder(
            src,
            src_key_padding_mask=src_key_padding_mask
        )

        batch_size = x.size(0)

        tgt = self.query_token.repeat(batch_size, 1, 1)
        # shape: (batch, 1, d_model)

        decoded = self.decoder(
            tgt=tgt,
            memory=memory,
            memory_key_padding_mask=src_key_padding_mask
        )

        cls_output = decoded[:, 0, :]

        logits = self.fc(cls_output)

        return logits

    def training_step(self, batch, batch_idx):
        x, y = batch

        logits = self(x)

        loss = F.cross_entropy(
            logits,
            y,
            weight=self.class_weights
        )

        self.log("train_loss", loss, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch

        logits = self(x)

        loss = F.cross_entropy(
            logits,
            y,
            weight=self.class_weights
        )

        self.log("val_loss", loss, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=1e-3)

# ---------------- CLASS WEIGHTS ----------------
weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(train_labels),
    y=train_labels
)

weights = torch.tensor(weights, dtype=torch.float)

# ---------------- TRAIN ----------------
model = LitRecipeTransformer(
    vocab_size=len(vocab),
    class_weights=weights,
    d_model=64,
    nhead=4,
    num_encoder_layers=2,
    num_decoder_layers=2,
    dim_feedforward=128,
    dropout=0.1
)

trainer = L.Trainer(
    max_epochs=40,
    accelerator="auto",
    logger=False,
    enable_checkpointing=False
)

trainer.fit(model, train_loader, val_loader)

# ---------------- EVAL ----------------
device = model.device
model.eval()

preds, actuals = [], []

with torch.no_grad():
    for x, y in val_loader:
        x = x.to(device)

        logits = model(x)
        pred = logits.argmax(dim=1).cpu().numpy()

        preds.extend(pred)
        actuals.extend(y.numpy())

print("Accuracy:", accuracy_score(actuals, preds))
print("Precision:", precision_score(actuals, preds, zero_division=0))
print("Recall:", recall_score(actuals, preds, zero_division=0))
print("F1:", f1_score(actuals, preds, zero_division=0))

# ---------------- PREDICT ----------------
def predict(text):
    tokens = preprocess(text)
    ids = encode(tokens)

    x = torch.tensor([ids], dtype=torch.long).to(model.device)

    model.eval()

    with torch.no_grad():
        pred = model(x).argmax(dim=1).item()

    return "vegetable" if pred == 1 else "fruit"

print(predict("banana mango smoothie"))
print(predict("potato carrot soup"))
print(predict("Blueberry muffins"))
print(predict("Spinach curry"))
print(predict("Avocado toast"))
