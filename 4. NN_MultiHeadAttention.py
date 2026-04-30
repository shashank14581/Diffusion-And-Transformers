#!pip install -q lightning

import re, torch, lightning as L
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
    texts, labels, test_size=0.2, stratify=labels, random_state=42
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

# ---------------- LIGHTNING MODEL ----------------
class LitRecipeClassifier(L.LightningModule):
    def __init__(
        self,
        vocab_size,
        class_weights,
        d_model=64,
        num_heads=4,
        num_classes=2
    ):
        super().__init__()

        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=0)

        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            batch_first=True
        )

        self.fc = nn.Linear(d_model, num_classes)

        self.register_buffer("class_weights", class_weights)

    def forward(self, x):
        # x shape: (batch, seq_len)

        key_padding_mask = x == 0
        # shape: (batch, seq_len)
        # True means: ignore this token

        x = self.emb(x)
        # shape: (batch, seq_len, d_model)

        attn_out, attn_weights = self.attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=key_padding_mask
        )

        token_mask = (~key_padding_mask).unsqueeze(-1)
        # shape: (batch, seq_len, 1)

        attn_out = attn_out * token_mask

        pooled = attn_out.sum(dim=1) / token_mask.sum(dim=1).clamp(min=1)

        return self.fc(pooled)

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
model = LitRecipeClassifier(
    vocab_size=len(vocab),
    class_weights=weights,
    d_model=64,
    num_heads=4
)

trainer = L.Trainer(
    max_epochs=30,
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

        output = model(x)
        pred = output.argmax(dim=1).cpu().numpy()

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
