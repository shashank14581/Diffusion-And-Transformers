#!pip install -q lightning

import torch, lightning as L
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

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

labels = [0,0,0,0,0, 1,1,1,1,1]   # 0 = fruit recipe, 1 = vegetable recipe

vocab = {"<pad>": 0, "<unk>": 1}
for text in texts:
    for word in text.split():
        if word not in vocab:
            vocab[word] = len(vocab)

class RecipeDataset(Dataset):
    def __init__(self, texts, labels, max_len=8):
        self.texts, self.labels, self.max_len = texts, labels, max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        ids = [vocab.get(w, 1) for w in self.texts[idx].split()]
        ids = ids[:self.max_len] + [0] * (self.max_len - len(ids))
        return torch.tensor(ids), torch.tensor(self.labels[idx])

class SelfAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)

    def forward(self, x):
        Q, K, V = self.q(x), self.k(x), self.v(x)
        scores = Q @ K.transpose(-2, -1) / (x.size(-1) ** 0.5)
        weights = F.softmax(scores, dim=-1)
        return weights @ V

class LitRecipeClassifier(L.LightningModule):
    def __init__(self, vocab_size, d_model=32):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.attn = SelfAttention(d_model)
        self.fc = nn.Linear(d_model, 2)

    def forward(self, x):
        x = self.emb(x)
        x = self.attn(x)
        x = x.mean(dim=1)
        return self.fc(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = F.cross_entropy(self(x), y)
        self.log("loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=1e-3)

loader = DataLoader(RecipeDataset(texts, labels), batch_size=4, shuffle=True)

model = LitRecipeClassifier(len(vocab))
trainer = L.Trainer(max_epochs=30, accelerator="auto", logger=False)
trainer.fit(model, loader)

def predict(text):
    ids = [vocab.get(w, 1) for w in text.split()]
    ids = ids[:8] + [0] * (8 - len(ids))
    x = torch.tensor([ids])
    pred = model(x).argmax(1).item()
    return "fruit recipe" if pred == 0 else "vegetable recipe"

print(predict("banana mango smoothie"))
print(predict("potato carrot soup"))
