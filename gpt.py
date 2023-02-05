import torch
import torch.nn as nn
from torch.nn import functional as F

import time

device = "mps" if torch.backends.mps.is_available() else "cpu"
device = torch.device(device)

with open("input.txt", 'r', encoding="utf-8") as data:
    text = data.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = { ch:i for i,ch in enumerate(chars)} 
itos = { i:ch for i,ch in enumerate(chars)} 

encode = lambda s: [stoi[ch] for ch in s] # encode string to list of integres
decode = lambda l: ''.join(itos[i] for i in l) # decode list of integers to string

text_tensor = torch.tensor(encode(text), dtype=torch.long)

split_ind = int(0.9 * len(text_tensor))
train = text_tensor[:split_ind]
val = text_tensor[split_ind:]

block_size = 8
batch_size = 4
def get_batch(split = "train"):
    data = train if split == "train" else val
    idx = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in idx])
    y = torch.stack([data[i+1: i+block_size+1] for i in idx])
    return x.to(device),y.to(device)

xb, yb = get_batch()


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
    
    def forward(self, idx, target = None):
        logits = self.token_embedding_table(idx)
        loss = None
        if target != None:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)

            target = target.view(B*T)
            loss = F.cross_entropy(logits, target)
        return logits, loss
    
    def generate(self, idx, max_num_tokens):
        for _ in range(max_num_tokens):
            logits, _ = self(idx, None) #B, T, C
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            pred = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, pred], dim=1)
        return idx


model = BigramLanguageModel(vocab_size).to(device)
out, loss = model(xb, yb)
optimizer = torch.optim.AdamW(model.parameters(), lr = 1e-3)

batch_size = 32
start = time.time()
for i in range(1000):

    xb, yb = get_batch()

    _, loss = model(xb, yb)

    optimizer.zero_grad(set_to_none = True)
    loss.backward()
    optimizer.step()
print(i, loss.item(), time.time() - start)

#inp = torch.zeros((1,1), dtype=torch.long).to(device)
#out = model.generate(inp, max_num_tokens=500)[0].tolist()
#print(decode(out))