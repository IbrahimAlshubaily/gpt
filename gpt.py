import torch

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
    return x,y

x, y = get_batch()
print(x.shape)
print(x)
print(y.shape)
print(y)

for i in range(batch_size):
    for j in range(block_size):
        print(f"context: {x[i,:j+1].tolist()}, target: {y[i,j]}")