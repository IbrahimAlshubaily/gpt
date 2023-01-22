
with open("input.txt", 'r', encoding="utf-8") as data:
    text = data.read()


chars = sorted(list(set(text)))
vocab_size = len(chars)

#print("vocab size: ", vocab_size)
#print("vocab: ", ''.join(chars))

stoi = { ch:i for i,ch in enumerate(chars)} 
itos = { i:ch for i,ch in enumerate(chars)} 

encode = lambda s: [stoi[ch] for ch in s] # encode string to list of integres
decode = lambda l: ''.join(itos[i] for i in l) # decode list of integers to string

#sample = "hii there"
#print(encode(sample))
#print(decode(encode(sample)))

