import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

EPOCH = 500
BATCH = 128
EMBSIZE = 100
HIDSIZE = 256
DROPRAT = 0.2
        
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

filename = 'alice.txt'
raw_text = open('Data/' + filename, 'r', encoding='utf-8').read()
raw_text = raw_text.lower()

chars = sorted(list(set(raw_text)))
char_to_int = dict((c, i) for i, c in enumerate(chars))

n_chars = len(raw_text)
n_vocab = len(chars)
n_seqch = 100

class TextGen(nn.Module):
    def __init__(self):
        super(TextGen, self).__init__()
        self.emb = nn.Embedding(n_vocab, EMBSIZE)
        self.gru1 = nn.GRU(EMBSIZE, HIDSIZE, 1, batch_first=True)
        self.gru2 = nn.GRU(EMBSIZE, HIDSIZE, 1, batch_first=True)
        self.gru3 = nn.GRU(EMBSIZE, HIDSIZE, 1, batch_first=True)
        self.dro = nn.Dropout(p=DROPRAT)
        self.fcl = nn.Linear(HIDSIZE, n_vocab)
    
    def forward(self, inp):
        inp = self.emb(inp)
        opt, _ = self.gru1(inp)
        opt = self.dro(opt)
        opt, _ = self.gru2(inp)
        opt = self.dro(opt)
        _, opt = self.gru3(inp)
        opt = self.dro(opt)
        opt = self.fcl(opt).view(-1, n_vocab)
        
        return opt

X, y = [], []
for i in range(0, n_chars - n_seqch, 1):
    seq_in = raw_text[i:i + n_seqch]
    seq_out = raw_text[i + n_seqch]
    X.append([char_to_int[char] for char in seq_in])
    y.append(char_to_int[seq_out])

model = TextGen().to(device)
optim = optim.Adagrad(model.parameters())
loss = nn.CrossEntropyLoss()

for ep in range(EPOCH):
    l = 0
    l_add = 0
    xb = []
    yb = []
    for i in tqdm(range(len(X)), ncols=75):
        xb.append(X[i])
        yb.append(y[i])
        
        if len(xb) % BATCH == 0 or i == len(X) - 1:
            xb = torch.tensor(xb, device=device).view(len(xb), n_seqch)
            yb = torch.tensor(yb, device=device).view(len(xb))
            opt = model(xb)
            l = loss(opt, yb)
            
            optim.zero_grad()
            l.backward()
            optim.step()
            
            l_add = l_add + l.item()
            
            l = 0
            xb = []
            yb = []
    print('Ep {} loss => {}'.format(ep + 1, l_add / len(X)))

with torch.no_grad(), open('Output/' + filename, 'w', encoding='utf-8') as f:
    model_eval = model.eval()
    inp = X[50000]
    output_words = [chars[i] for i in inp]
    for i in range(5000):
        opt = torch.tensor(inp, device=device).view(1, n_seqch)
        opt = model_eval(opt)
        _, opt = opt.topk(1, 1)
        output = opt.cpu().view(-1).item()
        output_words.append(chars[output])
        inp = inp[1:] + [output]
    f.write(''.join(output_words))
        