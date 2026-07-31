import torch
from torch.nn import functional as F

def make_translation_table():
    return str.maketrans({
        'ъ': 'ь', 'ё': 'е', 'Ё': 'Е',
        '"': '', ';': ',', '\u00A0': ' ',
        '«': '', '»': '', '(': '', ')': '',
        '|': '\n'
    })

def prepare_data(filepath, trans, splitter="---"):
    with open(filepath, "r", encoding="utf-8") as f:
        text_trash = f.read()
    
    text = clear_text(text_trash, trans, lit_title=True, splitter=splitter)
    chars = sorted(list(set(text)))
    encode, decode = coders(chars)
    data = torch.tensor(encode(text), dtype=torch.long)
    
    return data, chars, encode, decode


def make_batch(data, block_size, batch_size):
    # fabric of batches
    
    def sample_indices():
        # chose random position 
        return torch.randint(len(data) - block_size, (batch_size,))

    def slice_data(indices):
        # create one batch
        x = torch.stack(list(map(lambda i: data[i:i+block_size], indices.tolist())))
        y = torch.stack(list(map(lambda i: data[i+1:i+block_size+1], indices.tolist())))
        return x, y

    return lambda: slice_data(sample_indices())


def make_bigram_model(voc_size):
    embedding_weight = torch.randn(voc_size, voc_size, requires_grad=True)
    
    def logits_fn(idx):
        # (B, T) -> (B, T, C) logits
        return F.embedding(idx, embedding_weight)

    def loss_fn(logits, targets):
        # (logits + targets) -> loss
        B, T, C = logits.shape
        return F.cross_entropy(logits.view(B*T, C), targets.view(B*T))
        
    return logits_fn, loss_fn, embedding_weight


def generate_step(logits_fn, idx, count, temperature=1.0):
    sequence = idx
    for _ in range(count):
        logits = logits_fn(sequence)[:, -1, :] / temperature
        probs = F.softmax(logits, dim=1)
        idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
        sequence = torch.cat((sequence, idx_next), dim=1) # (B, T+1)
        yield sequence


def train(optimizer, logits_fn, loss_fn, get_batch, count):
    for step in range(count):
        x, y = get_batch()
        
        logits = logits_fn(x)
        loss = loss_fn(logits, y)
        
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        
        yield loss.item()


def clear_text(text, dct, lit_title=True, splitter="---"):
    # clear text out of mask in dictionary and makes upper words to lower.
    cnt_upper = lambda x: x.lower() if sum(1 for ch in x if ch.isupper()) > 1 and lit_title else x
    proc_line = lambda line: ' '.join(map(cnt_upper, line.split()))
    proc_text = lambda text: '\n'.join(map(proc_line, text.split('\n')))

    return '\n'.join(proc_text(text).translate(dct).split(splitter))


def coders(chars):
    # makes encoder & decoder
    stoi = { ch:i for i,ch in enumerate(chars) }
    itos = { i:ch for i,ch in enumerate(chars) }
    encoder = lambda s: [stoi[ch] for ch in s]
    decoder = lambda q: ''.join([itos[i] for i in q]) 
    return encoder, decoder


if __name__ == "__main__":
    torch.manual_seed(239)

    # config
    block_size = 64
    batch_size = 32
    lr = 1e-3
    count = 10000
    
    # prepare data
    trans = make_translation_table()
    data, chars, encode, decode = prepare_data("data/poems.txt", trans)
    voc_size = len(chars)
     
    n = int(0.9*len(data))
    train_data, val_data = data[:n], data[n:]
    
    # model
    logits_fn, loss_fn, weights = make_bigram_model(voc_size)
    optimizer = torch.optim.AdamW([weights], lr=lr)
    
    # studing
    train_batch = make_batch(train_data, block_size, batch_size)
    tr = train(optimizer, logits_fn, loss_fn, train_batch, count=count)

    for step, loss_value in enumerate(tr, 1):
        if step % 500 == 0:
            print(step, f"{loss_value:.4f}")
    
    # generation
    idx = torch.zeros((1,1), dtype=torch.long)
    gen = generate_step(logits_fn, idx, count=100)
    for i, seq in enumerate(gen, 1):
        pass
    print(decode(seq[0].tolist()))
