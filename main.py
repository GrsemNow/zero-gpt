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


def generate_step(logits_fn, idx, count, block_size,temperature=1.0):
    sequence = idx
    for _ in range(count):
        idx_cond = sequence[:, -block_size:] if sequence.shape[1] > block_size else sequence 

        logits = logits_fn(idx_cond)[:, -1, :] / temperature
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


@torch.no_grad()
def estimate_loss(logits_fn, loss_fn, train_batch, val_batch, eval_iters=100):
    # return middle losses

    loss_train = 0.0
    for _ in range(eval_iters):
        x, y = train_batch()
        logits = logits_fn(x)
        loss = loss_fn(logits, y)
        loss_train += loss.item()

    loss_val = 0.0
    for _ in range(eval_iters):
        x, y = val_batch()
        logits = logits_fn(x)
        loss = loss_fn(logits, y)
        loss_val += loss.item()
    
    return loss_train / eval_iters, loss_val / eval_iters


def self_attention(x, W_q, W_k, W_v, head_size, tril_mask=True):
    # ONE head of self attension
    B, T, C = x.shape

    q = x @ W_q # (B, T, head_size)
    k = x @ W_k # (...)
    v = x @ W_v # (...)

    wei = q @ k.transpose(-2, -1)
    wei = wei / (head_size ** 0.5)

    if tril_mask:
        tril = torch.tril(torch.ones(T, T))
        wei = wei.masked_fill(tril == 0, float('-inf'))

    wei = F.softmax(wei, dim=-1) # (B, T, T)
    out = wei @ v 

    return out


def make_attention_model(voc_size, n_embd, block_size, head_size):
    embedding_weight = torch.randn(voc_size, n_embd, requires_grad=True)
    position_embedding_table = torch.randn(block_size, n_embd, requires_grad=True)

    # weightes
    W_q = torch.randn(n_embd, head_size, requires_grad=True)
    W_k = torch.randn(n_embd, head_size, requires_grad=True)
    W_v = torch.randn(n_embd, head_size, requires_grad=True)
    
    # linear layer
    lm_head_weight = torch.randn(head_size, voc_size, requires_grad=True)
    lm_head_bias = torch.randn(voc_size, requires_grad=True)
    
    def logits_fn(idx):
        # (B, T) -> (B, T, C) logits
        B, T = idx.shape
        tok_emb = F.embedding(idx, embedding_weight)
        pos_emb = F.embedding(torch.arange(T), position_embedding_table)
        x = (tok_emb + pos_emb)
        
        # self-attention
        x = self_attention(x, W_q, W_k, W_v, head_size)

        logits = x @ lm_head_weight + lm_head_bias
        return logits

    def loss_fn(logits, targets):
        # (logits + targets) -> loss
        B, T, C = logits.shape
        return F.cross_entropy(logits.view(B*T, C), targets.view(B*T))
        
    params = {
            'embedding_weight': embedding_weight,
            'position_embedding_table': position_embedding_table,
            'W_q': W_q,
            'W_k': W_k,
            'W_v': W_v,
            'lm_head_weight': lm_head_weight,
            'lm_head_bias': lm_head_bias,
            }
    return logits_fn, loss_fn, params

if __name__ == "__main__":
    torch.manual_seed(239)

    # config
    block_size = 64
    batch_size = 32
    lr = 1e-3
    count = 30000
    eval_iters = 200
    n_embd = 64
    head_size = 32
    
    # prepare data
    trans = make_translation_table()
    data, chars, encode, decode = prepare_data("data/poems.txt", trans)
    voc_size = len(chars)
     
    n = int(0.9*len(data))
    train_data, val_data = data[:n], data[n:]
    
    # model
    logits_fn, loss_fn, params = make_attention_model(voc_size, n_embd, block_size, head_size)
    optimizer = torch.optim.AdamW(list(params.values()), lr=lr)
    
    # studing
    train_batch = make_batch(train_data, block_size, batch_size)
    val_batch = make_batch(val_data, block_size, batch_size)
    tr = train(optimizer, logits_fn, loss_fn, train_batch, count=count)

    for step, loss_value in enumerate(tr, 1):
        if step % 500 == 0:
            loss_train, loss_val = estimate_loss(logits_fn, loss_fn, train_batch, val_batch, eval_iters)
            print(step, f"{loss_train:.4f}, {loss_val:.4f}")
    
    # generation
    idx = torch.zeros((1,1), dtype=torch.long)
    gen = generate_step(logits_fn, idx, count=200, block_size=block_size)
    for i, seq in enumerate(gen, 1):
        pass
    print(decode(seq[0].tolist()))
