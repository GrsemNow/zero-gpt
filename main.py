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


def make_attention_model(voc_size, n_embd, block_size, num_head):
    head_size = n_embd // num_head

    embedding_weight = torch.randn(voc_size, n_embd, requires_grad=True)
    position_embedding_table = torch.randn(block_size, n_embd, requires_grad=True)
    
    # weightes
    W_q_list, W_k_list, W_v_list = make_multi_head_params(n_embd, num_head, head_size)
    
    # feed-forvard params
    W1, b1, W2, b2 = make_ffn_params(n_embd)

    # linear layer
    scale_out = 1.0 / (n_embd ** 0.5)
    lm_head_weight = scaled_randn(n_embd, voc_size, scale=scale_out)
    lm_head_bias = torch.zeros(voc_size, requires_grad=True)    
    
    def logits_fn(idx):
        # (B, T) -> (B, T, C) logits
        B, T = idx.shape
        tok_emb = F.embedding(idx, embedding_weight)
        pos_emb = F.embedding(torch.arange(T), position_embedding_table)
        x = tok_emb + pos_emb
        
        # self-attention
        x = multi_head_attention(x, W_q_list, W_k_list, W_v_list, tril_mask=True)
        
        # ffn
        x = feed_forward(x, W1, b1, W2, b2)

        logits = x @ lm_head_weight + lm_head_bias
        return logits

    def loss_fn(logits, targets):
        # (logits + targets) -> loss
        B, T, C = logits.shape
        return F.cross_entropy(logits.view(B*T, C), targets.view(B*T))
        
    params = {
            'embedding_weight': embedding_weight,
            'position_embedding_table': position_embedding_table,
            **{f'W_q_{i}': w for i, w in enumerate(W_q_list)},
            **{f'W_k_{i}': w for i, w in enumerate(W_k_list)},
            **{f'W_v_{i}': w for i, w in enumerate(W_v_list)},
            'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2,
            'lm_head_weight': lm_head_weight,
            'lm_head_bias': lm_head_bias,
            }
    return logits_fn, loss_fn, params


def make_multi_head_params(n_embd, num_head, head_size):
    W_q_list = []
    W_k_list = []
    W_v_list = []
    
    scale = 1.0 / (n_embd ** 0.5)
    for _ in range(num_head):
        W_q = scaled_randn(n_embd, head_size, scale=scale)
        W_k = scaled_randn(n_embd, head_size, scale=scale)
        W_v = scaled_randn(n_embd, head_size, scale=scale)
        W_q_list.append(W_q)
        W_k_list.append(W_k)
        W_v_list.append(W_v)

    return W_q_list, W_k_list, W_v_list

def multi_head_attention(x, W_q_list, W_k_list, W_v_list, tril_mask=True):
    head_out = []

    for W_q, W_k, W_v in zip(W_q_list, W_k_list, W_v_list):
        out = self_attention(x, W_q, W_k, W_v, head_size=W_q.shape[-1], tril_mask=tril_mask)
        head_out.append(out)
    
    return torch.cat(head_out, dim=-1)


def make_ffn_params(n_embd):
    scale1 = (2.0 / n_embd) ** 0.5
    W1 = scaled_randn(n_embd, 4*n_embd, scale=scale1)
    b1 = torch.zeros(4*n_embd, requires_grad=True)

    scale2 = (2.0 / (4*n_embd)) ** 0.5
    W2 = scaled_randn(4*n_embd, n_embd, scale=scale2)
    b2 = torch.zeros(n_embd, requires_grad=True)

    return W1, b1, W2, b2


def feed_forward(x, W1, b1, W2, b2):
    x = F.relu(x @ W1 + b1)
    x = x @ W2 + b2
    return x    


def scaled_randn(*shape, scale=1.0):
    # create scaled random tensor
    # need because nn.Linear include hidden scaled
    t = torch.empty(*shape)
    t.normal_()
    t.mul_(scale)
    t.requires_grad_(True)
    return t

if __name__ == "__main__":
    torch.manual_seed(239)

    # config
    block_size = 64
    batch_size = 32
    lr = 1e-3
    count = 5000
    eval_iters = 200
    n_embd = 64
    num_head = 4
    
    # prepare data
    trans = make_translation_table()
    data, chars, encode, decode = prepare_data("data/poems.txt", trans)
    voc_size = len(chars)
     
    n = int(0.9*len(data))
    train_data, val_data = data[:n], data[n:]
    
    # model
    logits_fn, loss_fn, params = make_attention_model(voc_size, n_embd, block_size, num_head)
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
