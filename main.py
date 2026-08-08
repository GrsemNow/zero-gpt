import torch
from torch.nn import functional as F
import numpy as np
# import pandas as pd
import os

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
    def sample_indices():
        return torch.randint(len(data) - block_size, (batch_size,))

    def slice_data(indices):
        x = torch.stack([data[i:i+block_size] for i in indices])
        y = torch.stack([data[i+1:i+block_size+1] for i in indices])
        return x, y

    return lambda: slice_data(sample_indices())


def make_bigram_model(voc_size):
    embedding_weight = torch.randn(voc_size, voc_size, requires_grad=True)
    
    def logits_fn(idx):
        return F.embedding(idx, embedding_weight)

    def loss_fn(logits, targets):
        B, T, C = logits.shape
        return F.cross_entropy(logits.view(B*T, C), targets.view(B*T))
        
    return logits_fn, loss_fn, embedding_weight


def generate_step(logits_fn, idx, count, block_size, temperature=1.0, device='cpu'):
    idx = idx.to(device)
    sequence = idx
    for _ in range(count):
        idx_cond = sequence[:, -block_size:] if sequence.shape[1] > block_size else sequence
        logits = logits_fn(idx_cond, training=False)[:, -1, :] / temperature
        probs = F.softmax(logits, dim=1)
        idx_next = torch.multinomial(probs, num_samples=1)
        sequence = torch.cat((sequence, idx_next), dim=1)
        yield sequence


def train(optimizer, logits_fn, loss_fn, get_batch, count, device):
    for step in range(count):
        x, y = get_batch()
        x, y = x.to(device), y.to(device)
        
        logits = logits_fn(x, training=True)
        loss = loss_fn(logits, y)
        
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        
        yield loss.item()


def clear_text(text, dct, lit_title=True, splitter="---"):
    cnt_upper = lambda x: x.lower() if sum(1 for ch in x if ch.isupper()) > 1 and lit_title else x
    proc_line = lambda line: ' '.join(map(cnt_upper, line.split()))
    proc_text = lambda text: '\n'.join(map(proc_line, text.split('\n')))
    return '\n'.join(proc_text(text).translate(dct).split(splitter))


def coders(chars):
    stoi = { ch:i for i,ch in enumerate(chars) }
    itos = { i:ch for i,ch in enumerate(chars) }
    encoder = lambda s: [stoi[ch] for ch in s]
    decoder = lambda q: ''.join([itos[i] for i in q])
    return encoder, decoder


@torch.no_grad()
def estimate_loss(logits_fn, loss_fn, train_batch, val_batch, eval_iters, device):
    loss_train = 0.0
    for _ in range(eval_iters):
        x, y = train_batch()
        x, y = x.to(device), y.to(device)
        logits = logits_fn(x, training=False)
        loss = loss_fn(logits, y)
        loss_train += loss.item()

    loss_val = 0.0
    for _ in range(eval_iters):
        x, y = val_batch()
        x, y = x.to(device), y.to(device)
        logits = logits_fn(x, training=False)
        loss = loss_fn(logits, y)
        loss_val += loss.item()
    
    return loss_train / eval_iters, loss_val / eval_iters


def self_attention(x, W_q, W_k, W_v, head_size, tril_mask=True):
    B, T, C = x.shape
    q = x @ W_q
    k = x @ W_k
    v = x @ W_v
    wei = q @ k.transpose(-2, -1) / (head_size ** 0.5)
    if tril_mask:
        tril = torch.tril(torch.ones(T, T, device=x.device))
        wei = wei.masked_fill(tril == 0, float('-inf'))
    wei = F.softmax(wei, dim=-1)
    out = wei @ v
    return out


def scaled_randn(*shape, scale=1.0, device='cpu'):
    t = torch.empty(*shape, device=device)
    t.normal_()
    t.mul_(scale)
    t.requires_grad_(True)
    return t

def make_multi_head_params(n_embd, num_head, head_size, device='cpu'):
    W_q_list, W_k_list, W_v_list = [], [], []
    scale = 1.0 / (n_embd ** 0.5)
    for _ in range(num_head):
        W_q = scaled_randn(n_embd, head_size, scale=scale, device=device)
        W_k = scaled_randn(n_embd, head_size, scale=scale, device=device)
        W_v = scaled_randn(n_embd, head_size, scale=scale, device=device)
        W_q_list.append(W_q)
        W_k_list.append(W_k)
        W_v_list.append(W_v)
    return W_q_list, W_k_list, W_v_list

def make_ffn_params(n_embd, device='cpu'):
    scale1 = (2.0 / n_embd) ** 0.5
    W1 = scaled_randn(n_embd, 4*n_embd, scale=scale1, device=device)
    b1 = torch.zeros(4*n_embd, requires_grad=True, device=device)
    scale2 = (2.0 / (4*n_embd)) ** 0.5
    W2 = scaled_randn(4*n_embd, n_embd, scale=scale2, device=device)
    b2 = torch.zeros(n_embd, requires_grad=True, device=device)
    return W1, b1, W2, b2

def make_block_params(n_embd, num_head, device='cpu'):
    head_size = n_embd // num_head
    scale = 1.0 / (n_embd ** 0.5)
    W_q_list, W_k_list, W_v_list = make_multi_head_params(n_embd, num_head, head_size, device)
    W_proj = scaled_randn(n_embd, n_embd, scale=scale, device=device)
    W1, b1, W2, b2 = make_ffn_params(n_embd, device)
    ln1_g = torch.ones(n_embd, requires_grad=True, device=device)
    ln1_b = torch.zeros(n_embd, requires_grad=True, device=device)
    ln2_g = torch.ones(n_embd, requires_grad=True, device=device)
    ln2_b = torch.zeros(n_embd, requires_grad=True, device=device)
    return {
        'W_q_list': W_q_list, 'W_k_list': W_k_list, 'W_v_list': W_v_list,
        'W_proj': W_proj,
        'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2,
        'ln1_g': ln1_g, 'ln1_b': ln1_b,
        'ln2_g': ln2_g, 'ln2_b': ln2_b,
    }

def make_attention_model(voc_size, n_embd, block_size, num_head, num_layers, dropout=0.0, device='cpu'):
    embedding_weight = torch.randn(voc_size, n_embd, requires_grad=True, device=device)
    position_embedding_table = torch.randn(block_size, n_embd, requires_grad=True, device=device)
    
    all_params = {}
    for i in range(num_layers):
        block_params = make_block_params(n_embd, num_head, device)
        for key, value in block_params.items():
            if isinstance(value, list):
                for j, v in enumerate(value):
                    all_params[f'{key}_{i}_{j}'] = v
            else:
                all_params[f'{key}_{i}'] = value
    
    all_params['embedding_weight'] = embedding_weight
    all_params['position_embedding_table'] = position_embedding_table
    
    scale = 1.0 / (n_embd ** 0.5)
    lm_head_weight = scaled_randn(n_embd, voc_size, scale=scale, device=device)
    lm_head_bias = torch.zeros(voc_size, requires_grad=True, device=device)
    all_params['lm_head_weight'] = lm_head_weight
    all_params['lm_head_bias'] = lm_head_bias

    final_ln_g = torch.ones(n_embd, requires_grad=True, device=device)
    final_ln_b = torch.zeros(n_embd, requires_grad=True, device=device)
    all_params['final_ln_g'] = final_ln_g
    all_params['final_ln_b'] = final_ln_b

    def logits_fn(idx, training=True):
        B, T = idx.shape
        tok_emb = F.embedding(idx, embedding_weight)
        pos_emb = F.embedding(torch.arange(T, device=device), position_embedding_table)
        x = tok_emb + pos_emb
        
        for i in range(num_layers):
            W_q_list = [all_params[f'W_q_list_{i}_{j}'] for j in range(num_head)]
            W_k_list = [all_params[f'W_k_list_{i}_{j}'] for j in range(num_head)]
            W_v_list = [all_params[f'W_v_list_{i}_{j}'] for j in range(num_head)]
            W_proj = all_params[f'W_proj_{i}']
            W1 = all_params[f'W1_{i}']; b1 = all_params[f'b1_{i}']
            W2 = all_params[f'W2_{i}']; b2 = all_params[f'b2_{i}']
            ln1_g = all_params[f'ln1_g_{i}']; ln1_b = all_params[f'ln1_b_{i}']
            ln2_g = all_params[f'ln2_g_{i}']; ln2_b = all_params[f'ln2_b_{i}']

            x_prev = x
            x = layer_norm(x, ln1_g, ln1_b)
            x = multi_head_attention(x, W_q_list, W_k_list, W_v_list, W_proj, dropout=dropout, tril_mask=True, training=training)
            x = x + x_prev
        
            x_prev = x
            x = layer_norm(x, ln2_g, ln2_b)
            x = feed_forward(x, W1, b1, W2, b2, dropout=dropout, training=training)
            x = x + x_prev
        
        x = layer_norm(x, final_ln_g, final_ln_b)
        logits = x @ lm_head_weight + lm_head_bias
        return logits

    def loss_fn(logits, targets):
        B, T, C = logits.shape
        return F.cross_entropy(logits.view(B*T, C), targets.view(B*T))
    
    return logits_fn, loss_fn, all_params

def multi_head_attention(x, W_q_list, W_k_list, W_v_list, W_proj, dropout=0.0, tril_mask=True, training=True):
    head_out = []
    for W_q, W_k, W_v in zip(W_q_list, W_k_list, W_v_list):
        out = self_attention(x, W_q, W_k, W_v, head_size=W_q.shape[-1], tril_mask=tril_mask)
        head_out.append(out)
    out = torch.cat(head_out, dim=-1)
    out = out @ W_proj
    out = F.dropout(out, p=dropout, training=training)
    return out

def feed_forward(x, W1, b1, W2, b2, dropout=0.0, training=True):
    x = F.relu(x @ W1 + b1)
    x = x @ W2 + b2
    x = F.dropout(x, p=dropout, training=training)
    return x

def layer_norm(x, gamma, beta, eps=1e-5):
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    x_norm = (x - mean) / torch.sqrt(var + eps)
    return x_norm * gamma + beta

def train_base(main_config, train_config, data_path, device='cpu', trans=None, save_dir='saves'):
    os.makedirs(save_dir, exist_ok=True)

    # unpacking main_config
    block_size = main_config['block_size']
    batch_size = main_config['batch_size']
    eval_iters = main_config['eval_iters']
    n_embd = main_config['n_embd']
    num_head = main_config['num_head']
    num_layers = main_config['num_layers']
    
    # unpacking train_config
    lr = train_config['lr']
    count = train_config['count']
    patience = train_config['patience']
    check_int = train_config['check_int']
    dropout = train_config['dropout']

    # prepare data
    data, chars, encode, decode = prepare_data(data_path, trans)
    voc_size = len(chars)
    
    n = int(0.9*len(data))
    train_data, val_data = data[:n], data[n:]

    # model
    logits_fn, loss_fn, params = make_attention_model(
            voc_size, n_embd, block_size, num_head, num_layers,
            dropout=dropout, device=device)
    optimizer = torch.optim.AdamW(list(params.values()), lr=lr) 
    
    # training
    train_batch = make_batch(train_data, block_size, batch_size)
    val_batch = make_batch(val_data, block_size, batch_size)
    tr = train(optimizer, logits_fn, loss_fn, train_batch, count, device)
    
    # data for curves
    log_steps = []
    log_train_loss = []
    log_val_loss = []
    log_perplexity = []
    log_lr = []
    
    # train cicle
    best_val_loss = float('inf')
    wait = 0

    for step, loss_value in enumerate(tr, 1):
        if step % 500 == 0:
            train_loss, val_loss = estimate_loss(
                logits_fn, loss_fn, train_batch, val_batch, eval_iters, device
            )
            print(f"{step}: train={train_loss:.4f}, val={val_loss:.4f}")
            log_steps.append(step)
            log_train_loss.append(train_loss)
            log_val_loss.append(val_loss)
            log_perplexity.append(np.exp(val_loss))
            log_lr.append(optimizer.param_groups[0]['lr'])
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                wait = 0
                torch.save({
                    'step': step,
                    'params': params,
                    'best_val_loss': best_val_loss,
                    'voc_size': voc_size,
                    'chars': chars,
                }, f'{save_dir}/best_model.pt')
                print(f"val_loss={best_val_loss:.4f} save")
            else:
                wait += 500
                if wait >= patience:
                    print(f"STOP on {step}")
                    break

        if step % check_int == 0:
            torch.save({
                'step': step,
                'params': params,
                'train_loss': train_loss if 'train_loss' in locals() else None,
                'val_loss': val_loss if 'val_loss' in locals() else None,
            }, f'{save_dir}/checkpoint_{step}.pt')
            print(f"check saves on {step}")
  
    # save logs
    log_data = {
        'steps': log_steps,
        'train_loss': log_train_loss,
        'val_loss': log_val_loss,
        'perplexity': log_perplexity,
        'learning_rate': log_lr,
    }
    torch.save(log_data, f'{save_dir}/train_logs.pt')

    # generation
    idx = torch.zeros((1,1), dtype=torch.long, device=device)
    gen = generate_step(logits_fn, idx, count=200, block_size=block_size, device=device)
    for seq in gen:
        pass
    gen_text = decode(seq[0].tolist())
    print(gen_text)

    with open(f'{save_dir}/gen_text.txt', 'w', encoding='utf-8') as f:
        f.write(gen_text)
    
def tune_fine(main_config, tune_config, model_path, data_path, device='cpu', save_dir='saves/fine'):
    os.makedirs(save_dir, exist_ok=True)

    # unpacking main_config
    block_size = main_config['block_size']
    batch_size = main_config['batch_size']
    eval_iters = main_config['eval_iters']
    n_embd = main_config['n_embd']
    num_head = main_config['num_head']
    num_layers = main_config['num_layers']
    
    # unpacking train_config
    lr = tune_config['lr']
    count = tune_config['count']
    patience = tune_config['patience']
    check_int = tune_config['check_int']
    dropout = tune_config['dropout']

    # load model
    model = torch.load(model_path, map_location=device)
    params = model['params']
    voc_size = model['voc_size']
    chars = model['chars']
    encode, decode = coders(chars)

    # load data
    with open(data_path, 'r', encoding='utf-8') as f:
        text = f.read()
    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    train_data, val_data = data[:n], data[n:]

    # make model
    logits_fn, loss_fn, new_params = make_attention_model(
        voc_size, n_embd, block_size, num_head, num_layers,
        dropout=dropout, device=device
    )
    
    # copy params
    for key, value in params.items():
        new_params[key].data.copy_(value.data)
    
    # optimizer
    optimizer = torch.optim.AdamW(list(new_params.values()), lr=lr)

    # data for curves
    log_steps = []
    log_train_loss = []
    log_val_loss = []
    log_perplexity = []
    log_lr = []
    
    # training
    train_batch = make_batch(train_data, block_size, batch_size)
    val_batch = make_batch(val_data, block_size, batch_size)
    tr = train(optimizer, logits_fn, loss_fn, train_batch, count, device)

    # tune cicle
    best_val_loss = float('inf')
    wait = 0

    for step, loss_value in enumerate(tr, 1):
        if step % 500 == 0:
            train_loss, val_loss = estimate_loss(
                logits_fn, loss_fn, train_batch, val_batch, eval_iters, device
            )
            print(f"{step}: train={train_loss:.4f}, val={val_loss:.4f}")
            log_steps.append(step)
            log_train_loss.append(train_loss)
            log_val_loss.append(val_loss)
            log_perplexity.append(np.exp(val_loss))
            log_lr.append(optimizer.param_groups[0]['lr'])

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                wait = 0
                torch.save({
                    'step': step,
                    'params': new_params,
                    'best_val_loss': best_val_loss,
                    'voc_size': voc_size,
                    'chars': chars,
                }, f'{save_dir}/best_tuned.pt')
                 print(f"val_loss={best_val_loss:.4f} save")
            else:
                wait += 500
                if wait >= patience:
                    print(f"STOP on {step}")
                    break

        if step % check_int == 0:
            torch.save({
                'step': step,
                'params': new_params,
                'train_loss': train_loss if 'train_loss' in locals() else None,
                'val_loss': val_loss if 'val_loss' in locals() else None,
            }, f'{save_dir}/checkpoint_{step}.pt')
            print(f"check saves on {step}") 

    # save logs
    log_data = {
        'steps': log_steps,
        'train_loss': log_train_loss,
        'val_loss': log_val_loss,
        'perplexity': log_perplexity,
        'learning_rate': log_lr,
    }
    torch.save(log_data, f'{save_dir}/tune_logs.pt')

    # generation
    idx = torch.zeros((1,1), dtype=torch.long, device=device)
    gen = generate_step(logits_fn, idx, count=4100, block_size=block_size, device=device)
    for seq in gen:
        pass
    gen_text = decode(seq[0].tolist())
    print(gen_text)

    with open(f'{save_dir}/gen_text_tune.txt', 'w', encoding='utf-8') as f:
        f.write(gen_text)





if __name__ == "__main__":
    # system params for declarativity
    torch.manual_seed(239)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # 'cpu' is base
    print(device)

    main_config = {
            'block_size': 128,
            'batch_size': 64,
            'eval_iters': 200,
            'n_embd': 256,
            'num_head': 4,
            'num_layers': 6
            }
    
    train_config = {
            'patience': 5000,
            'check_int': 2000,
            'dropout': 0.1,
            'lr': 1e-4,
            'count': 80000
            }

    tune_config = {
            'patience': 1000,
            'check_int': 1000,
            'dropout': 0.2,
            'lr': 1e-5,
            'count': 10000
            }
    
    trans = make_translation_table()

    # base training
    train_base(main_config, train_config, 'data/poems.txt', trans=trans, device=device, save_dir='saves')
    
    # fine tune (clean dataset)
    tune_fine(
            main_config=main_config,
            tune_config=tune_config, 
            model_path='saves/best_model.pt', 
            data_path='data/boris.txt', 
            device=device, 
            save_dir='saves/fine'
            )
