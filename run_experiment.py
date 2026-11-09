"""Run one report experiment (ablation grid / causal-mask study).

Usage (from the project root, e.g. in Colab):
    python run_experiment.py --model M2 --train T3
    python run_experiment.py --model M1 --train T1 --noncausal   # mask study
    python run_experiment.py --summary                           # print tables

Same data pipeline, loss, evaluation and seed as Lab03.ipynb. Each run keeps
the checkpoint with the lowest validation ending loss and writes
  model/ckpt_<name>.pth   and   results/<name>.json
"""
import argparse, json, math, random, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from model.transformer import build_model  # noqa: E402

MAX_LEN, IGNORE_INDEX, SEED, BATCH_SIZE = 128, -100, 42, 64
PAD_ID, BOS_ID, EOS_ID, SEP_ID = 0, 1, 2, 4

# ---- Models: only the depth differs (d_model / heads / d_ff fixed) ----------
BASE = dict(vocab_size=8000, max_len=MAX_LEN, d_model=256, n_heads=8,
            n_layers=4, d_ff=1024, dropout=0.1, pad_id=PAD_ID)
MODELS = {
    'M1': dict(BASE),                    # baseline: 4 layers
    'M2': dict(BASE, n_layers=2),        # shallower
    'M3': dict(BASE, n_layers=6),        # deeper
}
# ---- Training settings -------------------------------------------------------
TRAINS = {
    'T1': dict(objective='ending', lr=3e-4, wd=0.01, sched=None),        # baseline
    'T2': dict(objective='full',   lr=3e-4, wd=0.01, sched=None),        # + full-story loss
    'T3': dict(objective='ending', lr=1e-3, wd=0.01, sched='cosine'),    # warmup + cosine
}
WARMUP_FRAC = 0.03


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def read_jsonl(path):
    with path.open(encoding='utf-8') as f:
        return [json.loads(l) for l in f if l.strip()]


class Stories(Dataset):
    def __init__(self, rows, tok):
        self.examples = []
        for row in rows:
            sents = [tok.encode(row[f'sentence{i}'], add_special_tokens=False).ids for i in range(1, 6)]
            prompt = [BOS_ID]
            for s in sents[:4]:
                prompt += s + [SEP_ID]
            ending = sents[4] + [EOS_ID]
            if len(prompt) + len(ending) <= MAX_LEN:  # exclude, never truncate
                self.examples.append(dict(storyid=row['storyid'], input_ids=prompt + ending,
                                          ending_labels=[IGNORE_INDEX] * len(prompt) + ending))

    def __len__(self): return len(self.examples)
    def __getitem__(self, i): return self.examples[i]


def collate(exs):
    L = max(len(e['input_ids']) for e in exs)
    ids = torch.full((len(exs), L), PAD_ID, dtype=torch.long)
    lab = torch.full_like(ids, IGNORE_INDEX)
    for i, e in enumerate(exs):
        n = len(e['input_ids'])
        ids[i, :n] = torch.tensor(e['input_ids']); lab[i, :n] = torch.tensor(e['ending_labels'])
    return dict(input_ids=ids, ending_labels=lab)


def loss_sum(model, batch, objective, device):
    ids, lab = batch['input_ids'].to(device), batch['ending_labels'].to(device)
    logits = model(ids[:, :-1])
    targets, ignore = (lab[:, 1:], IGNORE_INDEX) if objective == 'ending' else (ids[:, 1:], PAD_ID)
    count = targets.ne(ignore).sum()
    s = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), targets.reshape(-1),
                        ignore_index=ignore, reduction='sum')
    return s, count


@torch.no_grad()
def evaluate(model, loader, device):  # identical to the notebook: ending tokens + EOS only
    model.eval(); tot, n = 0.0, 0
    for b in loader:
        s, c = loss_sum(model, b, 'ending', device); tot += s.item(); n += c.item()
    loss = tot / n
    return loss, math.exp(loss)


@torch.no_grad()
def generate(model, row, tok, device, max_new=40):  # greedy, same as the notebook
    model.eval()
    p = [BOS_ID]
    for i in range(1, 5):
        p += tok.encode(row[f'sentence{i}'], add_special_tokens=False).ids + [SEP_ID]
    ids = torch.tensor([p], device=device); out = []
    for _ in range(min(max_new, MAX_LEN - len(p))):
        lg = model(ids)[0, -1].clone(); lg[[PAD_ID, BOS_ID, SEP_ID]] = float('-inf')
        nid = int(lg.argmax()); out.append(nid)
        if nid == EOS_ID: break
        ids = torch.cat([ids, ids.new_tensor([[nid]])], 1)
    return tok.decode(out, skip_special_tokens=True).strip()


def summary():
    res = {p.stem: json.loads(p.read_text()) for p in sorted((ROOT / 'results').glob('*.json'))}
    cell = lambda k: (f"{res[k]['best_val_loss']:.4f} / {res[k]['best_val_ppl']:.3f}" if k in res else 'n/a')
    print('Ablation (val ending loss / PPL, best epoch by val loss)')
    print('      ' + ''.join(f'{t:>20}' for t in TRAINS))
    for m in MODELS:
        print(f'{m:<6}' + ''.join(f'{cell(f"{m}_{t}"):>20}' for t in TRAINS))
    print('Params:', {m: next((r['params'] for k, r in res.items() if k.startswith(m + '_')), 'n/a') for m in MODELS})
    print('\nCausal mask study')
    for k in ('M1_T1', 'M1_T1_noncausal'):
        if k in res:
            r = res[k]
            print(f"{k}: params {r['params']:,} | val {cell(k)} | best epoch {r['best_epoch']} | time {r['train_seconds']:.0f}s")
            for g in r['generations'][:3]:
                print('   gen:', g['generated'])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', choices=MODELS); ap.add_argument('--train', choices=TRAINS)
    ap.add_argument('--noncausal', action='store_true')
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--subset', type=int, default=0, help='smoke test: use N train stories')
    ap.add_argument('--summary', action='store_true')
    a = ap.parse_args()
    if a.summary:
        return summary()
    assert a.model and a.train, '--model and --train are required'

    name = f"{a.model}_{a.train}" + ('_noncausal' if a.noncausal else '')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tr = TRAINS[a.train]
    cfg = dict(MODELS[a.model]);
    if a.noncausal: cfg['causal'] = False

    set_seed(SEED)
    tok = Tokenizer.from_file(str(ROOT / 'model' / 'tokenizer.json'))
    train_raw = read_jsonl(ROOT / 'dataset' / 'train.jsonl')
    val_raw = read_jsonl(ROOT / 'dataset' / 'val.jsonl')
    if a.subset: train_raw, val_raw = train_raw[:a.subset], val_raw[:max(200, a.subset // 5)]
    train_set, val_set = Stories(train_raw, tok), Stories(val_raw, tok)
    gen = torch.Generator().manual_seed(SEED)
    pin = device.type == 'cuda'
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, generator=gen, collate_fn=collate, pin_memory=pin)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate, pin_memory=pin)

    set_seed(SEED)
    model = build_model(cfg).to(device).float()
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=tr['lr'], weight_decay=tr['wd'])
    sched = None
    if tr['sched'] == 'cosine':
        total = a.epochs * len(train_loader); warm = max(1, int(WARMUP_FRAC * total))
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm
                                                  else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    print(f'== {name} | device {device} | params {n_params:,} | {tr} | causal={not a.noncausal}', flush=True)

    ckpt = ROOT / 'model' / f'ckpt_{name}.pth'
    (ROOT / 'results').mkdir(exist_ok=True)
    best, best_ep, hist, t0 = float('inf'), 0, [], time.perf_counter()
    for ep in range(1, a.epochs + 1):
        model.train(); tot, cnt = 0.0, 0
        for b in train_loader:
            opt.zero_grad(set_to_none=True)
            s, c = loss_sum(model, b, tr['objective'], device)
            loss = s / c
            if not torch.isfinite(loss): raise ValueError('Non-finite training loss.')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if sched: sched.step()
            tot += s.item(); cnt += c.item()
        vl, vp = evaluate(model, val_loader, device)
        hist.append(dict(epoch=ep, train_loss=tot / cnt, val_loss=vl, val_ppl=vp))
        if vl < best:
            best, best_ep = vl, ep
            torch.save(dict(model_state_dict=model.state_dict(), config=cfg, epoch=ep,
                            val_metrics=dict(ending_loss=vl, ending_ppl=vp), training_config=dict(tr, name=name)), ckpt)
        print(f'Epoch {ep:02d}/{a.epochs} | train {tot/cnt:.4f} | val {vl:.4f} | PPL {vp:.3f}', flush=True)
    secs = time.perf_counter() - t0

    # Generations from the best checkpoint (same 5 stories as the notebook)
    best_model = build_model(torch.load(ckpt, map_location='cpu', weights_only=True)['config']).to(device).float()
    best_model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True)['model_state_dict'])
    ok = {e['storyid'] for e in val_set.examples}
    rows = random.Random(SEED).sample([r for r in val_raw if r['storyid'] in ok], k=5)
    gens = [dict(storyid=r['storyid'], context=[r[f'sentence{i}'] for i in range(1, 5)],
                 generated=generate(best_model, r, tok, device), reference=r['sentence5']) for r in rows]
    out = dict(name=name, model=a.model, train=a.train, causal=not a.noncausal, params=n_params, config=cfg,
               train_cfg=tr, epochs=a.epochs, best_epoch=best_ep, best_val_loss=best, best_val_ppl=math.exp(best),
               train_seconds=secs, history=hist, generations=gens)
    (ROOT / 'results' / f'{name}.json').write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(f'DONE {name}: best epoch {best_ep} | val loss {best:.4f} | PPL {math.exp(best):.3f} | {secs:.0f}s', flush=True)
    for g in gens[:2]: print('  gen:', g['generated'], '| ref:', g['reference'])


if __name__ == '__main__':
    main()
