import os
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import argparse, json, time, random, math
from pathlib import Path
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten
from mlx_lm import load
from mlx_lm.tuner.utils import linear_to_lora_layers, load_adapters
from mlx_lm.tuner.trainer import grad_checkpoint
from common import GROUPS, LABELS, LETTERS, ROOT, ROWS, RULES, messages, metrics, save, split, user_text
p = argparse.ArgumentParser()
p.add_argument('mode', choices=['baseline', 'train', 'verify', 'reason'])
p.add_argument('--group', choices=GROUPS)
p.add_argument('--seed', type=int, choices=[544, 2026], default=544)
p.add_argument('--adapter')
a = p.parse_args()
if a.mode == 'train' and (not a.group):
    p.error('--group is required for training')
if a.mode == 'verify' and (not a.adapter):
    p.error('--adapter is required for verification')
mx.set_cache_limit(256 * 1024 ** 2)
mx.random.seed(a.seed)
start = time.perf_counter()
model, tok = load(os.environ['M17_QWEN'])
label_ids = [tok.encode(x, add_special_tokens=False) for x in LETTERS]
assert all((len(x) == 1 for x in label_ids))
label_ids = mx.array([x[0] for x in label_ids])

def encode(r):
    ids = tok.apply_chat_template(messages(r), tokenize=True, add_generation_prompt=True)
    if len(ids) > 2048:
        raise ValueError('Input too long')
    return mx.array([ids])

def logits(model, x):
    hidden = model.model(x)[:, -1:, :]
    return model.lm_head(hidden)[:, 0, :].astype(mx.float32)

def predict(rows):
    model.eval()
    out = []
    for r in rows:
        x = encode(r)
        scores = logits(model, x)[0, label_ids]
        probs = mx.softmax(scores)
        mx.eval(probs)
        values = probs.tolist()
        out.append({'comparison_id': r['comparison_id'], 'source_pmcid': r['source_pmcid'], 'gold': r['label'], 'predicted': LABELS[max(range(3), key=values.__getitem__)], 'label_scores': values, 'tokens': x.shape[1]})
    return out

def loss_fn(model, x, y):
    return nn.losses.cross_entropy(logits(model, x), y, reduction='mean')

def add_lora():
    model.freeze()
    cfg = {'rank': 8, 'scale': 16.0, 'dropout': 0.0, 'keys': ['self_attn.q_proj', 'self_attn.v_proj']}
    linear_to_lora_layers(model, len(model.layers), cfg)
    grad_checkpoint(model.layers[0])
    return {'fine_tune_type': 'lora', 'num_layers': len(model.layers), 'lora_parameters': cfg}
print(json.dumps({'stage': 'loaded', 'seconds': time.perf_counter() - start, 'memory_gib': mx.get_active_memory() / 2 ** 30}), flush=True)
if a.mode == 'reason':
    import re
    from mlx_lm import generate
    from mlx_lm.sample_utils import make_sampler
    system = RULES.replace('Return exactly one letter: A, B, or C.', 'First compare the interventions and the main outcomes. Explain briefly whether the main contribution is already present, materially different, or impossible to assess from the supplied text. Then end with exactly FINAL: overlapping, FINAL: distinct, or FINAL: insufficient_evidence. Do not infer missing study details.')
    records = []
    for r in ROWS:
        text = user_text(r)
        prompt = tok.apply_chat_template([{'role': 'system', 'content': system}, {'role': 'user', 'content': text}], tokenize=False, add_generation_prompt=True)
        answer = generate(model, tok, prompt=prompt, max_tokens=256, sampler=make_sampler(temp=0), verbose=False)
        normalized = answer.replace('*', '').replace('`', '')
        matches = re.findall('FINAL:\\s*(overlapping|distinct|insufficient[_ ]evidence)\\b', normalized, re.I)
        label = matches[-1].lower().replace(' ', '_') if matches else 'invalid'
        records.append({'comparison_id': r['comparison_id'], 'source_pmcid': r['source_pmcid'], 'gold': r['label'], 'predicted': label, 'answer': answer})
        save(ROOT / 'qwen_reason.json', {'records': records, 'prompt': system, 'max_tokens': 256, 'temperature': 0, 'complete': len(records) == len(ROWS), 'invalid_count': sum((x['predicted'] == 'invalid' for x in records)), 'elapsed_seconds': time.perf_counter() - start})
        print(r['comparison_id'], label, flush=True)
    if any((r['predicted'] == 'invalid' for r in records)):
        print('Invalid outputs must be counted as errors in aggregation', flush=True)
    else:
        print(json.dumps(metrics(records)), flush=True)
elif a.mode == 'baseline':
    zero = predict(ROWS)
    save(ROOT / 'qwen_zero.json', {'records': zero, 'metrics': metrics(zero)})
elif a.mode == 'train':
    train, test, dropped = split(a.group)
    outdir = ROOT / f'qwen_{a.group}_{a.seed}'
    outdir.mkdir(exist_ok=False)
    cfg = add_lora()
    save(outdir / 'adapter_config.json', cfg)
    trainable = sum((v.size for _, v in tree_flatten(model.trainable_parameters())))
    lr = 1e-05
    optimizer = optim.Adam(learning_rate=lr)
    vg = nn.value_and_grad(model, loss_fn)
    ids = {r['comparison_id'] for r in train}
    before = [r for r in json.loads((ROOT / 'qwen_zero.json').read_text())['records'] if r['comparison_id'] in ids]
    assert len(before) == len(train)
    step = 0
    history = []
    rng = random.Random(a.seed)
    for epoch in range(1, 4):
        order = list(train)
        rng.shuffle(order)
        losses = []
        model.train()
        for r in order:
            x = encode(r)
            y = label_ids[mx.array([LABELS.index(r['label'])])]
            loss, grads = vg(model, x, y)
            grads, norm = optim.clip_grad_norm(grads, 1.0)
            optimizer.update(model, grads)
            mx.eval(model.trainable_parameters(), optimizer.state, loss, norm)
            value = loss.item()
            if not math.isfinite(value):
                raise RuntimeError('Nonfinite loss')
            losses.append(value)
            step += 1
            if step % 5 == 0:
                print(json.dumps({'stage': 'train', 'epoch': epoch, 'step': step, 'loss': value, 'seconds': time.perf_counter() - start, 'peak_gib': mx.get_peak_memory() / 2 ** 30}), flush=True)
        record = {'epoch': epoch, 'mean_loss': sum(losses) / len(losses)}
        history.append(record)
        print(json.dumps(record), flush=True)
    trained = predict(train)
    held = predict(test)
    mx.save_safetensors(str(outdir / 'adapters.safetensors'), dict(tree_flatten(model.trainable_parameters())))
    report = {'group': a.group, 'seed': a.seed, 'train_ids': [r['comparison_id'] for r in train], 'excluded_cross_role_ids': [r['comparison_id'] for r in dropped], 'trainable_parameters': trainable, 'history': history, 'before_train': metrics(before), 'after_train': metrics(trained), 'train_records': trained, 'records': held, 'metrics': metrics(held), 'seconds': time.perf_counter() - start, 'peak_memory_gib': mx.get_peak_memory() / 2 ** 30}
    report.update({'learning_rate': lr, 'epochs': 3})
    save(outdir / 'results.json', report)
    print(json.dumps({k: v for k, v in report.items() if k not in ['records', 'train_records']}), flush=True)
elif a.mode == 'verify':
    load_adapters(model, a.adapter)
    prev = json.loads((Path(a.adapter) / 'results.json').read_text())
    ids = {r['comparison_id'] for r in prev['records']}
    results = predict([r for r in ROWS if r['comparison_id'] in ids])
    old = {r['comparison_id']: r for r in prev['records']}
    delta = max((abs(v - w) for r in results for v, w in zip(r['label_scores'], old[r['comparison_id']]['label_scores'])))
    assert delta < 1e-05
    save(Path(a.adapter) / 'reload_check.json', {'passed': True, 'max_score_difference': delta, 'fresh_process': True, 'offline': True})
    print('Reload passed', delta)
