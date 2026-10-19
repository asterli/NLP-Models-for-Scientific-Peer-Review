import os
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import sys, time, json, random, gc
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, set_seed
from common import GROUPS, LABELS, PACKAGE, ROOT, metrics, save, split
sys.path.insert(0, str(PACKAGE.parents[1] / 'src'))
from novelty import encode_pairs
torch.set_num_threads(4)
tok = AutoTokenizer.from_pretrained(os.environ['M17_ENCODER'], local_files_only=True)
all_results = {}
for seed in [544, 2026]:
    pooled = []
    for group in GROUPS:
        existing = ROOT / f'encoder_{group}_{seed}.json'
        if existing.exists():
            pooled += json.loads(existing.read_text())['records']
            continue
        start = time.perf_counter()
        set_seed(seed)
        train, test, dropped = split(group)
        model = AutoModelForSequenceClassification.from_pretrained(os.environ['M17_ENCODER'], local_files_only=True, num_labels=3, id2label=dict(enumerate(LABELS)), label2id={x: i for i, x in enumerate(LABELS)}, use_safetensors=True)
        opt = torch.optim.AdamW(model.parameters(), lr=2e-05)

        def predict(rows):
            model.eval()
            records = []
            with torch.inference_mode():
                for r in rows:
                    inputs = encode_pairs(tok, [r['claim']], [r['prior_work']], 512)
                    probs = model(**inputs).logits.softmax(-1)[0].tolist()
                    records.append({'comparison_id': r['comparison_id'], 'source_pmcid': r['source_pmcid'], 'gold': r['label'], 'predicted': LABELS[max(range(3), key=probs.__getitem__)]})
            return records
        before = predict(test)
        history = []
        rng = random.Random(seed)
        for epoch in range(1, 11):
            order = list(train)
            rng.shuffle(order)
            model.train()
            losses = []
            for offset in range(0, len(order), 4):
                batch = order[offset:offset + 4]
                x = encode_pairs(tok, [r['claim'] for r in batch], [r['prior_work'] for r in batch], 512)
                opt.zero_grad()
                loss = model(**x, labels=torch.tensor([LABELS.index(r['label']) for r in batch])).loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                opt.step()
                losses.append(loss.item())
            history.append(sum(losses) / len(losses))
        records = predict(test)
        fit = predict(train)
        pooled += records
        result = {'seed': seed, 'group': group, 'epochs': 10, 'before': metrics(before), 'after_train': metrics(fit), 'metrics': metrics(records), 'records': records, 'losses': history, 'seconds': time.perf_counter() - start}
        save(ROOT / f'encoder_{group}_{seed}.json', result)
        print(json.dumps({k: v for k, v in result.items() if k != 'records'}), flush=True)
        del model, opt
        gc.collect()
    all_results[str(seed)] = {'records': pooled, 'metrics': metrics(pooled)}
save(ROOT / 'encoder_summary.json', all_results)
