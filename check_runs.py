import json
from pathlib import Path

root = Path('outputs/training_log/combined/semantic_group_fusion')
nan_count = 0
total = 0
for p in sorted(root.glob('*/metrics.json')):
    d = json.loads(p.read_text())
    h = d['history']
    sel = h.get('selection_metric', '')
    m = d['metrics']
    total += 1
    if sel == '1_minus_eval_f1_macro':
        nan_count += 1
    qwk = m.get('quadratic_weighted_kappa', float('nan'))
    print(f"{p.parent.name:<28} sel={sel:<25} qwk={qwk:.4f} acc={m['accuracy']:.4f} best_ep={h['best_epoch']}")
print()
print(f"{nan_count}/{total} runs used 1_minus_eval_f1_macro as selection metric")
print()
# Also check naive model for comparison
naive = Path('outputs/training_log/combined/transformer/ce/metrics.json')
if naive.exists():
    nd = json.loads(naive.read_text())
    nm = nd['metrics']
    nh = nd['history']
    nsel = nh.get('selection_metric','')
    print(f"Naive transformer/ce  sel={nsel}  qwk={nm.get('quadratic_weighted_kappa',float('nan')):.4f}  acc={nm['accuracy']:.4f}  best_ep={nh['best_epoch']}")
