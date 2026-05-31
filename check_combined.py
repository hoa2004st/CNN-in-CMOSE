import json
from pathlib import Path

print("=== All transformer/ce results ===")
for p in sorted(Path('outputs/training_log').rglob('*/metrics.json')):
    if 'transformer' not in str(p) or 'semantic' in str(p):
        continue
    if 'ce' not in str(p) and 'cross_entropy' not in str(p):
        continue
    d = json.load(open(p))
    m = d['metrics']
    h = d['history']
    cfg = d['config']
    label = str(p.parent).replace('outputs/training_log/', '').replace('outputs\\training_log\\', '')
    acc = m['accuracy']
    qwk = m.get('quadratic_weighted_kappa', float('nan'))
    ep = h['best_epoch']
    sel = h.get('selection_metric', '?')
    print(f"  {label:<55} acc={acc:.4f}  qwk={qwk:.4f}  ep={ep}  sel={sel}")

print()
print("=== All training_log subfolders ===")
for p in sorted(Path('outputs/training_log').iterdir()):
    if p.is_dir():
        n_metrics = len(list(p.rglob('metrics.json')))
        print(f"  {p.name:<20} {n_metrics} metrics.json files")
