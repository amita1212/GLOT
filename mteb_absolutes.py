import csv
from collections import defaultdict

v = defaultdict(list)
for r in csv.DictReader(open('results/mteb_table3.csv', encoding='utf-8')):
    if r['task'] == 'mteb' and r['mteb_score']:
        v[(r['model'].split('/')[-1], r['arm'], r['mteb_task'])].append(
            float(r['mteb_score']) * 100)

ARMS = ['baseline', 'A_threshold', 'B_threshold', 'C_threshold',
        'AC_threshold', 'ABC_threshold']

for task in ['STS12', 'SciFact', 'RedditClustering', 'TwitterSemEval2015']:
    print(f'\n=== {task} : absolute scores x100 ===')
    print(f"{'arm':<12s}{'BERT':>16s}{'RoBERTa':>16s}")
    for a in ARMS:
        row = f"{a.replace('_threshold',''):<12s}"
        for m in ['bert-base-uncased', 'roberta-base']:
            x = v[(m, a, task)]
            row += f'{sum(x)/len(x):11.2f} (n{len(x):2d})' if x else f"{'--':>16s}"
        print(row)
