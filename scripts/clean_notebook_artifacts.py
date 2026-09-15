import json
from pathlib import Path

p = Path('notebooks/01_data_preparation.ipynb')
nb = json.loads(p.read_text(encoding='utf-8'))

cells = []
removed = []

for idx, cell in enumerate(nb['cells']):
    src = ''.join(cell.get('source', []))

    # Remove the standalone bogus code cell that is just a broken text token.
    if src.strip() == 'matrice de confusion':
        removed.append(idx)
        continue

    # Remove dead Plotly cells left behind after the final dashboard export cell.
    if 'plotly.graph_objects as go' in src or 'go.Pie' in src or 'churn_counts = df["Churn"].value_counts()' in src:
        removed.append(idx)
        continue

    if 'make_subplots(' in src and 'rows=2,' in src and 'box' in src:
        removed.append(idx)
        continue

    cells.append(cell)

nb['cells'] = cells
p.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding='utf-8')
print('removed_cells', removed)
print('remaining_cells', len(nb['cells']))
