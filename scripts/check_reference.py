"""Verify the compact, versioned experiment snapshot without downloading data."""
from pathlib import Path
import csv,hashlib,json

ROOT=Path(__file__).resolve().parents[1]

def main():
    base=ROOT/'benchmarks'/'reference'
    identity=json.loads((base/'source_identity.json').read_text(encoding='utf-8'))
    for path,record in identity['files'].items():
        if hashlib.sha256((ROOT/path).read_bytes()).hexdigest()!=record['sha256']:
            raise RuntimeError(f'Numerical source differs from recorded snapshot: {path}')
    with (base/'classification_runs.csv').open(encoding='utf-8',newline='') as f:
        runs=list(csv.DictReader(f))
    with (base/'performance_samples.csv').open(encoding='utf-8',newline='') as f:
        timings=list(csv.DictReader(f))
    assert len(runs)==9 and len(timings)==150
    outcomes=json.loads((base/'classification.json').read_text(encoding='utf-8'))
    assert outcomes['selected_variant']=='mlp'
    print(json.dumps({'status':'passed','numerical_source_files':len(identity['files']),
                      'classification_runs':len(runs),'raw_performance_rows':len(timings)}))

if __name__=='__main__':main()
