"""Fixed, reproducible training campaigns and descriptive seed aggregation.

The campaign selects models using validation CE only. Raw test outcomes remain
visible for every declared configuration and never influence selection.
"""
from __future__ import annotations

from contextlib import contextmanager
import csv
from datetime import datetime, timezone, timedelta
import hashlib
import importlib.metadata
import json
import math
import numbers
import os
from pathlib import Path
import platform
import re
import sys

import numpy as np
from threadpoolctl import threadpool_info, threadpool_limits

from .data import load_fashion
from .training import train_model


METRIC_PATHS = {
    'test_accuracy': ('test', 'accuracy'),
    'test_macro_f1': ('test', 'macro_f1'),
    'test_cross_entropy': ('test', 'cross_entropy'),
    'validation_cross_entropy': ('validation', 'cross_entropy'),
    'fit_seconds': ('fit_seconds',),
}


def _now() -> str:
    return datetime.now(timezone(timedelta(hours=8))).isoformat()


def _write(path: Path, payload) -> None:
    """Replace complete JSON rather than leaving a partial campaign on interruption."""
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2,
                                    allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def aggregate_runs(runs: list[dict]) -> dict:
    """Group runs by variant and return mean/sample-std of five declared metrics.

    n=1 has std=None, since sample standard deviation is undefined. Duplicate
    (variant,seed), missing metrics, nonfinite values or negative times are
    rejected. Time O(number of runs), memory O(number of runs).
    """
    grouped = {}
    seen = set()
    for run in runs:
        variant = run['variant']
        if not isinstance(variant, str) or not variant:
            raise ValueError('run variant must be a nonempty string')
        seed = run['seed'] if 'seed' in run else run['config']['seed']
        if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, numbers.Integral):
            raise ValueError('run seed must be an integer')
        pair = (variant, int(seed))
        if pair in seen:
            raise ValueError(f'duplicate variant/seed record: {pair}')
        seen.add(pair)
        item = grouped.setdefault(variant, {'seeds': [], 'values': {k: [] for k in METRIC_PATHS}})
        item['seeds'].append(int(seed))
        for name, path in METRIC_PATHS.items():
            value = run
            for key in path:
                value = value[key]
            value = float(value)
            if not math.isfinite(value):
                raise ValueError(f'nonfinite {name} for {pair}')
            if name.endswith('accuracy') or name.endswith('macro_f1'):
                if not 0 <= value <= 1:
                    raise ValueError(f'{name} must be in [0,1]')
            elif value < 0:
                raise ValueError(f'{name} must be nonnegative')
            item['values'][name].append(value)
    aggregate = {}
    for variant, item in grouped.items():
        count = len(item['seeds'])
        aggregate[variant] = {
            'n': count,
            'seeds': sorted(item['seeds']),
            'metrics': {name: {
                'mean': float(np.mean(values)),
                'std': float(np.std(values, ddof=1)) if count > 1 else None,
            } for name, values in item['values'].items()},
            'uncertainty_scope': 'descriptive sample std over seeds; not a confidence interval or significance test',
        }
    return aggregate


def select_variant(aggregates: dict) -> str:
    """Choose minimum mean selected validation CE; ties keep declared run order."""
    if not aggregates:
        raise ValueError('cannot select a variant from no completed runs')
    values = {name: float(group['metrics']['validation_cross_entropy']['mean'])
              for name, group in aggregates.items()}
    if any(not math.isfinite(value) or value < 0 for value in values.values()):
        raise ValueError('validation cross-entropy means must be finite and nonnegative')
    return min(values, key=values.get)


def _version(distribution: str):
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _environment() -> dict:
    cpu_name = platform.processor()
    if sys.platform == 'win32':
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                r'HARDWARE\DESCRIPTION\System\CentralProcessor\0') as key:
                cpu_name = str(winreg.QueryValueEx(key, 'ProcessorNameString')[0]).strip()
        except OSError:
            pass
    ram = None
    try:
        import psutil
        ram = psutil.virtual_memory().total
    except ImportError:
        pass
    return {
        'checked_at': _now(), 'python': sys.version, 'python_executable': sys.executable,
        'platform': platform.platform(), 'cpu': cpu_name,
        'logical_processors': os.cpu_count(), 'ram_bytes': ram,
        'versions': {name: _version(name) for name in
                     ('numpy', 'torch', 'threadpoolctl', 'psutil', 'matplotlib')},
        'thread_environment': {name: os.environ.get(name) for name in
                               ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS')},
        'threadpoolctl': threadpool_info(),
        'control': 'threadpool_limits(limits=1,user_api=blas) throughout campaign',
        'device': 'CPU; no GPU training or performance claim',
    }


def _source_identity(root: Path) -> dict:
    paths = sorted((root / 'src').rglob('*.py')) + [root / 'configs' / 'formal.json']
    records = {}
    for path in paths:
        content = path.read_bytes()
        records[path.relative_to(root).as_posix()] = {
            'bytes': len(content), 'sha256': hashlib.sha256(content).hexdigest(),
        }
    return {'captured_at': _now(), 'scope': 'project src/*.py recursively and frozen formal configuration',
            'files': records}


def _output_path(root: Path, output, quick: bool) -> Path:
    if output is None:
        if quick:
            timestamp = datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d_%H%M%S_%f')
            target = root / 'artifacts' / ('experiment_smoke_' + timestamp)
        else:
            target = root / 'results' / 'formal_v1'
    else:
        target = Path(output)
        if not target.is_absolute():
            target = root / target
    target = target.resolve()
    if not target.is_relative_to(root):
        raise ValueError('campaign output must remain inside the TraceGrad project')
    if target.exists() and (not target.is_dir() or any(target.iterdir())):
        raise FileExistsError(f'refusing to overwrite nonempty campaign output: {target}')
    if quick and target.is_relative_to(root / 'results'):
        raise ValueError('smoke-only experiments must not be placed under formal results/')
    return target


def _summary(path: Path, campaign: dict) -> None:
    fields = ['protocol', 'smoke_only', 'variant', 'seed', 'best_epoch',
              'train_cross_entropy', 'validation_cross_entropy', 'validation_accuracy',
              'test_cross_entropy', 'test_accuracy', 'test_macro_f1',
              'fit_seconds', 'wall_seconds', 'parameter_count', 'directory']
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for run in campaign['runs']:
            writer.writerow({
                'protocol': campaign['protocol'], 'smoke_only': campaign['smoke_only'],
                'variant': run['variant'], 'seed': run['seed'], 'best_epoch': run['best_epoch'],
                'train_cross_entropy': run['train']['cross_entropy'],
                'validation_cross_entropy': run['validation']['cross_entropy'],
                'validation_accuracy': run['validation']['accuracy'],
                'test_cross_entropy': run['test']['cross_entropy'],
                'test_accuracy': run['test']['accuracy'], 'test_macro_f1': run['test']['macro_f1'],
                'fit_seconds': run['fit_seconds'], 'wall_seconds': run['wall_seconds'],
                'parameter_count': run['parameter_count'], 'directory': run['directory'],
            })


def run_campaign(root: str | Path, quick: bool = False, output=None) -> Path:
    """Run the predeclared variants/seeds, incrementally persisting raw evidence.

    Formal uses unchanged54000/6000/10000 splits,3variants×3seeds,8epochs.
    Quick uses deterministic2000/1000/1000 subsets,1seed,2epochs and an explicit
    smoke-only directory. Output is campaign.json. Existing nonempty output is
    rejected before data loading or training. Sequential runtimes across model
    architectures are descriptive costs, not equal-work speedup measurements.
    """
    root = Path(root).resolve()
    target = _output_path(root, output, quick)
    config = json.loads((root / 'configs' / 'formal.json').read_text(encoding='utf-8'))
    variants = config['variants']
    names = [variant['name'] for variant in variants]
    if (not names or len(names) != len(set(names))
            or any(not isinstance(name, str) or re.fullmatch(r'[A-Za-z][A-Za-z0-9_-]*', name) is None
                   for name in names)):
        raise ValueError('variant names must be unique, nonempty simple identifiers')
    effective = dict(config)
    if quick:
        effective.update(protocol='experiment_smoke', epochs=2, seeds=[17])
    seeds = effective['seeds']
    if (not seeds or len(seeds) != len(set(seeds))
            or any(isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in seeds)):
        raise ValueError('seeds must be unique nonnegative integers')
    target.mkdir(parents=True, exist_ok=True)
    campaign_path = target / 'campaign.json'
    campaign = {
        'format': 'tracegrad_campaign_v1', 'status': 'running',
        'protocol': effective['protocol'], 'smoke_only': bool(quick),
        'started_at': _now(), 'expected_runs': len(variants) * len(seeds),
        'configuration': effective, 'runs': [], 'aggregates': {}, 'selected_variant': None,
        'selection': 'minimum mean selected validation cross-entropy across seeds; test never used',
        'test_policy': effective['test_policy'],
        'timing_scope': 'fit_seconds excludes validation; architecture runtimes descriptive, not equal-work speedups',
        'std_scope': 'sample standard deviation ddof1; n1 undefined and stored null; no significance claim',
    }
    _write(target / 'config.json', effective)
    _write(target / 'code_identity.json', _source_identity(root))
    _write(campaign_path, campaign)
    run_name = None
    try:
        with threadpool_limits(limits=1, user_api='blas'):
            _write(target / 'environment.json', _environment())
            data = load_fashion(root / 'data' / 'raw' / 'fashion-mnist',
                                split_seed=int(effective['split_seed']))
            provenance = dict(data['provenance'])
            if quick:
                for split, limit in [('train', 2000), ('val', 1000), ('test', 1000)]:
                    data['x_' + split] = data['x_' + split][:limit]
                    data['y_' + split] = data['y_' + split][:limit]
                provenance['smoke_only'] = True
                provenance['smoke_subset'] = {'train': 2000, 'validation': 1000, 'test': 1000}
                provenance['smoke_subset_rule'] = 'first rows of each independently fixed official-protocol split'
                data['provenance'] = provenance
            _write(target / 'data_provenance.json', provenance)
            common = {key: effective[key] for key in
                      ('protocol', 'epochs', 'batch_size', 'lr', 'optimizer', 'dtype', 'classes')}
            for variant in variants:
                for seed in seeds:
                    run_name = f"{variant['name']}_seed{seed}"
                    run_path = target / run_name
                    run_config = {**common, 'seed': seed,
                                  'hidden_dim': variant['hidden_dim'], 'l2': variant['l2'],
                                  'smoke_only': bool(quick)}
                    print(f"run={run_name} protocol={campaign['protocol']} smoke_only={quick}", flush=True)
                    result = train_model(data, run_config, run_path)
                    record = {**result, 'variant': variant['name'], 'seed': seed,
                              'directory': run_path.relative_to(root).as_posix()}
                    campaign['runs'].append(record)
                    campaign['aggregates'] = aggregate_runs(campaign['runs'])
                    campaign['completed_runs'] = len(campaign['runs'])
                    campaign['updated_at'] = _now()
                    _write(target / 'aggregate.json', campaign['aggregates'])
                    _summary(target / 'summary.csv', campaign)
                    _write(campaign_path, campaign)
            campaign['selected_variant'] = select_variant(campaign['aggregates'])
            campaign['status'] = 'completed'
            campaign['finished_at'] = _now()
            _write(campaign_path, campaign)
    except Exception as error:
        campaign.update(status='failed', failed_run=run_name, failed_at=_now(),
                        error={'type': type(error).__name__, 'message': str(error)})
        _write(campaign_path, campaign)
        raise
    return campaign_path


def run_benchmark(root: str | Path, output=None):
    """Separate opt-in benchmark entry; never implicitly added to a ML campaign."""
    from .benchmarks import run_benchmarks
    return run_benchmarks(Path(root).resolve(), output=output)
