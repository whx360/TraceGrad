"""Author-sourced Fashion-MNIST loading with explicit validation-only split.

No estimator is fitted during preprocessing: uint8 pixels are divided by 255.
Downloads are separate from loading, so training never silently uses the network.
"""
from __future__ import annotations

from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import struct
import urllib.error
import urllib.request
import zlib

import numpy as np


CLASSES = (
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot',
)
FILES = {
    'train-images-idx3-ubyte.gz': '8d4fb7e6c68d591d4c3dfef9ec88bf0d',
    'train-labels-idx1-ubyte.gz': '25c81989df183df01b3e8a0aad5dffbe',
    't10k-images-idx3-ubyte.gz': 'bef4ecab320f06d8554ea6380940ec79',
    't10k-labels-idx1-ubyte.gz': 'bb300cfdad3c16e7a12a480ee83cd310',
}
AUTHOR_REPOSITORY = 'https://github.com/zalandoresearch/fashion-mnist'
SOURCE_BASES = (
    'https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/',
    'https://fashion-mnist.s3.eu-central-1.amazonaws.com/',
)


def _file_record(path: Path) -> dict:
    content = path.read_bytes()
    expected = FILES[path.name]
    actual = hashlib.md5(content).hexdigest()
    if actual != expected:
        raise ValueError(f'{path.name}: advertised MD5 mismatch ({actual} != {expected})')
    return {
        'name': path.name, 'compressed_bytes': len(content), 'md5': actual,
        'expected_author_md5': expected, 'sha256': hashlib.sha256(content).hexdigest(),
        'author_md5_matched': True,
    }


def fetch_fashion(data_dir: str | Path) -> dict:
    """Fetch/verify the four original gzip files and return saved provenance.

    Existing valid files need no network. New bytes are accepted only after the
    author's advertised MD5 matches, then atomically replace the destination.
    Time O(total compressed and decoded bytes); memory includes one compressed
    file and its decoded IDX payload during validation.
    Raises RuntimeError if all author-hosted URLs fail for a required file.
    """
    directory = Path(data_dir)
    directory.mkdir(parents=True, exist_ok=True)
    old_path = directory / 'source_manifest.json'
    previous = {}
    if old_path.exists():
        try:
            previous = json.loads(old_path.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            previous = {}
    prior_by_name = {r['name']: r for r in previous.get('files', []) if 'name' in r}
    records = []
    for name, expected in FILES.items():
        path = directory / name
        valid = False
        if path.exists():
            try:
                record = _file_record(path)
                valid = True
            except ValueError:
                pass
        if valid:
            record = {**prior_by_name.get(name, {}), **record, 'cached': True}
        else:
            errors = []
            for base in SOURCE_BASES:
                url = base + name
                try:
                    request = urllib.request.Request(
                        url, headers={'User-Agent': 'TraceGrad-research/0.1'})
                    with urllib.request.urlopen(request, timeout=60) as response:
                        content = response.read()
                        resolved = response.geturl()
                    if hashlib.md5(content).hexdigest() != expected:
                        raise ValueError('download does not match author-advertised MD5')
                    temporary = directory / (name + '.part')
                    temporary.write_bytes(content)
                    temporary.replace(path)
                    record = _file_record(path)
                    record.update({
                        'requested_url': url, 'resolved_url': resolved,
                        'downloaded_at': datetime.now(timezone.utc).isoformat(),
                        'cached': False, 'prior_attempt_failures': errors,
                    })
                    break
                except (OSError, ValueError, urllib.error.URLError) as error:
                    errors.append({'url': url, 'error': str(error)})
            else:
                raise RuntimeError(f'Cannot download {name}; author-source errors: {errors}')
        read_idx_gzip(path)
        records.append(record)
    manifest = {
        'dataset': 'Fashion-MNIST', 'official_repository': AUTHOR_REPOSITORY,
        'source_ref': 'author-hosted source; downloaded bytes checked against advertised MD5',
        'checked_at': datetime.now(timezone.utc).isoformat(),
        'files': records, 'all_passed': True,
        'identity_limit': 'MD5 matches the author table; SHA-256 identifies local bytes. Neither is a signed authorship certificate.',
    }
    if 'initial_default_network_failures' in previous:
        manifest['initial_default_network_failures'] = previous['initial_default_network_failures']
    old_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return manifest


def read_idx_gzip(path: str | Path) -> np.ndarray:
    """Read uint8 IDX image/label data, checking exact header/payload sizes.

    Returns labels shape (N,) for magic2049 or images (N,H,W) for magic2051.
    N/H/W must be positive. Trailing bytes, truncated payloads and bad gzip
    streams raise ValueError. Time and memory O(decompressed payload bytes).
    """
    path = Path(path)
    try:
        raw = gzip.decompress(path.read_bytes())
    except (OSError, EOFError, zlib.error) as error:
        if not path.exists():
            raise FileNotFoundError(f'Dataset file missing: {path}; run fetch first') from error
        raise ValueError(f'{path.name}: invalid gzip stream') from error
    if len(raw) < 8:
        raise ValueError(f'{path.name}: truncated IDX header')
    magic, count = struct.unpack('>II', raw[:8])
    if count == 0:
        raise ValueError(f'{path.name}: IDX count must be positive')
    if magic == 2049:
        offset, shape = 8, (count,)
    elif magic == 2051:
        if len(raw) < 16:
            raise ValueError(f'{path.name}: truncated image IDX header')
        rows, columns = struct.unpack('>II', raw[8:16])
        if rows == 0 or columns == 0:
            raise ValueError(f'{path.name}: image dimensions must be positive')
        offset, shape = 16, (count, rows, columns)
    else:
        raise ValueError(f'{path.name}: unsupported IDX magic {magic}')
    expected = offset
    payload_count = 1
    for dimension in shape:
        payload_count *= dimension
    expected += payload_count
    if len(raw) != expected:
        raise ValueError(f'{path.name}: IDX length mismatch ({len(raw)} != {expected})')
    return np.frombuffer(raw, dtype=np.uint8, offset=offset).reshape(shape)


def stratified_split(labels: np.ndarray, validation_per_class: int = 600,
                     seed: int = 2026) -> tuple[np.ndarray, np.ndarray]:
    """Return sorted int64 training/validation indices sampled per class.

    Uses a private default_rng(seed), leaving global RNG state unchanged.
    Each integer-labelled class must retain at least one training sample.
    Time O(C*N + N log N) for C classes (C=10 here), memory O(N) for indices.
    """
    y = np.asarray(labels)
    if y.ndim != 1 or not y.size or not np.issubdtype(y.dtype, np.integer):
        raise ValueError('labels must be a nonempty one-dimensional integer array')
    if (isinstance(validation_per_class, (bool, np.bool_))
            or not isinstance(validation_per_class, (int, np.integer))
            or validation_per_class <= 0):
        raise ValueError('validation_per_class must be a positive integer')
    rng = np.random.default_rng(seed)
    validation = []
    for label in np.unique(y):
        indices = np.flatnonzero(y == label)
        if len(indices) <= validation_per_class:
            raise ValueError(f'class {label} cannot retain training examples after validation split')
        validation.append(rng.permutation(indices)[:validation_per_class])
    val_idx = np.sort(np.concatenate(validation)).astype(np.int64)
    mask = np.ones(y.size, dtype=bool)
    mask[val_idx] = False
    return np.flatnonzero(mask).astype(np.int64), val_idx


def _indices_sha256(indices: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(indices, dtype='<i8').tobytes()).hexdigest()


def load_fashion(data_dir: str | Path, split_seed: int = 2026) -> dict:
    """Load the unchanged official test and a 54000/6000 training split.

    Returns x_train/x_val/x_test float32 matrices (N,784), integer label
    vectors, train_indices/val_indices and provenance. Pixel scaling is /255
    only. Network is never accessed. Time/memory O(total number of pixels).
    """
    directory = Path(data_dir)
    source_path = directory / 'source_manifest.json'
    source_manifest = json.loads(source_path.read_text(encoding='utf-8')) if source_path.exists() else {}
    source_by_name = {r['name']: r for r in source_manifest.get('files', []) if 'name' in r}
    records = [{**source_by_name.get(name, {}), **_file_record(directory / name)} for name in FILES]
    images = read_idx_gzip(directory / 'train-images-idx3-ubyte.gz')
    labels = read_idx_gzip(directory / 'train-labels-idx1-ubyte.gz')
    test_images = read_idx_gzip(directory / 't10k-images-idx3-ubyte.gz')
    test_labels = read_idx_gzip(directory / 't10k-labels-idx1-ubyte.gz')
    if (images.shape != (60000, 28, 28) or labels.shape != (60000,)
            or test_images.shape != (10000, 28, 28) or test_labels.shape != (10000,)):
        raise ValueError('Fashion-MNIST official count/shape mismatch')
    if not (np.all(labels < 10) and np.all(test_labels < 10)):
        raise ValueError('Fashion-MNIST labels must be in 0..9')
    train_idx, val_idx = stratified_split(labels, validation_per_class=600, seed=split_seed)

    def normalize(x):
        return x.reshape(len(x), 784).astype(np.float32) / np.float32(255.0)

    return {
        'x_train': normalize(images[train_idx]), 'y_train': labels[train_idx].astype(np.int64),
        'x_val': normalize(images[val_idx]), 'y_val': labels[val_idx].astype(np.int64),
        'x_test': normalize(test_images), 'y_test': test_labels.astype(np.int64),
        'train_indices': train_idx, 'val_indices': val_idx,
        'provenance': {
            'dataset': 'Fashion-MNIST', 'official_repository': AUTHOR_REPOSITORY,
            'files': records, 'split_seed': split_seed,
            'split': {'train': 54000, 'validation': 6000, 'test': 10000},
            'split_rule': '600 examples per class selected from official training set only',
            'train_indices_sha256': _indices_sha256(train_idx),
            'val_indices_sha256': _indices_sha256(val_idx),
            'normalization': 'uint8 -> float32 /255; no fitted statistics',
            'test_scope': 'unchanged official 10000 test images; never used to choose hyperparameters',
        },
    }
