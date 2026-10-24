"""Small independent IDX and splitting tests; no network or dataset required."""
import gzip
import hashlib
import io
from pathlib import Path
import struct
import tempfile
import unittest
from unittest import mock

import numpy as np

from tracegrad.data import read_idx_gzip, stratified_split, fetch_fashion, CLASSES

TEST_TEMP_ROOT = Path(__file__).resolve().parents[1] / 'artifacts' / 'data_tests'


class IDXTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.path = Path(self.directory.name) / 'fixture.gz'

    def tearDown(self):
        self.directory.cleanup()

    def write(self, raw):
        self.path.write_bytes(gzip.compress(raw))

    def test_labels(self):
        self.write(struct.pack('>II', 2049, 3) + bytes([0, 9, 4]))
        actual = read_idx_gzip(self.path)
        np.testing.assert_array_equal(actual, [0, 9, 4])
        self.assertEqual(actual.dtype, np.uint8)

    def test_images(self):
        self.write(struct.pack('>IIII', 2051, 2, 2, 3) + bytes(range(12)))
        actual = read_idx_gzip(self.path)
        self.assertEqual(actual.shape, (2, 2, 3))
        np.testing.assert_array_equal(actual.ravel(), np.arange(12))

    def test_bad_magic(self):
        self.write(struct.pack('>II', 999, 1) + b'0')
        with self.assertRaisesRegex(ValueError, 'magic'):
            read_idx_gzip(self.path)

    def test_truncated_header(self):
        for raw in [b'abc', struct.pack('>II', 2051, 1)]:
            with self.subTest(raw=raw):
                self.write(raw)
                with self.assertRaisesRegex(ValueError, 'header'):
                    read_idx_gzip(self.path)

    def test_truncated_or_trailing_payload(self):
        for suffix in [b'', b'12']:
            with self.subTest(suffix=suffix):
                self.write(struct.pack('>II', 2049, 1) + suffix)
                with self.assertRaisesRegex(ValueError, 'length'):
                    read_idx_gzip(self.path)

    def test_zero_count_or_dimension(self):
        for raw in [struct.pack('>II', 2049, 0), struct.pack('>IIII', 2051, 1, 0, 3)]:
            with self.subTest(raw=raw):
                self.write(raw)
                with self.assertRaisesRegex(ValueError, 'positive'):
                    read_idx_gzip(self.path)

    def test_bad_gzip(self):
        self.path.write_bytes(b'not gzip')
        with self.assertRaisesRegex(ValueError, 'gzip'):
            read_idx_gzip(self.path)

    def test_missing(self):
        with self.assertRaises(FileNotFoundError):
            read_idx_gzip(self.path)


class SplitTests(unittest.TestCase):
    def test_partition_and_balance(self):
        labels = np.repeat([0, 1, 2], 8)
        train, validation = stratified_split(labels, validation_per_class=2, seed=17)
        self.assertEqual(train.dtype, np.int64)
        self.assertEqual(len(train), 18)
        self.assertEqual(len(validation), 6)
        self.assertEqual(len(np.intersect1d(train, validation)), 0)
        np.testing.assert_array_equal(np.sort(np.r_[train, validation]), np.arange(24))
        np.testing.assert_array_equal(np.bincount(labels[validation]), [2, 2, 2])
        self.assertTrue(np.all(np.diff(train) > 0))
        self.assertTrue(np.all(np.diff(validation) > 0))

    def test_reproducible_private_rng(self):
        labels = np.repeat([0, 1], 20)
        np.random.seed(5)
        before = np.random.get_state()
        first = stratified_split(labels, 3, 2026)
        second = stratified_split(labels, 3, 2026)
        after = np.random.get_state()
        for left, right in zip(first, second):
            np.testing.assert_array_equal(left, right)
        np.testing.assert_array_equal(before[1], after[1])
        self.assertFalse(np.array_equal(first[1], stratified_split(labels, 3, 7)[1]))

    def test_invalid_inputs(self):
        cases = [([], 1), ([1., 2.], 1), ([[1, 2]], 1), ([1, 2, 3], 0),
                 ([1, 2, 3], True), ([1, 1, 2, 2], 2)]
        for labels, count in cases:
            with self.subTest(labels=labels, count=count):
                with self.assertRaises(ValueError):
                    stratified_split(np.asarray(labels), count)

    def test_class_names(self):
        self.assertEqual(len(CLASSES), 10)
        self.assertEqual(CLASSES[0], 'T-shirt/top')
        self.assertEqual(CLASSES[9], 'Ankle boot')


class FetchTests(unittest.TestCase):
    def setUp(self):
        # Project-local temp paths also work in Windows restricted-process tests,
        # where redirected system-Temp file replacement can be denied.
        TEST_TEMP_ROOT.mkdir(parents=True, exist_ok=True)
        self.directory = tempfile.TemporaryDirectory(dir=TEST_TEMP_ROOT)
        self.root = Path(self.directory.name)
        self.name = 'train-labels-idx1-ubyte.gz'
        self.content = gzip.compress(struct.pack('>II', 2049, 3) + bytes([0, 1, 2]), mtime=0)
        self.patcher = mock.patch.dict('tracegrad.data.FILES',
                                      {self.name: hashlib.md5(self.content).hexdigest()}, clear=True)
        self.patcher.start()

    def tearDown(self):
        self.patcher.stop()
        self.directory.cleanup()

    def response(self, content):
        response = io.BytesIO(content)
        response.geturl = lambda: 'https://author.example/test.gz'
        return response

    def test_verified_cache_avoids_network(self):
        (self.root / self.name).write_bytes(self.content)
        with mock.patch('tracegrad.data.urllib.request.urlopen') as open_url:
            manifest = fetch_fashion(self.root)
        open_url.assert_not_called()
        self.assertTrue(manifest['all_passed'])
        self.assertTrue(manifest['files'][0]['author_md5_matched'])

    def test_download_verified_before_acceptance(self):
        with mock.patch('tracegrad.data.urllib.request.urlopen', return_value=self.response(self.content)):
            manifest = fetch_fashion(self.root)
        self.assertEqual((self.root / self.name).read_bytes(), self.content)
        self.assertFalse(manifest['files'][0]['cached'])

    def test_bad_first_source_falls_back(self):
        with mock.patch('tracegrad.data.urllib.request.urlopen',
                        side_effect=[self.response(b'wrong bytes'), self.response(self.content)]):
            manifest = fetch_fashion(self.root)
        self.assertEqual(len(manifest['files'][0]['prior_attempt_failures']), 1)
        self.assertEqual((self.root / self.name).read_bytes(), self.content)

    def test_invalid_download_does_not_replace_destination(self):
        target = self.root / self.name
        target.write_bytes(b'preexisting invalid data')
        with mock.patch('tracegrad.data.urllib.request.urlopen',
                        side_effect=[self.response(b'bad1'), self.response(b'bad2')]):
            with self.assertRaises(RuntimeError):
                fetch_fashion(self.root)
        self.assertEqual(target.read_bytes(), b'preexisting invalid data')


if __name__ == '__main__':
    unittest.main()
