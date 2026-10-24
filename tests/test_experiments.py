"""Pure aggregation/selection checks; no datasets, training or timing tests."""
import math
import unittest

from tracegrad.experiments import aggregate_runs, select_variant


def run(variant, seed, accuracy, validation_ce, fit_seconds=2.):
    return {'variant': variant, 'seed': seed,
            'test': {'accuracy': accuracy, 'macro_f1': accuracy - .01,
                     'cross_entropy': 1. - accuracy},
            'validation': {'cross_entropy': validation_ce},
            'fit_seconds': fit_seconds}


class AggregationTests(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(aggregate_runs([]), {})
        with self.assertRaises(ValueError):
            select_variant({})

    def test_sample_std_and_mean(self):
        result = aggregate_runs([run('mlp', 17, .8, .4), run('mlp', 29, .9, .2, 4.)])['mlp']
        self.assertEqual(result['n'], 2)
        self.assertEqual(result['seeds'], [17, 29])
        self.assertAlmostEqual(result['metrics']['test_accuracy']['mean'], .85)
        self.assertAlmostEqual(result['metrics']['test_accuracy']['std'], math.sqrt(.005))
        self.assertAlmostEqual(result['metrics']['fit_seconds']['std'], math.sqrt(2.))

    def test_one_seed_has_undefined_sample_std(self):
        result = aggregate_runs([run('linear', 17, .8, .3)])['linear']
        self.assertEqual(result['n'], 1)
        self.assertIsNone(result['metrics']['test_accuracy']['std'])

    def test_selection_ignores_test(self):
        groups = aggregate_runs([run('worse_val_better_test', 17, .99, .5),
                                 run('better_val_worse_test', 17, .7, .3)])
        self.assertEqual(select_variant(groups), 'better_val_worse_test')

    def test_ties_keep_declared_order(self):
        groups = aggregate_runs([run('first', 17, .7, .3), run('second', 17, .99, .3)])
        self.assertEqual(select_variant(groups), 'first')

    def test_duplicate_seed_is_not_counted_as_replication(self):
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            aggregate_runs([run('mlp', 17, .8, .3), run('mlp', 17, .8, .3)])

    def test_config_seed_supported(self):
        record = run('mlp', 17, .8, .3)
        record['config'] = {'seed': record.pop('seed')}
        self.assertEqual(aggregate_runs([record])['mlp']['seeds'], [17])

    def test_nonfinite_or_out_of_range_rejected(self):
        for accuracy in [float('nan'), float('inf'), -0.1, 1.1]:
            with self.subTest(accuracy=accuracy), self.assertRaises(ValueError):
                aggregate_runs([run('mlp', 17, accuracy, .3)])
        with self.assertRaises(ValueError):
            aggregate_runs([run('mlp', 17, .8, -.3)])
        with self.assertRaises(ValueError):
            aggregate_runs([run('mlp', 17, .8, .3, -1.)])

    def test_missing_metric_rejected(self):
        record = run('mlp', 17, .8, .3)
        del record['test']['macro_f1']
        with self.assertRaises(KeyError):
            aggregate_runs([record])

    def test_seed_type_rejected(self):
        for seed in [True, 17.5, '17']:
            with self.subTest(seed=seed), self.assertRaises(ValueError):
                aggregate_runs([run('mlp', seed, .8, .3)])


if __name__ == '__main__':
    unittest.main()
