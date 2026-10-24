import tempfile
import unittest
from pathlib import Path
import numpy as np

from tracegrad import Tensor
from tracegrad.benchmarks import _eager_matmul, _paired_measure, _ce_case, _mlp_case, run_benchmarks


class BenchmarkTests(unittest.TestCase):
    def test_selective_matmul_skips_unneeded_vjp_only(self):
        for left_grad,right_grad in [(False,True),(True,False),(True,True)]:
            a=Tensor(np.arange(6,dtype=float).reshape(2,3),requires_grad=left_grad)
            b=Tensor(np.arange(12,dtype=float).reshape(3,4),requires_grad=right_grad)
            out=a@b
            contributions=out._vjp(np.ones((2,4)))
            self.assertEqual(contributions[0] is None,not left_grad)
            self.assertEqual(contributions[1] is None,not right_grad)
            out.sum().backward()
            if left_grad:np.testing.assert_array_equal(a.grad,np.ones((2,4))@b.data.T)
            if right_grad:np.testing.assert_array_equal(b.grad,a.data.T@np.ones((2,4)))

    def test_eager_and_selective_match_all_leaf_gradients(self):
        data_a=np.array([[1.,2.],[3.,4.]])
        data_b=np.array([[.3,-.2],[.7,.1]])
        outputs=[]
        for operation in [lambda a,b:a@b,_eager_matmul]:
            a=Tensor(data_a,requires_grad=True);b=Tensor(data_b,requires_grad=True)
            loss=(operation(a,b)**2).sum();loss.backward();outputs.append((loss.item(),a.grad,b.grad))
        for a,b in zip(outputs[0],outputs[1]):np.testing.assert_array_equal(a,b)

    def test_paired_order_and_counts_are_exact(self):
        calls=[]
        records=_paired_measure(lambda:calls.append('b'),lambda:calls.append('o'),repeats=3,inner=2,warmup=1)
        self.assertEqual(calls,['b','o','b','b','o','o','o','o','b','b','b','b','o','o'])
        self.assertEqual(len(records),6)
        self.assertTrue(all(row['seconds_per_call']>=0 for row in records))
        self.assertEqual([r['variant'] for r in records],['baseline','optimized','optimized','baseline','baseline','optimized'])

    def test_correctness_preflight_and_buffer_accounting(self):
        rng=np.random.default_rng(4)
        base,opt,correctness,base_graph,opt_graph=_ce_case(rng.normal(size=(7,5)),np.arange(7)%5)
        self.assertTrue(correctness['passed'])
        self.assertGreater(base_graph['nodes'],opt_graph['nodes'])
        self.assertGreater(base_graph['unique_array_bytes'],opt_graph['unique_array_bytes'])
        self.assertAlmostEqual(base(),opt(),places=12)
        base,opt,correctness,_,_=_mlp_case(rng.normal(size=(5,8)).astype(np.float32),np.arange(5)%10)
        self.assertTrue(correctness['passed'])
        self.assertAlmostEqual(base(),opt(),places=6)

    def test_formal_output_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);out=root/'occupied';out.mkdir();(out/'record.txt').write_text('preserve')
            with self.assertRaises(FileExistsError):run_benchmarks(root,out)
            self.assertEqual((out/'record.txt').read_text(),'preserve')


if __name__=='__main__':unittest.main()
