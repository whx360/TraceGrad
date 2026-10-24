"""Independent scalar examples, finite differences and graph-contract regression."""
import unittest
import numpy as np
from tracegrad.tensor import Tensor, cross_entropy, cross_entropy_composed, graph_statistics, no_grad


class TensorTests(unittest.TestCase):
    def close(self, actual, expected, atol=1e-8, rtol=1e-7):
        np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol)

    def gradcheck(self, function, arrays, atol=2e-6, rtol=2e-5):
        tensors = [Tensor(a, requires_grad=True) for a in arrays]
        output = function(*tensors)
        self.assertEqual(output.shape, ())
        output.backward()
        for index, array in enumerate(arrays):
            numerical = np.empty_like(array, dtype=np.float64)
            for coordinate in np.ndindex(array.shape):
                plus = [np.array(a, dtype=np.float64, copy=True) for a in arrays]
                minus = [np.array(a, dtype=np.float64, copy=True) for a in arrays]
                plus[index][coordinate] += 1e-6
                minus[index][coordinate] -= 1e-6
                fplus = function(*[Tensor(a) for a in plus]).item()
                fminus = function(*[Tensor(a) for a in minus]).item()
                numerical[coordinate] = (fplus - fminus) / 2e-6
            self.close(tensors[index].grad, numerical, atol=atol, rtol=rtol)

    def test_hand_scalar_chain(self):
        x = Tensor(3., requires_grad=True)
        y = x*x + 2*x + 1
        self.assertEqual(y.item(), 16.)
        y.backward()
        self.close(x.grad, 8.)

    def test_shared_node_and_two_paths(self):
        x = Tensor(2., requires_grad=True)
        z = x*x
        y = z*z + z
        y.backward()
        self.close(x.grad, 36.)

    def test_leaf_accumulation_and_zero(self):
        x = Tensor(2., requires_grad=True)
        (3*x).backward()
        (4*x).backward()
        self.close(x.grad, 7.)
        x.zero_grad()
        self.assertTrue(x.grad is None or np.all(x.grad == 0))
        (5*x).backward()
        self.close(x.grad, 5.)

    def test_retain_graph_fresh_intermediates(self):
        x = Tensor(2., requires_grad=True)
        z = x*x
        y = z*z
        y.backward(retain_graph=True)
        self.close(x.grad, 32.)
        y.backward()
        self.close(x.grad, 64.)
        with self.assertRaises(RuntimeError):
            y.backward()

    def test_seed_for_vector_output(self):
        x = Tensor([1.,2.,3.], requires_grad=True)
        y = x*x
        with self.assertRaises((ValueError, RuntimeError)):
            y.backward()
        y.backward(np.array([1.,2.,4.]))
        self.close(x.grad, [2.,8.,24.])

    def test_invalid_seed_shape(self):
        x = Tensor([1.,2.], requires_grad=True)
        with self.assertRaises(ValueError):
            (x*x).backward(np.ones((2,1)))

    def test_elementwise_gradients(self):
        a = np.array([[.7,1.2],[-.3,2.]])
        b = np.array([[1.1,.8],[1.7,2.2]])
        for fn in [lambda x,y:(x+y).sum(),lambda x,y:(x-y).sum(),
                   lambda x,y:(x*y).sum(),lambda x,y:(x/y).sum(),
                   lambda x,y:(2/x-y/3+4*x).mean()]:
            with self.subTest(fn=fn):
                self.gradcheck(fn,[a,b])

    def test_quotient_extreme_finite_gradients(self):
        for a,b,expected in [(1e200,1e200,-1e-200),(1e-300,1e-200,-1e100)]:
            x=Tensor(a,requires_grad=True)
            y=Tensor(b,requires_grad=True)
            (x/y).backward()
            self.close(y.grad,expected,atol=0,rtol=1e-12)

    def test_composed_ce_large_common_offset(self):
        for dtype,offset in [(np.float64,1e16),(np.float32,1e8)]:
            x=Tensor(np.array([[offset,offset]],dtype=dtype),requires_grad=True)
            loss=cross_entropy_composed(x,np.array([0]))
            self.close(loss.data,np.log(2),atol=1e-7)
            loss.backward()
            self.close(x.grad,[[-.5,.5]])

    def test_broadcast_missing_and_singleton_axes(self):
        a = np.arange(12,dtype=float).reshape(2,2,3)/10+.2
        for shape in [(3,),(1,3),(2,1,3),()]:
            b = np.full(shape,.8)
            with self.subTest(shape=shape):
                self.gradcheck(lambda x,y:((x*y+x)/(y+2)).sum(),[a,b])

    def test_sum_axes_and_keepdims(self):
        a = np.arange(24,dtype=float).reshape(2,3,4)/20+.1
        for axis in [None,0,-1,(0,2),()]:
            for keep in [True,False]:
                with self.subTest(axis=axis,keep=keep):
                    self.gradcheck(lambda x:x.sum(axis=axis,keepdims=keep).sum(),[a])
                    self.gradcheck(lambda x:x.mean(axis=axis,keepdims=keep).sum(),[a])

    def test_invalid_axis(self):
        x = Tensor(np.ones((2,3)),requires_grad=True)
        for axis in [2,-3,(0,0)]:
            with self.subTest(axis=axis):
                with self.assertRaises((ValueError,IndexError)):
                    x.sum(axis=axis)

    def test_matmul_and_bias(self):
        rng=np.random.default_rng(1)
        self.gradcheck(lambda x,w,b:((x@w+b)**2).mean(),
                       [rng.normal(size=(3,2)),rng.normal(size=(2,4)),rng.normal(size=(4,))])

    def test_matmul_contract(self):
        with self.assertRaises(ValueError):
            Tensor([1.,2.]) @ Tensor([1.,2.])
        with self.assertRaises(ValueError):
            Tensor(np.ones((2,3))) @ Tensor(np.ones((2,4)))

    def test_reshape_transpose_gradient(self):
        x=np.arange(12,dtype=float).reshape(2,2,3)/10
        self.gradcheck(lambda a:((a.transpose(2,0,1).reshape(3,4))**2).sum(),[x])
        self.close(Tensor([[1.,2.],[3.,4.]]).T.data, [[1.,3.],[2.,4.]])

    def test_exp_log_power(self):
        x=np.array([.2,.7,1.3])
        self.gradcheck(lambda a:(a.exp()+a.log()+a**2+a**.5).sum(),[x])
        t=Tensor([0.,2.],requires_grad=True)
        (t**0).sum().backward()
        self.close(t.grad,[0.,0.])

    def test_relu_subgradient(self):
        t=Tensor([-1.,0.,2.],requires_grad=True)
        t.relu().sum().backward()
        self.close(t.grad,[0.,0.,1.])

    def test_log_domain(self):
        for data in [[0.],[-1.]]:
            with self.assertRaises((ValueError,FloatingPointError)):
                Tensor(data).log()

    def test_logsumexp_stable_and_grad(self):
        x=np.array([[1000.,1001.,999.],[-1000.,-999.,-1002.]])
        self.gradcheck(lambda a:a.logsumexp(axis=1).sum(),[x],atol=2e-6)
        expected=np.max(x,axis=1)+np.log(np.exp(x-np.max(x,axis=1,keepdims=True)).sum(axis=1))
        self.close(Tensor(x).logsumexp(axis=1).data,expected)

    def test_cross_entropy_hand_and_zero_sum(self):
        x=Tensor([[0.,0.],[np.log(3),0.]],requires_grad=True)
        loss=cross_entropy(x,np.array([0,1],dtype=np.int64))
        self.close(loss.data,(np.log(2)+np.log(4))/2)
        loss.backward()
        self.close(x.grad,[[-.25,.25],[.375,-.375]])
        self.close(x.grad.sum(axis=1),[0.,0.])

    def test_ce_mean_does_not_overflow_finite_result(self):
        x=Tensor([[0.,-1e308],[0.,-1e308]],requires_grad=True)
        loss=cross_entropy(x,np.array([1,1]))
        self.close(loss.data,1e308,atol=0,rtol=1e-14)
        loss.backward()
        self.close(x.grad,[[.5,-.5],[.5,-.5]])

    def test_cross_entropy_finite_difference(self):
        rng=np.random.default_rng(2)
        labels=np.array([0,2,1])
        self.gradcheck(lambda a:cross_entropy(a,labels),[rng.normal(size=(3,4))])

    def test_fused_composed_and_shift_invariance(self):
        rng=np.random.default_rng(3)
        raw=rng.normal(size=(5,4))
        labels=np.array([0,1,2,3,1])
        xs=[Tensor(raw,requires_grad=True),Tensor(raw,requires_grad=True)]
        losses=[cross_entropy(xs[0],labels),cross_entropy_composed(xs[1],labels)]
        for loss in losses: loss.backward()
        self.close(losses[0].data,losses[1].data)
        self.close(xs[0].grad,xs[1].grad)
        self.close(cross_entropy(Tensor(raw+10000),labels).data,losses[0].data,atol=1e-10)

    def test_cross_entropy_invalid_labels(self):
        for labels in [np.array([0.,1.]),np.array([0]),np.array([0,3]),np.array([-1,0])]:
            with self.subTest(labels=labels):
                with self.assertRaises((ValueError,TypeError)):
                    cross_entropy(Tensor(np.ones((2,3))),labels)

    def test_constructor_owns_data(self):
        raw=np.array([1.,2.])
        x=Tensor(raw,requires_grad=True)
        raw[0]=100
        self.close(x.data,[1.,2.])
        with self.assertRaises(ValueError):
            x.data[0]=5

    def test_modified_saved_operand(self):
        x=Tensor([2.,3.],requires_grad=True)
        y=(x*x).sum()
        x._assign(np.array([4.,5.]))
        with self.assertRaises(RuntimeError):
            y.backward()

    def test_float32_and_no_grad(self):
        x=Tensor(np.array([.2,.3],dtype=np.float32),requires_grad=True)
        self.assertEqual((x+1).dtype,np.dtype('float32'))
        (x*x).sum().backward()
        self.assertEqual(x.grad.dtype,np.dtype('float32'))
        with no_grad():
            y=x*x
        self.assertFalse(y.requires_grad)
        self.assertFalse(x.detach().requires_grad)

    def test_constructor_invalid_and_nonfinite(self):
        for raw in [np.empty((0,2)),np.array([np.nan]),np.array([np.inf])]:
            with self.subTest(raw=raw):
                with self.assertRaises((ValueError,TypeError,FloatingPointError)):
                    Tensor(raw)
        with self.assertRaises((ValueError,TypeError)):
            Tensor([1,2],dtype=np.int64)

    def test_graph_statistics_qualitative(self):
        x=Tensor([[1.,2.],[3.,4.]],requires_grad=True)
        stats=graph_statistics((x*x).sum())
        self.assertIsInstance(stats,dict)
        self.assertTrue(any(isinstance(v,(int,np.integer)) and v>=3 for v in stats.values()))


if __name__=='__main__':
    unittest.main()
