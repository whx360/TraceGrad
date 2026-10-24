import unittest
import numpy as np
from tracegrad import Tensor
from tracegrad.optim import SGD, Adam


class OptimizerTests(unittest.TestCase):
    def test_sgd_hand_updates_and_zero_grad(self):
        p=Tensor([1.,2.],requires_grad=True);opt=SGD([p],lr=.1)
        p.grad=np.array([.5,-1.]);opt.step()
        np.testing.assert_allclose(p.data,[.95,2.1],rtol=0,atol=1e-15)
        opt.zero_grad();self.assertIsNone(p.grad)
        opt.step();self.assertEqual(opt.step_count,1)

    def test_momentum_hand_updates(self):
        p=Tensor([1.,2.],requires_grad=True);opt=SGD([p],lr=.1,momentum=.9)
        p.grad=np.array([.5,-1.]);opt.step();opt.step()
        np.testing.assert_allclose(p.data,[.855,2.29],rtol=0,atol=1e-15)

    def test_adam_hand_bias_correction(self):
        p=Tensor([1.,-2.],requires_grad=True);opt=Adam([p],lr=.1,beta1=.8,beta2=.9,eps=1e-7)
        g1=np.array([.5,-.25]);p.grad=g1;opt.step()
        expected=np.array([1.,-2.])-.1*g1/(np.abs(g1)+1e-7)
        np.testing.assert_allclose(p.data,expected,rtol=0,atol=1e-14)
        g2=np.array([-.1,.5]);p.grad=g2;opt.step()
        m=.8*(.2*g1)+.2*g2;v=.9*(.1*g1*g1)+.1*g2*g2
        expected-=.1*(m/(1-.8**2))/(np.sqrt(v/(1-.9**2))+1e-7)
        np.testing.assert_allclose(p.data,expected,rtol=0,atol=1e-14)

    def test_missing_gradient_does_not_age_adam_parameter(self):
        p=Tensor(1.,requires_grad=True);q=Tensor(1.,requires_grad=True)
        opt=Adam([p,q],lr=.1)
        p.grad=np.array(.3);opt.step();p.grad=None;q.grad=np.array(.3);opt.step()
        self.assertAlmostEqual(p.item(),q.item(),places=14)
        self.assertEqual(opt.state_dict()["steps"],[1,1])

    def test_invalid_later_gradient_has_no_partial_update(self):
        for kind in [SGD,Adam]:
            p=Tensor([1.,2.],requires_grad=True);q=Tensor([3.,4.],requires_grad=True)
            opt=kind([p,q],lr=.1)
            p.grad=np.array([1.,1.]);q.grad=np.array([np.nan,1.])
            with self.assertRaises(ValueError): opt.step()
            np.testing.assert_array_equal(p.data,[1.,2.]);np.testing.assert_array_equal(q.data,[3.,4.])
            self.assertEqual(opt.step_count,0)
            q.grad=np.array([1.])
            with self.assertRaises(ValueError): opt.step()
            np.testing.assert_array_equal(p.data,[1.,2.])

    def test_invalid_configuration_or_parameters(self):
        p=Tensor(1.,requires_grad=True)
        for params in [[],[p,p],[Tensor(1.)],[p*2]]:
            with self.assertRaises(ValueError):SGD(params,lr=.1)
        for lr in [0,-1,np.inf,np.nan]:
            with self.assertRaises(ValueError):Adam([p],lr=lr)
        for kwargs in [dict(momentum=-1),dict(momentum=1)]:
            with self.assertRaises(ValueError):SGD([p],lr=.1,**kwargs)
        for kwargs in [dict(beta1=1),dict(beta2=-1),dict(eps=0)]:
            with self.assertRaises(ValueError):Adam([p],**kwargs)

    def test_state_roundtrip_and_copy(self):
        for kind,kwargs in [(SGD,dict(momentum=.9)),(Adam,{})]:
            p=Tensor([1.,2.],requires_grad=True);first=kind([p],lr=.1,**kwargs)
            p.grad=np.array([.3,-.2]);first.step()
            q=Tensor(p.numpy(),requires_grad=True);second=kind([q],lr=.1,**kwargs)
            second.load_state_dict(first.state_dict())
            p.grad=np.array([-.7,.1]);q.grad=p.grad.copy();first.step();second.step()
            np.testing.assert_array_equal(p.data,q.data)
            self.assertEqual(first.step_count,second.step_count)
            bad=first.state_dict();bad["configuration"]["lr"]*=2
            with self.assertRaises(ValueError):second.load_state_dict(bad)
        state=first.state_dict();state["m"][0][:]=999
        self.assertFalse(np.any(first.state_dict()["m"][0]==999))

    def test_adam_rejects_corrupt_second_moment_state(self):
        p=Tensor([1.,2.],requires_grad=True);opt=Adam([p]);p.grad=np.array([.1,.2]);opt.step()
        state=opt.state_dict();state["v"][0][0]=-1
        with self.assertRaises(ValueError):opt.load_state_dict(state)
        self.assertGreaterEqual(opt.state_dict()["v"][0][0],0)


if __name__ == '__main__': unittest.main()
