import unittest
import numpy as np

from tracegrad import Tensor, cross_entropy, no_grad
from tracegrad.nn import Classifier, l2_penalty
from tracegrad.optim import Adam


class ModuleTests(unittest.TestCase):
    def test_initialization_is_reproducible_and_local(self):
        np.random.seed(123)
        expected = np.random.random(4)
        np.random.seed(123)
        a = Classifier(3, 5, 2, seed=17, dtype=np.float64)
        actual = np.random.random(4)
        np.testing.assert_array_equal(actual, expected)
        b = Classifier(3, 5, 2, seed=17, dtype=np.float64)
        for key in a.state_dict():
            np.testing.assert_array_equal(a.state_dict()[key], b.state_dict()[key])
        self.assertEqual(set(a.named_parameters()), {"w1", "b1", "w2", "b2"})
        self.assertTrue(all(p.requires_grad and p.is_leaf for p in a.parameters()))

    def test_linear_forward_can_be_hand_calculated(self):
        model = Classifier(2, 0, 2, dtype=np.float64)
        model.load_state_dict({"weight": np.array([[1.,2.],[3.,4.]]), "bias": np.array([.5,-.5])})
        out = model(Tensor([[2.,-1.]]))
        np.testing.assert_array_equal(out.data, [[-.5,-.5]])
        out.sum().backward()
        np.testing.assert_array_equal(model.named_parameters()["weight"].grad, [[2.,2.],[-1.,-1.]])
        np.testing.assert_array_equal(model.named_parameters()["bias"].grad, [1.,1.])

    def test_mlp_manual_forward_and_shape(self):
        model = Classifier(2, 2, 2, dtype=np.float64)
        state = {"w1":np.eye(2), "b1":np.zeros(2), "w2":np.array([[1.,2.],[3.,4.]]), "b2":np.ones(2)}
        model.load_state_dict(state)
        np.testing.assert_array_equal(model(Tensor([[-1.,2.]])).data, [[7.,9.]])

    def test_parameter_state_is_copied_and_load_strict(self):
        model = Classifier(2, 3, 2)
        original = model.state_dict()
        copied = model.state_dict()
        copied["w1"][:] = 99
        np.testing.assert_array_equal(model.state_dict()["w1"], original["w1"])
        for mutation, error in [
            (lambda s:s.pop("b2"), ValueError),
            (lambda s:s.update(w2=np.zeros((1,2),dtype=np.float32)), ValueError),
            (lambda s:s.update(b2=np.array([np.nan,0],dtype=np.float32)), ValueError),
            (lambda s:s.update(b2=np.zeros(2,dtype=np.float64)), TypeError),
        ]:
            candidate = model.state_dict(); mutation(candidate)
            with self.assertRaises(error): model.load_state_dict(candidate)
            for name, value in original.items(): np.testing.assert_array_equal(model.state_dict()[name], value)

    def test_l2_excludes_bias_and_has_expected_derivative(self):
        model = Classifier(2, 3, 2, dtype=np.float64)
        penalty = l2_penalty(model, .2)
        state = model.state_dict()
        self.assertAlmostEqual(penalty.item(), .1*(np.sum(state["w1"]**2)+np.sum(state["w2"]**2)))
        penalty.backward()
        for name, parameter in model.named_parameters().items():
            if name in model.weight_names: np.testing.assert_allclose(parameter.grad, .2*state[name])
            else: self.assertIsNone(parameter.grad)
        zero = l2_penalty(model, 0)
        self.assertEqual(zero.item(), 0)
        self.assertEqual(zero.dtype, model.dtype)

    def test_invalid_inputs(self):
        for args in [dict(input_dim=0), dict(hidden_dim=-1), dict(classes=1), dict(seed=-1)]:
            with self.assertRaises(ValueError): Classifier(**args)
        with self.assertRaises(TypeError): Classifier(dtype=np.float16)
        model=Classifier(2,3,2)
        with self.assertRaises(TypeError): model(np.ones((3,2),dtype=np.float32))
        with self.assertRaises(ValueError): model(Tensor(np.ones((3,4),dtype=np.float32)))
        with self.assertRaises(TypeError): model(Tensor(np.ones((3,2),dtype=np.float64)))
        for coefficient in [-1,np.nan,np.inf]:
            with self.assertRaises(ValueError):l2_penalty(model,coefficient)

    def test_no_grad_evaluation_constructs_no_graph(self):
        model=Classifier(2,3,2)
        with no_grad(): out=model(Tensor(np.ones((3,2),dtype=np.float32)))
        self.assertFalse(out.requires_grad)

    def test_tiny_xor_learning(self):
        x=Tensor(np.array([[-1.,-1.],[-1.,1.],[1.,-1.],[1.,1.]],dtype=np.float64))
        labels=np.array([0,1,1,0])
        model=Classifier(2,8,2,seed=4,dtype=np.float64)
        optimizer=Adam(model.parameters(),lr=.03)
        initial=cross_entropy(model(x),labels).item()
        for _ in range(160):
            optimizer.zero_grad();loss=cross_entropy(model(x),labels);loss.backward();optimizer.step()
        with no_grad():
            logits=model(x); final=cross_entropy(logits,labels).item()
        self.assertLess(final,.08)
        self.assertLess(final,initial/5)
        np.testing.assert_array_equal(logits.data.argmax(axis=1),labels)


if __name__ == '__main__': unittest.main()
