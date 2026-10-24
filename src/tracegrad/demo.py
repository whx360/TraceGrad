"""Deterministic dataset-free differentiation and learning demonstration."""
import numpy as np
from .tensor import Tensor,cross_entropy,no_grad
from .nn import Classifier
from .optim import Adam

def run_demo():
    """Return actual handcheck and XOR results; never accesses the network."""
    x=Tensor([1.,2.,3.],requires_grad=True)
    loss=(x*x).sum();loss.backward()
    np.testing.assert_array_equal(x.grad,[2.,4.,6.])
    inputs=Tensor(np.array([[-1.,-1.],[-1.,1.],[1.,-1.],[1.,1.]],dtype=np.float64))
    labels=np.array([0,1,1,0])
    model=Classifier(2,8,2,seed=4,dtype=np.float64)
    optimizer=Adam(model.parameters(),lr=.03)
    initial=cross_entropy(model(inputs),labels).item()
    for _ in range(160):
        optimizer.zero_grad(); objective=cross_entropy(model(inputs),labels)
        objective.backward();optimizer.step()
    with no_grad():
        logits=model(inputs);final=cross_entropy(logits,labels).item()
    predictions=logits.data.argmax(axis=1)
    np.testing.assert_array_equal(predictions,labels)
    return {'status':'passed','gradient_handcheck':{'loss':loss.item(),'gradient':x.grad.tolist()},
            'xor':{'initial_ce':initial,'final_ce':final,'predictions':predictions.tolist(),
                   'labels':labels.tolist(),'steps':160},
            'scope':'Synthetic functionality demo; not real-data generalization.'}
