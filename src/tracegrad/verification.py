"""Independent PyTorch numerical references; not used by TraceGrad training."""
import importlib.metadata
import numpy as np
from .tensor import Tensor, cross_entropy
from .nn import Classifier, l2_penalty
from .optim import Adam, SGD


def verify_reference():
    """Compare logits/loss/all gradients and repeated optimizer updates.

    Uses Torch's own autograd and optimizer implementations. Reference arrays
    share initial values only, never TraceGrad gradient code. Results explicitly
    state tested shapes/dtypes/tolerances, not all-program correctness.
    """
    import torch
    torch.set_num_threads(1)
    cases=[]
    for dtype,tdtype,tolerance in [(np.float64,torch.float64,2e-11),(np.float32,torch.float32,3e-5)]:
        for optimizer_name in ['sgd','adam']:
            model=Classifier(input_dim=5,hidden_dim=4,classes=3,seed=31,dtype=dtype)
            state=model.state_dict()
            params={key:torch.tensor(value,dtype=tdtype,requires_grad=True) for key,value in state.items()}
            if optimizer_name=='adam':
                ours=Adam(model.parameters(),lr=.003)
                reference=torch.optim.Adam(list(params.values()),lr=.003,betas=(.9,.999),eps=1e-8)
            else:
                ours=SGD(model.parameters(),lr=.03,momentum=.8)
                reference=torch.optim.SGD(list(params.values()),lr=.03,momentum=.8)
            rng=np.random.default_rng(200)
            max_logits=max_gradient=max_parameter=0.
            max_loss=0.
            for step in range(5):
                x=rng.normal(size=(7,5)).astype(dtype)
                labels=rng.integers(0,3,size=7,dtype=np.int64)
                ours.zero_grad(); reference.zero_grad(set_to_none=True)
                output=model(Tensor(x))
                loss=cross_entropy(output,labels)+l2_penalty(model,.02)
                tx=torch.tensor(x,dtype=tdtype)
                ty=torch.tensor(labels,dtype=torch.int64)
                tout=torch.relu(tx@params['w1']+params['b1'])@params['w2']+params['b2']
                tloss=torch.nn.functional.cross_entropy(tout,ty)+.01*(params['w1'].square().sum()+params['w2'].square().sum())
                np.testing.assert_allclose(output.data,tout.detach().numpy(),atol=tolerance,rtol=tolerance)
                np.testing.assert_allclose(loss.data,tloss.detach().numpy(),atol=tolerance,rtol=tolerance)
                max_logits=max(max_logits,float(np.max(np.abs(output.data-tout.detach().numpy()))))
                max_loss=max(max_loss,abs(float(loss.item())-float(tloss.item())))
                loss.backward(); tloss.backward()
                for key,parameter in model.named_parameters().items():
                    diff=float(np.max(np.abs(parameter.grad-params[key].grad.numpy())))
                    max_gradient=max(max_gradient,diff)
                    np.testing.assert_allclose(parameter.grad,params[key].grad.numpy(),atol=tolerance,rtol=tolerance)
                ours.step(); reference.step()
                for key,parameter in model.named_parameters().items():
                    diff=float(np.max(np.abs(parameter.data-params[key].detach().numpy())))
                    max_parameter=max(max_parameter,diff)
                    np.testing.assert_allclose(parameter.data,params[key].detach().numpy(),atol=tolerance,rtol=tolerance)
            cases.append({'dtype':np.dtype(dtype).name,'optimizer':optimizer_name,'steps':5,
                          'atol':tolerance,'rtol':tolerance,'max_logit_abs_error':max_logits,
                          'max_loss_abs_error':max_loss,'max_gradient_abs_error':max_gradient,
                          'max_parameter_abs_error':max_parameter})
    return {'status':'passed','reference':'PyTorch autograd + torch.optim',
            'torch_version':importlib.metadata.version('torch'),'cases':cases,
            'scope':'4 fixed dtype/optimizer cases, 5 steps each; not a formal proof'}
