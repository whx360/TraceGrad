"""Command-line interface for TraceGrad verification, training and benchmarks."""
import argparse
import json
import os
from pathlib import Path
import sys
import unittest
from datetime import datetime

for _name in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ[_name] = '1'

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))


def main():
    parser = argparse.ArgumentParser(description='TraceGrad CPU autodiff and training project')
    parser.add_argument('command', choices=['smoke','test','verify','fetch','train','experiment','benchmark','report','demo'])
    parser.add_argument('--group', choices=['tensor', 'all'], default='all')
    parser.add_argument('--quick',action='store_true',help='labelled smoke experiment, never formal metrics')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--epochs',type=int,default=8)
    parser.add_argument('--hidden',type=int,default=64)
    parser.add_argument('--seed',type=int,default=17)
    parser.add_argument('--checkpoint',type=Path)
    args = parser.parse_args()
    def write(path,payload):
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(payload,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    def tests():
        pattern = 'test_tensor.py' if args.group == 'tensor' else 'test_*.py'
        suite = unittest.defaultTestLoader.discover(str(ROOT/'tests'), pattern=pattern)
        outcome = unittest.TextTestRunner(verbosity=1).run(suite)
        write(ROOT/'artifacts'/'verification'/'unit_tests.json',{
            'status':'passed' if outcome.wasSuccessful() else 'failed','tests':outcome.testsRun,
            'failures':[{'test':str(t),'traceback':s} for t,s in outcome.failures],
            'errors':[{'test':str(t),'traceback':s} for t,s in outcome.errors],
            'skipped':[{'test':str(t),'reason':s} for t,s in outcome.skipped],
            'group':args.group,'checked_at':datetime.now().astimezone().isoformat()})
        return outcome.wasSuccessful()
    if args.command == 'smoke':
        import numpy as np
        from tracegrad.tensor import Tensor
        x=Tensor([1.,2.,3.],requires_grad=True)
        loss=(x*x).sum(); loss.backward()
        np.testing.assert_array_equal(x.grad,[2.,4.,6.])
        path = ROOT / 'artifacts' / 'spine_smoke.json'
        write(path,{'status':'passed','scope':'real autograd handcheck, not real-data model accuracy',
                    'loss':loss.item(),'gradient':x.grad.tolist()})
        print(str(path))
    elif args.command == 'test':
        return 0 if tests() else 1
    elif args.command == 'verify':
        if not tests(): return 1
        from tracegrad.verification import verify_reference
        result=verify_reference()
        write(ROOT/'artifacts'/'verification'/'torch_reference.json',result)
        print(json.dumps(result,ensure_ascii=False))
    elif args.command=='fetch':
        from tracegrad.data import fetch_fashion
        result=fetch_fashion(ROOT/'data'/'raw'/'fashion-mnist')
        print('Fashion-MNIST source files verified:',len(result['files']))
    elif args.command=='train':
        from tracegrad.data import load_fashion
        from tracegrad.training import train_model
        data=load_fashion(ROOT/'data'/'raw'/'fashion-mnist')
        out=args.output or ROOT/'artifacts'/('train_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
        result=train_model(data,{'epochs':args.epochs,'hidden_dim':args.hidden,'seed':args.seed},out)
        print(json.dumps(result['test'],ensure_ascii=False))
    elif args.command in ('experiment','benchmark'):
        from tracegrad.experiments import run_campaign,run_benchmark
        result=run_campaign(ROOT,quick=args.quick,output=args.output) if args.command=='experiment' else run_benchmark(ROOT,output=args.output)
        if isinstance(result,dict):
            print(json.dumps({'status':result['status'],'comparisons':[
                {'id':item['id'],'ratio_of_median_times':item['ratio_of_median_times']}
                for item in result['comparisons']]},ensure_ascii=False))
        else:
            print(str(result))
    elif args.command=='report':
        from tracegrad.reporting import build_report
        print(str(build_report(ROOT)))
    elif args.command=='demo':
        if args.checkpoint is None:
            from tracegrad.demo import run_demo
            print(json.dumps(run_demo(),ensure_ascii=False,indent=2))
            return 0
        import numpy as np
        from tracegrad.data import load_fashion,CLASSES
        from tracegrad.training import load_checkpoint,predict_logits
        path=args.checkpoint
        model,metadata=load_checkpoint(path)
        data=load_fashion(ROOT/'data'/'raw'/'fashion-mnist')
        logits=predict_logits(model,data['x_test'][:12])
        print('Checkpoint selected by validation CE; first12 official test images:')
        for i,prediction in enumerate(logits.argmax(axis=1)):
            actual=int(data['y_test'][i])
            print(f'{i:02d} predicted={CLASSES[prediction]:12s} actual={CLASSES[actual]:12s} correct={prediction==actual}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
