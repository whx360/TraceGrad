"""Reproducible training, frozen-checkpoint evaluation and safe inference files."""
from __future__ import annotations
import json
from pathlib import Path
import time
import numbers
import numpy as np
from .tensor import Tensor, cross_entropy, no_grad
from .nn import Classifier, l2_penalty
from .optim import Adam, SGD


def classification_metrics(logits, labels):
    """(N,C),(N,) -> CE/accuracy/macro-F1/confusion. O(NC); all C classes counted."""
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels)
    if logits.ndim != 2 or not logits.size or not np.isfinite(logits).all():
        raise ValueError('logits must be a nonempty finite 2-D matrix')
    if labels.shape != (len(logits),) or labels.dtype.kind not in 'iu':
        raise ValueError('labels must be integer vector matching logits')
    classes = logits.shape[1]
    if np.any(labels<0) or np.any(labels>=classes):
        raise ValueError('labels out of range')
    shifted = logits-logits.max(axis=1,keepdims=True)
    per_sample = np.log(np.exp(shifted).sum(axis=1))-shifted[np.arange(len(labels)),labels]
    with np.errstate(over='ignore',invalid='ignore'):
        ce = np.mean(per_sample)
    if not np.isfinite(ce) and np.isfinite(per_sample).all():
        scale = np.max(np.abs(per_sample))
        if scale > 0: ce = np.mean(per_sample/scale)*scale
    if not np.isfinite(ce):
        raise ValueError('evaluation CE must be finite')
    prediction = logits.argmax(axis=1)
    confusion = np.bincount(labels.astype(np.int64)*classes+prediction,
                            minlength=classes*classes).reshape(classes,classes)
    true_positive = np.diag(confusion)
    denominator = confusion.sum(axis=0)+confusion.sum(axis=1)
    f1 = np.divide(2*true_positive,denominator,out=np.zeros(classes,dtype=float),where=denominator!=0)
    return {'cross_entropy':float(ce),'accuracy':float(np.mean(prediction==labels)),
            'macro_f1':float(np.mean(f1)),'per_class_f1':f1.tolist(),
            'confusion_matrix':confusion.tolist(),'samples':len(labels)}


def predict_logits(model, x, batch_size=1024):
    """Frozen model inference without graph construction; output (N,C)."""
    x=np.asarray(x)
    if x.ndim!=2 or len(x)==0 or batch_size<=0:
        raise ValueError('nonempty matrix and positive batch size required')
    values=[]
    dtype=model.parameters()[0].dtype
    with no_grad():
        for offset in range(0,len(x),batch_size):
            values.append(model(Tensor(x[offset:offset+batch_size],dtype=dtype)).numpy())
    return np.concatenate(values)


def evaluate(model,x,y,batch_size=1024):
    return classification_metrics(predict_logits(model,x,batch_size),y)


def save_checkpoint(path,model,metadata):
    """Inference-only NPZ; arrays+JSON string, never Python pickle."""
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    arrays=model.state_dict()
    arrays['__metadata__']=np.array(json.dumps(metadata,ensure_ascii=False))
    np.savez_compressed(path,**arrays)


def load_checkpoint(path):
    with np.load(path,allow_pickle=False) as archive:
        metadata=json.loads(str(archive['__metadata__'].item()))
        state={key:archive[key].copy() for key in archive.files if key!='__metadata__'}
    config=metadata['model']
    model=Classifier(input_dim=int(config['input_dim']),hidden_dim=int(config['hidden_dim']),
                     classes=int(config['classes']),seed=0,dtype=np.dtype(config['dtype']))
    model.load_state_dict(state)
    return model,metadata


def _write(path,payload):
    Path(path).write_text(json.dumps(payload,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def train_model(data,config,output,verbose=True):
    """Train fixed config and select epoch by validation CE only.

    data contains x/y_train,val,test. Saved model is for inference; resume is not
    implemented. Time O(epochs*n*(input*hidden+hidden*classes)), with corresponding
    linear simplification for hidden_dim0. Batch graph space is proportional to
    parameters and batch activations. Final test metrics never select a model.
    """
    output=Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'refusing to overwrite nonempty run: {output}')
    cfg=dict(config)
    def integer(key,default):
        value=cfg.get(key,default)
        if isinstance(value,(bool,np.bool_)) or not isinstance(value,numbers.Integral):
            raise ValueError(f'{key} must be an integer, without truncation')
        return int(value)
    epochs=integer('epochs',8); batch=integer('batch_size',256)
    seed=integer('seed',17); hidden=integer('hidden_dim',64); classes=integer('classes',10)
    lr=float(cfg.get('lr',.001)); l2=float(cfg.get('l2',0.))
    if epochs<=0 or batch<=0 or hidden<0 or classes<2 or not np.isfinite(l2) or l2<0:
        raise ValueError('invalid training configuration')
    dtype=np.dtype(cfg.get('dtype','float32'))
    for split in ['train','val','test']:
        x,y=np.asarray(data['x_'+split]),np.asarray(data['y_'+split])
        if x.ndim!=2 or not len(x) or y.shape!=(len(x),) or y.dtype.kind not in 'iu':
            raise ValueError(f'invalid {split} arrays')
        if not np.isfinite(x).all() or np.any(y<0) or np.any(y>=classes):
            raise ValueError(f'invalid {split} values')
    dimensions=data['x_train'].shape[1]
    if any(data['x_'+s].shape[1]!=dimensions for s in ['val','test']):
        raise ValueError('feature dimension differs across splits')
    model=Classifier(input_dim=dimensions,hidden_dim=hidden,classes=classes,seed=seed,dtype=dtype)
    opt_name=cfg.get('optimizer','adam')
    if opt_name=='adam': optimizer=Adam(model.parameters(),lr=lr)
    elif opt_name=='sgd': optimizer=SGD(model.parameters(),lr=lr,momentum=float(cfg.get('momentum',0.)))
    else: raise ValueError('optimizer must be adam or sgd')
    output.mkdir(parents=True,exist_ok=True)
    cfg.update(seed=seed,epochs=epochs,batch_size=batch,hidden_dim=hidden,classes=classes,lr=lr,l2=l2,dtype=dtype.name)
    _write(output/'config.json',cfg)
    history=[]; best_loss=float('inf'); best_epoch=0; best_state=None
    shuffle_rng=np.random.default_rng(seed+1000)
    n=len(data['x_train']); started=time.perf_counter()
    for epoch in range(1,epochs+1):
        indices=shuffle_rng.permutation(n)
        online_ce=0.; online_objective=0.; fit_start=time.perf_counter()
        for start in range(0,n,batch):
            chosen=indices[start:start+batch]
            optimizer.zero_grad()
            logits=model(Tensor(data['x_train'][chosen],dtype=dtype))
            ce=cross_entropy(logits,data['y_train'][chosen])
            loss=ce+l2_penalty(model,l2) if l2 else ce
            online_ce+=float(ce.item())*len(chosen)
            online_objective+=float(loss.item())*len(chosen)
            loss.backward(); optimizer.step()
        fit_seconds=time.perf_counter()-fit_start
        validation=evaluate(model,data['x_val'],data['y_val'])
        row={'epoch':epoch,'online_train_ce':online_ce/n,'online_train_objective':online_objective/n,
             'validation_ce':validation['cross_entropy'],'validation_accuracy':validation['accuracy'],
             'fit_seconds':fit_seconds}
        history.append(row)
        if row['validation_ce']<best_loss:
            best_loss=row['validation_ce']; best_epoch=epoch; best_state=model.state_dict()
        _write(output/'history.json',history)
        if verbose:
            print(f"epoch={epoch}/{epochs} onlineCE={row['online_train_ce']:.4f} valCE={row['validation_ce']:.4f} bestValCE={best_loss:.4f} valAcc={row['validation_accuracy']:.4f} fit={fit_seconds:.2f}s",flush=True)
    model.load_state_dict(best_state)
    meta={'format':'tracegrad_inference_v1','model':{'input_dim':dimensions,'hidden_dim':hidden,'classes':classes,'dtype':dtype.name},
          'seed':seed,'selected_epoch':best_epoch,'selection':'minimum validation CE','config':cfg}
    save_checkpoint(output/'model.npz',model,meta)
    train_metrics=evaluate(model,data['x_train'],data['y_train'])
    val_metrics=evaluate(model,data['x_val'],data['y_val'])
    test_logits=predict_logits(model,data['x_test'])
    test_metrics=classification_metrics(test_logits,data['y_test'])
    np.savez_compressed(output/'predictions.npz',logits=test_logits,labels=data['y_test'])
    result={'status':'completed','protocol':cfg.get('protocol','custom'), 'config':cfg,
            'selection':'minimum validation CE epoch; test not used for selection',
            'best_epoch':best_epoch,'train':train_metrics,'validation':val_metrics,'test':test_metrics,
            'fit_seconds':sum(row['fit_seconds'] for row in history),
            'wall_seconds':time.perf_counter()-started,'parameter_count':sum(p.size for p in model.parameters()),
            'provenance':data.get('provenance',{}),'checkpoint_scope':'selected model for inference, no resume state'}
    _write(output/'result.json',result)
    return result
