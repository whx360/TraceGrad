"""Standalone offline experiment report from actual campaign records."""
from __future__ import annotations
import base64
from datetime import datetime
from html import escape
import io
import json
from pathlib import Path
import numpy as np


def _png(figure):
    buffer=io.BytesIO()
    figure.savefig(buffer,format='png',dpi=140,bbox_inches='tight')
    return 'data:image/png;base64,'+base64.b64encode(buffer.getvalue()).decode('ascii')


def build_report(root):
    """Read completed fixed-protocol records; emit one embedded-asset HTML.

    This function never trains/selects on test data. Displays existing validation
    selection and all fixed variants. Plotting memory/time is not benchmarked.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from .data import load_fashion,CLASSES
    root=Path(root)
    campaign=json.loads((root/'results'/'formal_v1'/'campaign.json').read_text(encoding='utf-8'))
    if campaign['status']!='completed' or campaign.get('smoke_only'):
        raise ValueError('report requires completed formal campaign')
    aggregate=campaign['aggregates']
    chosen=campaign['selected_variant']
    selected=min([r for r in campaign['runs'] if r['variant']==chosen],key=lambda r:r['validation']['cross_entropy'])
    runs=campaign['runs']
    def mean_std(variant,key,percentage=False):
        metric=aggregate[variant]['metrics'][key]
        scale=100 if percentage else 1
        return f"{metric['mean']*scale:.3f} ± {metric['std']*scale:.3f}" if metric['std'] is not None else f"{metric['mean']*scale:.3f}"
    rows=''.join(f"<tr><td>{escape(variant)}{' · 验证集选定' if variant==chosen else ''}</td><td>{mean_std(variant,'test_accuracy',True)}%</td><td>{mean_std(variant,'test_macro_f1',True)}%</td><td>{mean_std(variant,'test_cross_entropy')}</td><td>{mean_std(variant,'fit_seconds')} s</td></tr>" for variant in aggregate)
    plt.rcParams.update({'font.family':'DejaVu Sans','axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(11,3.5))
    for variant in aggregate:
        run=next(r for r in runs if r['variant']==variant and r['config']['seed']==17)
        history=json.loads((root/run['directory']/'history.json').read_text(encoding='utf-8'))
        axes[0].plot([h['epoch'] for h in history],[h['online_train_ce'] for h in history],marker='.',label=variant)
        axes[1].plot([h['epoch'] for h in history],[h['validation_ce'] for h in history],marker='.',label=variant)
    axes[0].set(title='Online training CE (seed17)',xlabel='Epoch',ylabel='CE')
    axes[1].set(title='Frozen validation CE (seed17)',xlabel='Epoch',ylabel='CE')
    for ax in axes: ax.grid(alpha=.2); ax.legend()
    curves=_png(fig); plt.close(fig)
    matrix=np.array(selected['test']['confusion_matrix'])
    fig,ax=plt.subplots(figsize=(8,6))
    im=ax.imshow(matrix,cmap='Blues')
    ax.set(xticks=range(10),yticks=range(10),xticklabels=CLASSES,yticklabels=CLASSES,xlabel='Predicted',ylabel='Actual',title=f"Test confusion: {chosen}, seed{selected['config']['seed']}")
    plt.setp(ax.get_xticklabels(),rotation=50,ha='right')
    for i in range(10):
        for j in range(10): ax.text(j,i,str(matrix[i,j]),ha='center',va='center',fontsize=7,color='white' if matrix[i,j]>matrix.max()*.55 else '#203047')
    fig.colorbar(im,ax=ax,shrink=.7)
    confusion=_png(fig); plt.close(fig)
    data=load_fashion(root/'data'/'raw'/'fashion-mnist')
    with np.load(root/selected['directory']/'predictions.npz',allow_pickle=False) as p:
        logits=p['logits'].astype(float); labels=p['labels']; prediction=logits.argmax(axis=1)
    shifted=logits-logits.max(axis=1,keepdims=True)
    probability=np.exp(shifted); probability/=probability.sum(axis=1,keepdims=True)
    confidence=probability.max(axis=1)
    errors=np.flatnonzero(prediction!=labels)
    order=errors[np.argsort(-confidence[errors])[:16]]
    fig,axes=plt.subplots(2,8,figsize=(12,4))
    for ax in axes.flat: ax.axis('off')
    for ax,index in zip(axes.flat,order):
        ax.imshow(data['x_test'][index].reshape(28,28),cmap='gray',vmin=0,vmax=1)
        ax.set_title(f"True: {CLASSES[labels[index]]}\nPred: {CLASSES[prediction[index]]}\np={confidence[index]:.2f}",fontsize=7)
    errors_image=_png(fig); plt.close(fig)
    offdiag=matrix.copy(); np.fill_diagonal(offdiag,0)
    pairs=np.dstack(np.unravel_index(np.argsort(offdiag.ravel())[-5:][::-1],offdiag.shape))[0]
    pair_text=''.join(f'<li>{escape(CLASSES[i])} → {escape(CLASSES[j])}：{matrix[i,j]} 张</li>' for i,j in pairs)
    verification=json.loads((root/'artifacts'/'verification'/'unit_tests.json').read_text(encoding='utf-8'))
    reference=json.loads((root/'artifacts'/'verification'/'torch_reference.json').read_text(encoding='utf-8'))
    max64=max(c['max_gradient_abs_error'] for c in reference['cases'] if c['dtype']=='float64')
    performance_path=root/'results'/'performance'/'benchmark.json'
    performance=json.loads(performance_path.read_text(encoding='utf-8')) if performance_path.exists() else None
    perf_html='<p>性能结果尚未生成。</p>'
    if performance:
        perf_html='<details><summary>查看真实性能原始汇总与口径</summary><pre>'+escape(json.dumps(performance,ensure_ascii=False,indent=2))+'</pre></details>'
    body=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>TraceGrad · 实验与展示</title>
<style>*{{box-sizing:border-box}}body{{margin:0;background:#f3f6fb;color:#18273d;font:16px/1.75 system-ui,"Microsoft YaHei",sans-serif}}main{{max-width:1120px;margin:auto;padding:40px 24px 90px}}header{{background:#10233d;color:#fff;padding:42px;border-radius:22px}}small,.muted{{color:#65768c}}header small{{color:#79e0d1}}h1{{font-size:clamp(30px,5vw,50px);line-height:1.2;margin:12px 0 20px}}h2{{margin-top:0;font-size:26px}}section{{background:white;border:1px solid #e0e7ef;border-radius:18px;padding:30px;margin-top:24px}}.cards{{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin-top:24px}}.card{{background:#fff;border-radius:16px;padding:22px;border:1px solid #e0e7ef}}.number{{font-size:30px;font-weight:750;color:#076b64}}img{{width:100%;height:auto}}table{{width:100%;border-collapse:collapse;white-space:nowrap}}td,th{{text-align:left;border-bottom:1px solid #e6ebf2;padding:12px}}.scroll{{overflow:auto}}pre{{font:12px/1.6 ui-monospace,monospace;overflow:auto;max-height:550px;background:#eef3f8;padding:20px}}code{{background:#eef3f8;padding:2px 6px;border-radius:4px}}a{{color:#076b64}}details{{margin-top:15px}}@media(max-width:700px){{main{{padding:16px 12px 50px}}header,section{{padding:22px}}.cards{{grid-template-columns:1fr}}}}</style>
<main><header><small>TRACEGRAD / CPU AUTODIFF / FORMAL EXPERIMENT</small><h1>从一个梯度，走到真实实验。</h1><p>自实现张量计算图、反向传播、神经网络层与优化器；以独立参考、真实分类任务和受控性能实验核验。</p><p>底层数组与矩阵乘由 NumPy / BLAS 提供；PyTorch 仅作独立数值对照。经典算法有来源，不声称原创。</p></header>
<div class="cards"><div class="card"><small>官方测试准确率 / 验证选定配置</small><div class="number">{mean_std(chosen,'test_accuracy',True)}%</div><small>3 seeds · 样本标准差</small></div><div class="card"><small>独立自动化检查</small><div class="number">{verification['tests']} tests</div><small>失败 {len(verification['failures'])} · 错误 {len(verification['errors'])} · 跳过 {len(verification['skipped'])}</small></div><div class="card"><small>float64 参数梯度对照最大绝对差</small><div class="number">{max64:.2e}</div><small>固定MLP / SGD与Adam各5步</small></div></div>
<section><h2>数据和实验协议</h2><p>Fashion-MNIST：官方训练数据固定分层划为 54,000 训练 / 6,000 验证，官方 10,000 测试不变。像素只除以255。种子17、29、43；每配置8轮；batch256；Adam学习率0.001；单BLAS线程。</p><p>每轮只依据验证集交叉熵选择checkpoint。报告全部预先固定配置；三配置平均验证CE选中 <strong>{escape(chosen)}</strong>。结果是离线固定数据上的证据，不表示其他任务或部署条件。</p><div class="scroll"><table><tr><th>固定配置</th><th>Test accuracy</th><th>Test macro-F1</th><th>Test CE</th><th>Fit time</th></tr>{rows}</table></div><p class="muted">均值±样本标准差，n=3；没有统计显著性声明。不同模型耗时包含不同计算量，不当作同函数优化加速比。L2只惩罚权重，直接写入目标，非AdamW。</p></section>
<section><h2>训练与验证曲线</h2><img alt="训练与验证CE" src="{curves}"><p class="muted">左图为参数持续更新期间的batch加权在线loss；右图为每轮冻结参数的验证loss，二者测量含义不同。展示seed17，所有种子逐epoch记录保存在各run目录。</p></section>
<section><h2>错误分析</h2><p>以下图像使用验证集选定配置中验证CE最低的种子，仅用于具体错误观察；整体指标使用三个种子。示例不是随机代表样本，而是该模型最自信的16个错误。</p><img alt="混淆矩阵" src="{confusion}"><ul>{pair_text}</ul><img alt="高置信错误样例" src="{errors_image}"><p>可研究问题：外观相近类别是否需要空间结构？当前MLP展平图像，未实现卷积；此观察不能单凭错误图证明卷积一定提升。</p></section>
<section><h2>性能证据与边界</h2><p>先profiling，再比较组合/融合交叉熵与是否计算不需要的输入梯度。等价性检查先于计时。图保留数组字节数不是进程峰值内存；损失算子加速不能外推为全训练加速。</p>{perf_html}</section>
<section><h2>直接运行与学习路线</h2><pre>python run.py smoke
python run.py verify
python run.py demo
python run.py train --epochs 8
python run.py report</pre><p>设计：docs/architecture.md → 数值契约：docs/numerics.md → 实验：docs/experiments.md。</p><p>模型文件仅供推理，CLI未实现断点续训。默认demo无需下载；--checkpoint用于真实模型。</p><p class="muted">生成于 {escape(datetime.now().astimezone().isoformat())}。本文件已嵌入全部图像，无外部运行资源。</p></section></main></html>'''
    output=root/'reports'/'TraceGrad_report.html'
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(body,encoding='utf-8')
    # Also save figures as reusable scientific artifacts, identical to report.
    figures=root/'reports'/'figures';figures.mkdir(exist_ok=True)
    for name,data_uri in [('learning_curves.png',curves),('confusion_matrix.png',confusion),('confident_errors.png',errors_image)]:
        (figures/name).write_bytes(base64.b64decode(data_uri.split(',',1)[1]))
    return output
