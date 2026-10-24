# Fixed-protocol experiments

Recorded2026-10-24 on Ryzen7 7435H/WindowsCPU/NumPy1.26.4, one BLASthread. The compact records in `benchmarks/reference` are measured snapshots. Runtimes vary across machines; a source identity file covers the numerical implementation and fixed configuration used for them.

## Data and selection

Fashion-MNIST author files are validated by advertisedMD5, recordedSHA256, IDXmagic/count/shape and exactpayloadlength. Official60000train is split by seed2026,600examples/class for validation:54000train/6000validation. Official10000test is unchanged. Pixels are float32/255, with no test-fitted preprocessing. Byte-identical image audit found0withintrain and0train-test duplicates; near-duplicates were not assessed.

Seeds17/29/43;8epochs,batch256,Adam lr.001. Softmax and784→64→10 ReLU MLP compare representation capacity. The L2 ablation changes only lambda0→1e-4, penalizing weights but not biases. Initialization/shuffle sources are fixed separately.

Each run selects the lowest validationCE epoch. The primary variant minimizes mean selected validationCE across seeds. All predetermined configurations are reported on test; test scores do not drive further tuning. Frozen evaluation uses float64logits for CE, whereas online batch losses use modeldtypefloat32. Online trainingloss and frozen full-setloss have different meanings.

## Classification results

| Configuration | TestAccuracy(mean ± sampleSD) | macro-F1 | TestCE | MeanvalidationCE |
|---|---:|---:|---:|---:|
|Softmax|83.480% ±0.202pp|0.833431|0.472524|0.442731|
|MLP|86.090% ±0.114pp|0.860119|0.397045|0.369716|
|MLP,L2=1e-4|85.927% ±0.060pp|0.858358|0.399736|0.372204|

SampleSD usesddof1,n3. These descriptive results are not confidence intervals or significance tests. MLP gains2.61percentagepoints overlinear. L2 is lower in all3pairedseeds by.14/.12/.23pp; this does not imply regularization is universally ineffective. The limited8epochbudget is not an optimized hyperparameter search or dataset-best claim.

The validation-selected illustrative run is MLPseed43,epoch7. Its frozen train/validation/testaccuracy is88.0407/87.3333/86.22%; the main result remains the three-seed86.09%. Shirt has56.9%recall, with208Shirt→T-shirt errors; Pullover→Coat contributes185errors. Similar appearance and absent spatial inductivebias are hypotheses, not proven causal explanations.

## Profiling and optimization

Original50stepMLP profile:0.1074311s total; backward cumulative0.0631973s; matrixVJP cumulative0.0409191s. Inspection found a discarded256×64 @64×784 inputgradient product, approximately25,690,112multiply/addFLOPs eachstep. SelectiveVJPs skip it when images need no gradient.

Formal paired timing:seed2026,3warmups/variant,15samples/variant,10innercalls/sample, alternatingorder. Times are medians of inner-call means, not single-request latency percentiles.

| Workload | Baseline(ms) | Optimized(ms) | Ratio | Graphnodes | Retainedgraphbytes |
|---|---:|---:|---:|---|---|
|MLPfloat32,256×784→64→10|1.68663|1.06237|1.5876×|11→11|1252140→1252140|
|CEfloat64,128×10|.42626|.12963|3.2883×|12→2|56352→21512|
|CEfloat64,512×10|.53482|.19398|2.7571×|12→2|225312→86024|
|CEfloat64,256×100|1.30329|.36378|3.5826×|12→2|1034272→411656|
|CEfloat64,256×1000|9.27509|4.10368|2.2602×|12→2|10250272→4098056|

All fixed-input loss/parameter-gradient differences in these comparisons were0. The MLP comparison includes gradientclearing+forward/backward, nooptimizer/data/validation. CEbaseline includes detached rowshift andone-hot graph selection; it is not every possible handwrittenCE. Graphbytes count deduplicated retainedndarraybuffers, not peakRSS or temporaryworkspace. No full-training, GPU, multithreading or PyTorch performance advantage is established.

## Reproduction

`python run.py verify` checks numericalcontracts. `fetch` downloads checked original data. `experiment` runs fixedvariants/seeds; `benchmark` separately generates paired rawtimings and profiles. Existing nonemptyoutputs are rejected; use a fresh`--output`path. `report` reads canonicalformalrecords afterverify. Rawdatasets/models/runtimeoutputs are not versioned.

Independent tests and these experiments provide bounded evidence. Fresh timing runs need not match recorded walltimes. Classic methods are attributed in references.md.
