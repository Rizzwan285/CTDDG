# ARCHITECTURE

Framework: **MXNet 1.9.1 Gluon**, single GPU (`mx.gpu()` hardcoded throughout).
All model code lives inside notebooks; there are **no `.py` modules**, and definitions are
**duplicated** between `pretraining.ipynb` and `Generating_samples.ipynb`.

## Config (`outputs/pretrain/logs/configs.json` + literals in `pretraining.ipynb`)
```
N_A=65 (atom types, from data/atom_types.txt)   N_B=4 (AROMATIC,SINGLE,DOUBLE,TRIPLE)   D=2
F_e=16   F_h=[32,64,128,128,256,256]   F_skip=256   F_c=[512]   Fh_policy=128   N_rnn=3
batch=8   k=5 (K, paths)   p=0.8 (α, randomness)   lr=1e-3   decay=1e-3/100 steps
clip_grad=3.0   summary_step=500   iterations=480000
```
⚠ `N_A`, `N_B`, `D` are **not** in `configs.json` — derived at runtime / passed literally.

## Class map

### `code/pretraining.ipynb` — unconditional
```
MoleculeSpec                    atom/bond registry; index = line number in atom_types.txt
BalancedSampler(Sampler)        length-bucketed batching
MolLoader(DataLoader)           collate + from_numpy_to_tensor (mx.gpu() hardcoded)
 └ MolRNNLoader                 + graph_to_rnn, rnn_to_graph, NX_cum for the GRU
GraphConvFn / EfficientGraphConvFn / SegmentSumFn      custom autograd Functions
Linear_BN · BatchNorm(custom) · GraphConv · Policy     layers
MoleculeGenerator(nn.Block)
 └ MoleculeGenerator_RNN        + gluon.rnn.GRU, _rnn_train / _rnn_test
    └ VanillaMolGen_RNN         ← TRAINED (6,986,506 params)
```

### `code/Generating_samples.ipynb` cell 14 — conditional (**never trained**)
```
_TwoLayerDense                  Dense→BN→ReLU→Dense→softmax
CMoleculeGenerator_RNN(MoleculeGenerator_RNN)
 └ CVanillaMolGen_RNN           + dense_policy_0, + linear_c[0..5]
```
### `code/Generating_samples.ipynb` cell 16 — sampling
```
_decode_step(...)               NumPy sequential decoder (append/connect/end)
Builder → CVanilla_RNN_Builder  loads configs.json + ckpt.params; .sample(); .to_nd()
```
⚠ **No `Vanilla_RNN_Builder`** — there is no way to sample from the pretrained checkpoint.

## Data flow (training)
```
SMILES line "<smi> <class>"
 → process_single(smiles, k, p)
     get_graph_from_smiles     RDKit → nx.Graph, atom_types, atom_ranks (CanonicalRankAtoms)
     traverse_graph × k        Q_α DFS; canonical rank w.p. p else random; accumulates log_p
     single_reorder / single_expand   → actions [n,5] = (type, atom, bond, append_pos, connect_pos)
     ⚠ tox_class = [1]*k       HARDCODED — smiles_class parsed then discarded
 → MolRNNLoader._collate_fn    batch offsets; graph_to_rnn / rnn_to_graph / NX_cum
 → merge_single_0              get_d() → D_2, D_3;  A_list = 4 bond slices + D_2 + D_3  (len 6)
 → from_numpy_to_tensor        → CSR sparse matrices on mx.gpu()
```

## Model flow (forward, mode='loss')
```
X[n] int32, last_append_mask[n]∈{0,1,2}
 → embedding_atom(X) + embedding_mask(mask)                    [n,16]
 → 6 × GraphConv:  concat([X, A₁X..A₄X, D₂X, D₃X]) @ W_i       [n,32→64→128→128→256→256]
      (BatchNorm+ReLU before each except layer 0)
      (conditional: + linear_c[i](c)[ids,:])
 → concat all 6                                                [n,864]
 → bn_skip → ReLU → linear_skip → ReLU                         [n,256]
 → dense (Linear_BN 256→512)                                   [n,512]
 → X_avg = SegmentSum(NX_rep)/NX ;  X_curr = X[NX_cum-1]
 → concat                                                      [m,1024]   (= h_G ‖ h_{v*})
 → GRU(3 layers, hidden 512)                                   [m,512]
 → Policy(X[n,512], NX, NX_rep, X_mol[m,512]):
       concat(X, X_mol[NX_rep])       [n,1024]
       linear_h→ReLU→linear_x→exp     [n,264]     264 = N_B + N_B*N_A
       linear_h_t→ReLU→linear_x_t→exp [m,1]
       joint normalise over (all atoms × 264) + end
   → append[n,65,4] · connect[n,4] · end[m]
```
Loss (`_likelihood`): masked log-probs per action → `SegmentSumFn(iw_ids)` → `+ loss_init`
→ reshape `[batch,K]` → `logsumexp(log_p_x − log_p_sigma, axis=1) − log K` → `forward` returns
`-l.mean()`.
⚠ `tox_class_batch` is an argument of `_likelihood` and is **never referenced**.

## Conditioning flow (`CVanillaMolGen_RNN`, never trained)
```
protein FASTA ─offline, frozen─► bio_embeddings ─► c ∈ ℝ^{N_C}  (text file, tab-separated)
   ├─► dense_policy_0 = _TwoLayerDense(N_C, N_A*3, N_A) ──► P(first atom)   [init action]
   └─► linear_c[i] = Dense(F_h[i], no bias, in_units=N_C)
           added as linear_c(c)[ids,:] to EVERY GCN layer output
```
`ids` = per-atom → per-molecule index (`NX_rep`). Essential: molecules have different sizes.
`_policy_0(c) = dense_policy_0(c) + 0.0 * policy_0.data(ctx)` — the zero term keeps the
unconditional `policy_0` parameter registered so pretrained checkpoints still load.

## Generation flow
```
CVanilla_RNN_Builder(model_loc, gpu_id=0)
  reads configs.json (MUST contain N_C) → CVanillaMolGen_RNN(N_A, N_B, D=2, **configs)
  load_parameters(ckpt.params)            ⚠ no allow_missing → must match exactly
.sample(num_samples, c, output_type, sanitize=True, random=True):
  mode='decode_0'    → _policy_0(c) → sample first atom (per molecule)
  loop (max 100):
    mode='decode_step' → append/connect/end; joint categorical draw over the flattened
                          concat(append, connect, [end]) renormalised to 1
    ⚠ the second-atom branch (A.shape[0]==0) is ALWAYS argmax, even when random=True
  → graph list → get_mol_from_graph_list(sanitize=True) → Chem.MolToSmiles
```
Validity is enforced **post hoc** by `Chem.SanitizeMol`; failures → `None` → dropped.
Duplicates are **not** removed at generation time.

## Checkpoint flow
```
outputs/pretrain/logs/
  configs.json     architecture spec; re-read on resume; ⚠ .gitignore excludes *.json
  ckpt.params      model.save_parameters()      27.9 MB = 6,986,506 × 4 B
  trainer.status   trainer.save_states()        55.9 MB = Adam m + v
  log.out          "step\ttime(h)\tloss\tlr"    ⚠ time is MINUTES, header is wrong

resume: is_continuous = all three files exist
        → read log.out last (or second-to-last if "Training finished") record
        → global_counter = step ; t0 = time.time() - t_final*60
        → load_parameters + trainer.load_states ; open log.out in append mode

fine-tuned (planned): outputs/CTDGD/Dataset{i}/model/{ckpt.params, configs.json+N_C}
   ⚠ Evaluation_metrics.ipynb instead reads outputs/finetune/log.out — inconsistent
transfer: load_parameters(pretrain, allow_missing=True, ignore_extra=True)
   carries ≈6.99 M of ≈8.08 M params (86%); new: dense_policy_0 + linear_c[0..5] (≈1.1 M)
   nothing is frozen — gluon.Trainer optimises every parameter
```

## Parameter budget (verified against the run's printout)
```
embedding_atom 1,040 · embedding_mask 48 · GraphConv×6 878,080 · BN 2,432 · bn_skip 3,456
linear_skip 222,208 · dense 133,120 · policy_0 65 · Policy 231,817 · GRU 5,514,240
TOTAL 6,986,506   ← matches "Number of parameters in the model: 6986506"
```
The GRU is 79% of the model.

## Does NOT exist
No attention/transformer · no property conditioning · no toxicity input to the network ·
no 3D/pocket geometry · **no beam search** · no multi-target · no VAE/GAN/RL ·
no valence checking during decoding · no multi-GPU.
