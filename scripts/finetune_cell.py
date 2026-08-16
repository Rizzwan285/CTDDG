"""
MISSING FINE-TUNING STAGE (CTDDG)
---------------------------------
Instructions:
1. Open `pretraining.ipynb`
2. Scroll to the very end of the notebook.
3. Paste all of the code below into a new cell and run it.

This code defines the conditional model (CVanillaMolGen_RNN) that was missing from the training notebook,
loads the unconditional weights, and fine-tunes them on the paired ligand-protein dataset.
"""

import json
import ast
import mxnet as mx
from mxnet import nd, gluon

# -------------------------------------------------------------------------
# 1. DEFINE CONDITIONAL MODEL ARCHITECTURES (Missing from pretraining.ipynb)
# -------------------------------------------------------------------------
class CMoleculeGenerator_RNN(MoleculeGenerator_RNN):
    def __init__(self, N_A, N_B, N_C, D, F_e, F_skip, F_c, Fh_policy, activation, N_rnn, *args, **kwargs):
        super(CMoleculeGenerator_RNN, self).__init__(N_A, N_B, D, F_e, F_skip, F_c, Fh_policy, activation, N_rnn, *args, **kwargs)
        self.N_C = N_C # Number of conditional variables (e.g. 1024 for protein embedding)
        with self.name_scope():
            self.dense_policy_0 = _TwoLayerDense(self.N_C, self.N_A * 3, self.N_A)

    def _policy_0(self, c):
        return nd.softmax(self.dense_policy_0(c), axis=-1)

    def forward(self, *input):
        if self.mode == 'loss' or self.mode == 'likelihood':
            X, A, iw_ids, last_append_mask, NX, NX_rep, action_0, actions, log_p, tox_class_batch, batch_size, iw_size, graph_to_rnn, rnn_to_graph, NX_cum, c = input
            
            # Repeat condition vector for iw_size
            c = nd.repeat(c, repeats=iw_size, axis=0)
            
            init = self._policy_0(c)
            append, connect, end = self._policy(X, A, NX, NX_rep, last_append_mask, graph_to_rnn, rnn_to_graph, NX_cum, c)
            l = self._likelihood(init, append, connect, end, action_0, actions, iw_ids, log_p, batch_size, iw_size, tox_class_batch)
            if self.mode == 'likelihood':
                return l
            else:
                return -l.mean()
        # (decode modes omitted for brevity since this is for training)
        else:
            raise ValueError("Mode must be 'loss' or 'likelihood' during fine-tuning.")

class CVanillaMolGen_RNN(CMoleculeGenerator_RNN):
    def __init__(self, N_A, N_B, N_C, D, F_e, F_h, F_skip, F_c, Fh_policy, activation, N_rnn):
        super(CVanillaMolGen_RNN, self).__init__(N_A, N_B, N_C, D, F_e, F_skip, F_c, Fh_policy, activation, N_rnn, F_h)

    def _build_graph_conv(self, F_h):
        self.F_h = list(F_h) if isinstance(F_h, tuple) else F_h
        self.conv, self.bn, self.linear_c = [], [], []
        for i, (f_in, f_out) in enumerate(zip([self.F_e] + self.F_h[:-1], self.F_h)):
            conv = GraphConv(f_in, f_out, self.N_B + self.D)
            self.conv.append(conv)
            self.register_child(conv)
            if i != len(self.F_h) - 1:
                bn = nn.BatchNorm()
                self.bn.append(bn)
                self.register_child(bn)
                # This is the key difference: conditioning projection for each GCN layer
                linear_c = nn.Dense(f_out, use_bias=False, in_units=self.N_C)
                self.linear_c.append(linear_c)
                self.register_child(linear_c)

    def _graph_conv_forward(self, X, A, c):
        X_out = [X]
        for i, conv in enumerate(self.conv):
            if i != len(self.F_h) - 1:
                bn = self.bn[i]
                linear_c = self.linear_c[i]
                # Add protein conditioning to GCN layer
                h = conv(self.activation(bn(X)), A)
                h_c = nd.repeat(linear_c(c), repeats=X.shape[0] // c.shape[0], axis=0) # Match atom dimension
                X_out.append(h + h_c)
            else:
                X_out.append(conv(X, A))
        X_out = nd.concat(*X_out[1:], dim=1)
        return self.activation(self.linear_skip(self.activation(self.bn_skip(X_out))))

    def _policy(self, X, A, NX, NX_rep, last_append_mask, graph_to_rnn, rnn_to_graph, NX_cum, c):
        X = self.embedding_atom(X) + self.embedding_mask(last_append_mask)
        X = self._graph_conv_forward(X, A, c)
        X = self.dense(X)
        X_mol = self._rnn_train(X, NX, NX_rep, graph_to_rnn, rnn_to_graph, NX_cum)
        append, connect, end = self.policy_h(X, NX, NX_rep, X_mol)
        return append, connect, end


# -------------------------------------------------------------------------
# 2. CONDITIONAL DATA LOADER
# -------------------------------------------------------------------------
class CMolRNNLoader(MolRNNLoader):
    def _collate_fn(self, batch):
        # batch is a list of tuples: (smiles, embedding_list)
        smiles_batch = [item[0] for item in batch]
        embed_batch = [item[1] for item in batch]
        
        # Append dummy class ' 1' to reuse existing process_single
        smiles_with_class = [f"{s} 1" for s in smiles_batch]
        
        # Call parent's collate (which calls MolLoader's collate under the hood)
        result_out = super(CMolRNNLoader, self)._collate_fn(smiles_with_class)
        
        # Add condition embedding tensor
        c = nd.array(embed_batch, ctx=mx.gpu(), dtype='float32')
        result_out.append(c)
        return result_out
        
    @staticmethod
    def from_numpy_to_tensor(record):
        # The last element is 'c' (the condition), the rest are from MolRNNLoader
        c = record[-1]
        parent_record = record[:-1]
        output = MolRNNLoader.from_numpy_to_tensor(parent_record)
        output.append(c)
        return output

# -------------------------------------------------------------------------
# 3. FINE-TUNING SCRIPT
# -------------------------------------------------------------------------
def run_finetuning(dataset_index=1):
    import time
    
    # 1. Paths
    # Assuming config.py is in the Python path, or adapt these manually:
    PROJECT_ROOT = "/workspace" # Adjust if necessary
    data_path = f"{PROJECT_ROOT}/data/bindingdb/train_dataset/train_4_org_1042_104/d{dataset_index}_tr_cdgcn.txt"
    pretrain_ckpt = f"{PROJECT_ROOT}/outputs/pretrain/logs"
    finetune_model_dir = f"{PROJECT_ROOT}/outputs/CTDGD/Dataset{dataset_index}/model"
    
    os.makedirs(finetune_model_dir, exist_ok=True)
    
    # 2. Load Conditional Dataset
    print(f"Loading {data_path}...")
    dataset = []
    with open(data_path, "r") as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                smi = parts[0]
                embed = ast.literal_eval(parts[1]) # parse JSON string to list
                dataset.append((smi, embed))
                
    N_C = len(dataset[0][1]) # Embedding dimension (should be 1024)
    print(f"Loaded {len(dataset)} paired samples. Embedding size: {N_C}")
    
    # 3. Setup Model and Load Pretrained Weights
    with open(os.path.join(pretrain_ckpt, 'configs.json')) as f:
        configs = json.load(f)
        
    # Inject N_C into configs for Generating_samples.ipynb compatibility
    configs['N_C'] = N_C
    
    model = CVanillaMolGen_RNN(
        get_mol_spec().num_atom_types, 
        get_mol_spec().num_bond_types, 
        D=2, 
        **configs
    )
    
    ctx = mx.gpu()
    # Initialize all parameters (crucial for the new dense_policy_0 and linear_c layers)
    model.collect_params().initialize(mx.init.Xavier(), force_reinit=True, ctx=ctx)
    
    # Load VanillaMolGen_RNN weights where they match, ignore missing conditioning layers
    pretrained_params_path = os.path.join(pretrain_ckpt, 'ckpt.params')
    print(f"Loading pretrained weights from {pretrained_params_path}")
    model.load_parameters(pretrained_params_path, ctx=ctx, allow_missing=True, ignore_extra=True)
    
    model.mode = 'loss'
    opt = mx.optimizer.Adam(learning_rate=1e-4, clip_gradient=10.0)
    trainer = gluon.Trainer(model.collect_params(), opt)
    
    # 4. DataLoader
    batch_size = 16
    sampler_train = BalancedSampler(cost=[len(item[0]) for item in dataset], batch_size=batch_size)
    loader_train = CMolRNNLoader(dataset, batch_size_sampler=sampler_train, num_workers=0, k=10, p=0.9)
    
    # 5. Training Loop
    iterations = 50000 # Adjust as needed
    log_freq = 100
    save_freq = 5000
    
    print("Starting Fine-tuning...")
    it_train = iter(loader_train)
    start_time = time.time()
    
    for i in range(1, iterations + 1):
        try:
            batch = next(it_train)
        except StopIteration:
            it_train = iter(loader_train)
            batch = next(it_train)
            
        record = CMolRNNLoader.from_numpy_to_tensor(batch)
        
        with autograd.record():
            loss = model(*record)
        loss.backward()
        trainer.step(batch_size=batch_size)
        
        if i % log_freq == 0:
            avg_loss = loss.mean().asscalar()
            print(f"Step {i} | Loss: {avg_loss:.4f} | Time: {time.time() - start_time:.2f}s")
            start_time = time.time()
            
        if i % save_freq == 0 or i == iterations:
            model.save_parameters(os.path.join(finetune_model_dir, 'ckpt.params'))
            with open(os.path.join(finetune_model_dir, 'configs.json'), 'w') as f:
                json.dump(configs, f)
            print(f"--> Saved checkpoint at step {i}")

# Run it
# run_finetuning(dataset_index=1)
