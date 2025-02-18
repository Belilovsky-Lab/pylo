# Installation Guide

## Prerequisites
- Set CUDA_HOME environment variable (if using CUDA)

## Installation Options

### Basic Installation (without CUDA)
```bash
pip install .
```

### CUDA-enabled Installation (This may take a while)
```bash
pip install . --config-settings="--build-option=--cuda"
```

# Install MuP with patch

"""
git clone https://github.com/microsoft/mup
cd mup
pip install -e .
"""


### Apply the following patch to the MuP repository
```
(base) btherien@therien-ws:~/github/mup$ git remote -v
origin  https://github.com/microsoft/mup (fetch)
origin  https://github.com/microsoft/mup (push)
(base) btherien@therien-ws:~/github/mup$ git diff mup/layer.py
diff --git a/mup/layer.py b/mup/layer.py
index 518a33b..7d31820 100644
--- a/mup/layer.py
+++ b/mup/layer.py
@@ -53,7 +53,7 @@ class MuReadout(Linear):

     def forward(self, x):
         return super().forward(
-            self.output_mult * x / self.width_mult())
+             x ) * (self.output_mult / self.width_mult())


 class MuSharedReadout(MuReadout):
(base) btherien@therien-ws:~/github/mup$
```




# FOR Xiaolong

## Run mlp d=3 w=2048 on random data from seed 42 in pytorch
```
TFDS_DATA_PATH="/home/datasets/tensorflow_datasets" CUDA_VISIBLE_DEVICES=1 DATA_PATH='.' python examples/mumlp.py
```

## Run mlp d=3 w=2048 on random data from seed 42 in jax
```
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0  mpirun -np 1 --bind-to none --allow-run-as-root python src/main.py                 --config config/meta_test/image-mlp-fmst_fedlagg-adafac.py                 --task mumlp-w2048-d3_random-64x64x3                 --optimizer mup_small_fc_mlp                 --name_suffix _m_mup_final                 --wandb_checkpoint_id eb-lab/mup-meta-training/woz3g9l0                 --local_batch_size 4096                 --test_project mup-meta-testing                 --num_runs 5                 --num_inner_steps 5000                 --needs_state                 --gradient_accumulation_steps 1                 --adafac_step_mult 0.01                 --test_interval 100                 --use_bf16 --selected_checkpoint global_step5000.pickle
```
