# Trexplorer Super - Coronary Artery Project Workflow

## Project Context

**Goal:** Train Trexplorer Super for coronary artery centerline tracking from CT scans.

**Constraint:** University dataset cannot be downloaded locally due to legal restrictions. University compute (RTX 2080 Super, 8GB) is limited. Local hardware (RTX 5060 TI, 16GB) is more capable.

**Strategy:** Pretrain on public synthetic data locally, then fine-tune on university coronary dataset when access is restored.

---

## Current Status

- [x] Analyzed Trexplorer Super codebase
- [x] Created JSON-to-Trexplorer conversion script for university dataset
- [x] Created low-memory config for 2080 Super
- [ ] Download SynTRX dataset from Zenodo
- [ ] Set up and organize SynTRX dataset
- [ ] Pretrain model on SynTRX (local)
- [ ] Convert university dataset (when access restored)
- [ ] Fine-tune on coronary data (university)

---

## Hardware Overview

| Location | GPU | VRAM | Batch Size | Role |
|----------|-----|------|------------|------|
| Local | RTX 5060 TI | 16 GB | 8 (default) | Pretraining on SynTRX |
| University | RTX 2080 Super | 8 GB | 2 (reduced) | Fine-tuning on coronary |

---

## Phase 1: Local Pretraining on SynTRX

### Step 1: Download Dataset

Download from: **https://zenodo.org/records/15888958**

The SynTRX dataset includes:
- 500 synthetic samples
- Images, masks, and centerline annotations
- Split: 368 train / 32 val / 100 test

Save to: `./data/syntrx/` (or extract there)

### Step 2: Organize Dataset

Edit `src/trxsuper/datasets/utils/organize_data.py`:
```python
dataset = "syntrx"  # Line 7
data_dir = "./data/syntrx_raw"  # Path to downloaded/extracted data
dst_dir = "./data/syntrx"  # Destination
```

Run:
```bash
cd /home/noahschuetz/GitHub/Trexplorer-Super
python src/trxsuper/datasets/utils/organize_data.py
```

### Step 3: Generate Validation Metadata

Edit `src/trxsuper/datasets/utils/generate_val_sub_vol_file.py`:
```python
dataset = 'syntrx'  # Line 96
annot_dir = './data/syntrx/annots_val_sub_vol'  # Line 97
out_dir = './data/syntrx/'  # Line 98
```

Run:
```bash
python src/trxsuper/datasets/utils/generate_val_sub_vol_file.py
```

### Step 4: Verify Directory Structure

After setup, you should have:
```
data/syntrx/
├── annots_train/          (368 .pickle files)
├── annots_val/            (32 .pickle files)
├── annots_val_sub_vol/    (32 .pickle files)
├── annots_test/           (100 .pickle files)
├── images_train/          (368 .nii.gz files)
├── images_val/            (32 .nii.gz files)
├── images_val_sub_vol/    (32 .nii.gz files)
├── images_test/           (100 .nii.gz files)
├── masks_train/           (368 .nii.gz files)
├── masks_val/             (32 .nii.gz files)
├── masks_val_sub_vol/     (32 .nii.gz files)
├── masks_test/            (100 .nii.gz files)
└── annots_val_sub_vol.pickle
```

### Step 5: Configure Training

Edit `cfgs/train.yaml`:
```yaml
data_dir: data/syntrx
output_dir: models/syntrx_pretrain
```

### Step 6: Start Pretraining

```bash
# Single GPU training
python src/train_trx.py

# Monitor GPU usage
watch -n 1 nvidia-smi
```

Training runs for 12,000 epochs by default. Checkpoints saved every 1,000 epochs.

### Step 7: Save Pretrained Weights

When pretraining completes (or at a good checkpoint), the weights will be in:
```
models/syntrx_pretrain/checkpoint.pth
```

Push to your git repo or save separately for transfer to university.

---

## Phase 2: University Dataset Conversion

### University Dataset Format

Your data format:
- `*.img.nii.gz` - CT images
- `*.label.nii.gz` - Segmentation masks
- `*.graph.json` - Graph annotations

JSON structure:
```json
{
  "directed": true,
  "graph": {"coordinateSystem": "RAS"},
  "nodes": [{"id": 1, "pos": [x, y, z], "is_root": false}, ...],
  "edges": [{"source": 1, "target": 2, "skeletons": [[x,y,z], ...], "length": 10.5}, ...]
}
```

### Conversion Script Location

Created at: `src/trxsuper/datasets/utils/convert_json_to_trexplorer.py`

### Conversion Commands

**Step 1: Convert JSON to pickle**
```bash
python src/trxsuper/datasets/utils/convert_json_to_trexplorer.py convert \
    --input_dir /path/to/json_files \
    --mask_dir /path/to/masks \
    --output_dir /path/to/annotations \
    --spacing 0.5 \
    --json_suffix .graph.json \
    --mask_suffix .label.nii.gz
```

**Step 2: Organize into Trexplorer structure**
```bash
python src/trxsuper/datasets/utils/convert_json_to_trexplorer.py organize \
    --data_dir /path/to/your/data \
    --output_dir ./data/coronary \
    --train_ratio 0.7 \
    --val_ratio 0.15 \
    --test_ratio 0.15
```

**Step 3: Generate validation metadata**
```bash
# Edit generate_val_sub_vol_file.py to point to coronary dataset
python src/trxsuper/datasets/utils/generate_val_sub_vol_file.py
```

**Test single file conversion:**
```bash
python src/trxsuper/datasets/utils/convert_json_to_trexplorer.py single \
    --json sample.graph.json \
    --mask sample.label.nii.gz \
    --output sample.pickle \
    --spacing 0.5
```

### What the Conversion Does

1. **Parses JSON graph** - nodes, edges, skeleton points
2. **Estimates radius** - via distance transform on segmentation mask
3. **Computes node labels** - endpoint (0), intermediate (1), bifurcation (2)
4. **Builds bigtree hierarchy** - required tree structure
5. **Expands skeletons** - edge skeleton points become intermediate nodes
6. **Computes trajectories** - paths from root to each endpoint
7. **Creates NetworkX graph** - for evaluation metrics

---

## Phase 3: Fine-tuning on University Hardware

### Low Memory Configuration

Created at: `cfgs/train_low_mem.yaml`

Key settings for 8GB VRAM:
```yaml
batch_size: 2      # Reduced from 8
amp: true          # Mixed precision (keep enabled)
cache_num: 16      # Reduced from 32
```

### Fine-tuning Commands

```bash
# Load pretrained weights and fine-tune
python src/train_trx.py with cfgs/train_low_mem.yaml \
    data_dir=data/coronary \
    output_dir=models/coronary_finetune \
    resume=models/syntrx_pretrain/checkpoint.pth
```

### If Out of Memory (OOM)

Try progressively:
1. `batch_size: 1`
2. `traj_train_len: 4` (down from 6)
3. `hidden_dim: 256` (down from 512, affects capacity)

---

## Files Created/Modified

| File | Purpose |
|------|---------|
| `src/trxsuper/datasets/utils/convert_json_to_trexplorer.py` | Converts your JSON format to Trexplorer pickle |
| `cfgs/train_low_mem.yaml` | Low memory config for 8GB GPUs |
| `WORKFLOW_PLAN.md` | This document |

---

## Trexplorer Pickle Format Reference

The conversion script produces pickle files with this structure:

```python
{
    'branches': [bigtree.Node, ...],           # Tree structures (root nodes)
    'bifur_ids': [['0-5', '1-3'], ...],        # Bifurcation node IDs per tree
    'endpts_ids': [['0-12', '2-8'], ...],      # Endpoint node IDs per tree
    'root_id': ['0-0', ...],                   # Root node IDs
    'interm_ids': [['0-1', '0-2', ...], ...],  # Intermediate node IDs per tree
    'branch_ids': [['0', '1', '2'], ...],      # Branch IDs per tree
    'all_ids': [['0-0', '0-1', ...], ...],     # All node IDs per tree
    'num_points': [[15, 8, 12], ...],          # Points per branch per tree
    'networkx': [nx.DiGraph, ...],             # NetworkX graphs for evaluation
    'trajectories': [[                          # Paths from root to endpoints
        {'path': ['0-0', '0-1', ...], 'bifur_ids': ['0-5'], 'endpt_id': '0-12'},
        ...
    ], ...]
}
```

Node attributes in bigtree:
- `position`: [x, y, z] coordinates
- `radius`: vessel radius (from distance transform)
- `label`: 0=endpoint, 1=intermediate, 2=bifurcation
- `node_name`: format "branch_id-point_id" (e.g., "0-3")

---

## Training Configuration Reference

Key parameters in `cfgs/train.yaml`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `batch_size` | 8 | Samples per batch |
| `epochs` | 12000 | Total training epochs |
| `lr` | 0.0001 | Learning rate |
| `seq_len` | 10 | Sequence length per trajectory |
| `traj_train_len` | 6 | Sub-samples per trajectory |
| `sub_vol_size` | 64 | Patch size (64³) |
| `num_queries` | 196 | Main trajectory queries |
| `num_bifur_queries` | 26 | Bifurcation queries |
| `val_interval` | 500 | Validation every N epochs |
| `save_model_interval` | 1000 | Checkpoint every N epochs |
| `amp` | true | Mixed precision training |

---

## Evaluation

```bash
# Evaluate trained model
python src/train_trx.py with eval \
    data_dir=data/coronary \
    resume=models/coronary_finetune/checkpoint.pth
```

Uses `cfgs/eval.yaml` overrides for inference settings.

---

## Troubleshooting

### CUDA Out of Memory
- Reduce `batch_size` (most effective)
- Ensure `amp: true` is set
- Reduce `cache_num` if CPU RAM is also limited

### No Pickle Files Generated
- Check JSON suffix matches your files (default: `.graph.json`)
- Check mask suffix matches (default: `.label.nii.gz`)
- Verify paths are correct

### Poor Convergence After Transfer
- Coronary arteries may need domain-specific tuning
- Consider `window_input` settings for intensity windowing
- May need more training epochs for fine-tuning

### Conversion Errors
- Ensure all nodes are connected (no isolated nodes)
- Verify at least one node has `is_root: true`
- Check coordinate system matches expectations

---

## Quick Reference Commands

```bash
# Pretrain on SynTRX (local, 16GB GPU)
python src/train_trx.py

# Fine-tune on coronary (university, 8GB GPU)
python src/train_trx.py with cfgs/train_low_mem.yaml \
    resume=models/syntrx_pretrain/checkpoint.pth

# Convert university dataset
python src/trxsuper/datasets/utils/convert_json_to_trexplorer.py convert \
    --input_dir /path/to/json --mask_dir /path/to/masks --output_dir ./annotations

# Evaluate
python src/train_trx.py with eval resume=models/checkpoint.pth

# Monitor GPU
watch -n 1 nvidia-smi
```

---

## Links

- **Dataset:** https://zenodo.org/records/15888958
- **Paper:** https://arxiv.org/abs/2507.10881
- **Original Trexplorer:** https://github.com/RomStriker/Trexplorer
