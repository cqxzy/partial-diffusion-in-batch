# RFdiffusion Partial Diffusion Batch Runner

Batch-run **RFdiffusion partial diffusion** on many binder–target complex PDBs using the official Docker image.

This script:

* Reads many `.pdb` complexes from a folder
* Fixes the **target chain** (A/B/…)
* Auto-infers **binder chain** (or you can specify it)
* Computes binder length automatically
* Builds strict contig for partial diffusion:

  * `contigmap.contigs=[{binder_len}-{binder_len}/0 {target_chain}{start}-{end}]`
* Runs RFdiffusion via Docker
* Writes logs + a TSV summary
* Optionally flattens all generated PDBs into one folder

---

## Requirements

* Python 3
* `gemmi` (Python package)
* Docker with GPU support (NVIDIA runtime)
* RFdiffusion docker image (default): `rosettacommons/rfdiffusion:latest`

Install gemmi:

```bash
pip install gemmi
```

---

## Quick Start

```bash
python run_partialdiffusion_inbatch.py \
  --input-dir /data/rfdiffusion-run/xing-work/input/partialdiffusion/test \
  --output-dir /data/rfdiffusion-run/xing-work/output/0215/partialdiffusion \
  --target-chain B \
  --num-designs 20 \
  --partial-T 20 \
  --hotspots 94,127,139,173
```

---

## How it works (important)

### Chain selection

* `--target-chain` is required.
* Binder chain:

  * If `--binder-chain` is set → use it
  * Else → pick the **longest protein chain** that is **not** target
  * If there is a tie (same length) → script fails and asks you to set `--binder-chain`

### Residue numbering / contig

* The script reads the **actual residue numbers in the input PDB** to build `B101-314` style ranges.
* It requires the target chain residue numbering to be **contiguous** (`max-min+1 == count`), otherwise it will stop and ask you to renumber (see below).

### Renumber options (recommended when PDB numbering is weird)

* `--renumber-target-to-1`: renumber only target chain protein residues to `1..N`
* `--renumber-all-to-1`: renumber all protein chains to `1..N` per chain
  Renumbered inputs are written to:
* `output_dir/_tmp_inputs/<pdb_stem>.pdb`

---

## CLI Options

Common:

* `--input-dir`: folder with input `.pdb`
* `--output-dir`: output root folder
* `--target-chain`: target chain ID (e.g. `A` or `B`)
* `--binder-chain`: optional binder chain override
* `--num-designs`: `inference.num_designs` (default `10`)
* `--partial-T`: `diffuser.partial_T` (default `20`)
* `--noise-scale-ca`: `denoiser.noise_scale_ca` (default `0.5`)
* `--noise-scale-frame`: `denoiser.noise_scale_frame` (default `0.5`)
* `--cuda-visible-devices`: passed into docker env `CUDA_VISIBLE_DEVICES` (default `0`)
* `--docker-image`: docker image (default `rosettacommons/rfdiffusion:latest`)

Hotspots:

* `--hotspots`: comma-separated list. Supported tokens:

  * `94`, `94A`, `B94`, `B94A`
* `--remap-hotspot-chain`: force hotspot chain IDs to the `--target-chain`

Execution / utilities:

* `--dry-run`: only print/log commands, do not execute
* `--skip-gpu-preflight`: skip docker/GPU torch availability check
* `--max-workers`: parallel jobs (default `1`)
* `--flatten`: copy generated `.pdb` files into `output_dir/ALL_PDB/`

---

## Output Layout

Default (per input PDB):

```
output_dir/
  run.log
  run_summary.tsv
  <pdb_stem>/
    <pdb_stem>_*.pdb / *.trb / (RFdiffusion outputs)
```

If `--flatten` is set:

```
output_dir/
  ALL_PDB/
    <pdb_stem>__<generated_name>.pdb
```

Logs:

* `run.log`: readable run log
* `run_summary.tsv`: one line per job, includes contig/hotspots/docker command and status

---

## Examples

### 1) Basic run (auto binder chain)

```bash
python run_partialdiffusion_inbatch.py \
  --input-dir ./inputs \
  --output-dir ./out \
  --target-chain B
```

### 2) Specify binder chain

```bash
python run_partialdiffusion_inbatch.py \
  --input-dir ./inputs \
  --output-dir ./out \
  --target-chain B \
  --binder-chain A
```

### 3) Hotspots (numbers only → mapped to target chain)

```bash
python run_partialdiffusion_inbatch.py \
  --input-dir ./inputs \
  --output-dir ./out \
  --target-chain B \
  --hotspots 94,127,139,173
```

### 4) Force hotspot chain to target chain (even if you wrote A94)

```bash
python run_partialdiffusion_inbatch.py \
  --input-dir ./inputs \
  --output-dir ./out \
  --target-chain B \
  --hotspots A94,A127 \
  --remap-hotspot-chain
```

### 5) Fix “target not starting at 1 / numbering not contiguous”

```bash
python run_partialdiffusion_inbatch.py \
  --input-dir ./inputs \
  --output-dir ./out \
  --target-chain B \
  --renumber-target-to-1
```

### 6) Flatten all output PDBs

```bash
python run_partialdiffusion_inbatch.py \
  --input-dir ./inputs \
  --output-dir ./out \
  --target-chain B \
  --flatten
```

### 7) Dry run

```bash
python run_partialdiffusion_inbatch.py \
  --input-dir ./inputs \
  --output-dir ./out \
  --target-chain B \
  --dry-run
```

---

## Notes

* This script counts **protein residues only** (waters/ions/ligands ignored).
* If target numbering is non-contiguous (gaps/insertion issues), use:

  * `--renumber-target-to-1` (usually enough) or `--renumber-all-to-1`
* Parallel (`--max-workers > 1`) may easily OOM a single GPU. Use carefully.

---
This work is done at [Jie-Yang Lab](https://jieyang-lab.com/), a young lab at UVA, [Department of Biochemistry-Molecular Genetics](https://med.virginia.edu/bmg/).
Supervised by [Prof. Jie Yang](https://med.virginia.edu/faculty/faculty-listing/wfw7nc/) 
