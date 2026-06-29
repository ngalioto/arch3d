# Hyperedge prediction — end-to-end workflow

This guide reproduces the hyperedge-prediction experiments from raw public data to a
trained model. It covers three cell types — **GM12878**, **BJ fibroblast**, and
**IR fibroblast** — but the steps generalize to any experiment.

## What the model needs (and where it comes from)

For each experiment, training reads two things from one directory:

```
multiway Pore-C contacts ─► .clusters ─► [process_clusters] ─► edge_list.npy
                                                                    │
                                                  [generate_kmers]  ▼
                                              all_{3,4,5}_counter.npy   ◄── HYPEREDGES
                                              all_{3,4,5}_freq_counter.npy

Hi-C contact matrix (.mcool, 5 kb) ─► [toeplitz_normalize] ─► O/E .npz
                                                                    │
                            [generate_embeddings] + pretrained ckpt ▼
                                              pretrained_embeddings.pt  ◄── NODE FEATURES
```

The two arrows meet in the training datamodule: a hyperedge is a set of node ids
(genomic bins at **100 kb**), and each node id indexes a row of
`pretrained_embeddings.pt`. **Everything must be hg38 and use the same 100 kb node
resolution** so the indices line up.

The Hi-C can be **real** (downloaded) or **virtual** (clique-expansion of the Pore-C
clusters, built by `process_clusters.py`). The published paper used virtual Hi-C; this
workflow uses real Hi-C for GM12878 and BJ, and virtual Hi-C for IR.

## Data sources

| Experiment | Pore-C (hyperedges) | Hi-C (node features) | Pretrained model |
|---|---|---|---|
| GM12878_deshpande_2022 | GEO **GSM4490689** (GSE149117), NlaIII `fragment_alignments` | 4DN **4DNFI46OLTQE** (in situ Hi-C, mcool) | HF `ngalioto/ARCH3D` `pretraining.ckpt` |
| BJ_fibroblast | GEO **GSM6505166** (GSE211897) | GEO **GSM2142400** / SRA **SRX1741600** → build mcool from reads | same |
| IR_fibroblast | GEO **GSM6505165** (GSE211897) | virtual (from its own Pore-C) | same |

Notes: NlaIII is chosen per the [Deshpande paper](https://www.nature.com/articles/s41587-022-01289-z)
(highest contact density). `fragment_alignments` (not `contacts.csv`) preserves the full
multi-way concatemer as a hyperedge. There is no ready 5 kb BJ mcool on a portal, so it
is built from raw reads.

## Prerequisites

```bash
python -m pip install .                 # makes `digitalcell` importable + deps
# digitalcell/data/constants.py must be present (genome ref helpers; provides hg38 coords)
```
A GPU is needed for `generate_embeddings` and training. The BJ read-processing step
(`process_bj_hic.sh`) needs `bwa-mem2`, `pairtools`, `cooler`, `samtools`, and — for fast
ENA downloads — `aria2c`; the script **expects** this toolchain under `/data/arch3d/tools`
(no sudo required): a venv with `pip install pairtools cooler`, the precompiled `bwa-mem2`
binary, and system `samtools`/`aria2c`. The BJ/IR cluster conversion needs `mat-io`
(`pip install mat-io`) to read the MATLAB `table` `.mat` files.

## Step 0 — Pretrained checkpoint

```bash
hf download ngalioto/ARCH3D pretraining.ckpt --local-dir /data/arch3d/checkpoints
```

## Step 1 — Pore-C → `.clusters`

A `.clusters` file has one line per concatemer: `read_id<TAB>chrN:pos<TAB>chrN:pos ...`
(autosomes chr1–22). Place each experiment's file at
`/data/arch3d/raw/porec_clusters/<EXP>/<EXP>.clusters`.

**GM12878** (pore_c `fragment_alignments.csv.gz`, per-monomer rows):
```bash
python digitalcell/tasks/hyperedge/porec_fragalign_to_clusters.py \
  /data/arch3d/raw/porec/GM12878_deshpande_2022/*NlaIII*fragment_alignments.csv.gz \
  -o /data/arch3d/raw/porec_clusters/GM12878_deshpande_2022/GM12878_deshpande_2022.clusters
```
(groups monomers by `read_name`, keeps autosomal fragments, emits each concatemer as one hyperedge.)

**BJ / IR** (MATLAB `table` `.mat`): these store a MATLAB `table` (MCOS) object that
`scipy.io.loadmat`/`pymatreader` cannot decode. Use **`mat-io`** (`pip install mat-io`),
which reads the table directly. The tables are *pairwise* contacts
(`read_id, chrA, posA, chrB, posB`; IR also has `expr` = replicate 1..4), i.e. the
all-vs-all expansion of each concatemer, so the converter groups by read id and unions the
A/B loci to rebuild the hyperedge:
```bash
# (optional) inspect the recovered table as CSV
python digitalcell/tasks/hyperedge/porec_mat_to_csv.py \
  /data/arch3d/raw/porec/BJ_fibroblast/GSM6505166_BJ_Population_Pairs_110421.mat -o BJ_porec.csv

# .mat (or the CSV) -> .clusters
python digitalcell/tasks/hyperedge/porec_pairs_to_clusters.py \
  /data/arch3d/raw/porec/BJ_fibroblast/GSM6505166_BJ_Population_Pairs_110421.mat \
  -o /data/arch3d/raw/porec_clusters/BJ_fibroblast/BJ_fibroblast.clusters
# likewise GSM6505165_v1234_np_uniqueIDs.mat -> IR_fibroblast/IR_fibroblast.clusters
```
Chromosomes are integer-encoded (1..22, 23=X, 24=Y); only autosomes chr1..chr22 are emitted.

## Step 2 — Hi-C source

**GM12878** — download the 4DN mcool (4DN file `4DNFI46OLTQE`, multi-resolution incl. 5 kb) from the 4DN public S3:
```bash
wget -O /data/arch3d/raw/hic/GM12878_deshpande_2022/4DNFI46OLTQE.mcool \
  "https://4dn-open-data-public.s3.amazonaws.com/fourfront-webprod/wfoutput/a43f4886-feb8-48c0-ac45-efaf19e6d870/4DNFI46OLTQE.mcool"
# (if the S3 path changes, resolve 4DNFI46OLTQE via data.4dnucleome.org)
```
**BJ** — build a 5 kb mcool from raw reads (bwa-mem2 → pairtools → cooler):
```bash
bash /data/arch3d/tools/process_bj_hic.sh   # -> raw/hic/BJ_fibroblast/BJ_fibroblast.mcool
```
**IR** — none; uses virtual Hi-C built in Step 3 (`HIC_SOURCE=virtual`).

## Step 3 — Preprocess each experiment (counters + embeddings)

`preprocess_hyperedge.sh` runs `process_clusters → generate_kmers → toeplitz_normalize →
generate_embeddings` and writes everything into `experiments/<EXP>/`.

```bash
cd digitalcell/tasks/hyperedge

EXP=GM12878_deshpande_2022 \
CLUSTERS=/data/arch3d/raw/porec_clusters/GM12878_deshpande_2022/GM12878_deshpande_2022.clusters \
HIC_SOURCE=/data/arch3d/raw/hic/GM12878_deshpande_2022/4DNFI46OLTQE.mcool \
bash preprocess_hyperedge.sh

EXP=BJ_fibroblast \
CLUSTERS=/data/arch3d/raw/porec_clusters/BJ_fibroblast/BJ_fibroblast.clusters \
HIC_SOURCE=/data/arch3d/raw/hic/BJ_fibroblast/BJ_fibroblast.mcool \
bash preprocess_hyperedge.sh

EXP=IR_fibroblast \
CLUSTERS=/data/arch3d/raw/porec_clusters/IR_fibroblast/IR_fibroblast.clusters \
HIC_SOURCE=virtual \
bash preprocess_hyperedge.sh
```
Each produces, in `experiments/<EXP>/`: `edge_list.npy`, `all_{3,4,5}_counter.npy`,
`all_{3,4,5}_freq_counter.npy`, `<EXP>.npz`, `pretrained_embeddings.pt`.

## Step 4 — Train / sweep

Paths are already wired in `conf/hyperedge_config.yaml`. Fill `logger` (W&B) fields, then:
```bash
python hyperedge.py --config conf/hyperedge_config.yaml      # single run
bash run_num_layers_sweep.sh                                  # task-head depth sweep (num_layers 1..4)
```
Multi-GPU is automatic on a single node (`strategy: ddp`, `devices: auto`); for multi-node
set `num_nodes` and launch under your cluster's launcher (e.g. torchrun).

## Scripts reference

| Script | Purpose |
|---|---|
| `porec_fragalign_to_clusters.py` | pore_c `fragment_alignments.csv` → `.clusters` (GM12878) |
| `porec_mat_to_csv.py` | MATLAB `table` `.mat` → CSV via `mat-io` (BJ/IR) |
| `porec_pairs_to_clusters.py` | pairwise contact table (.mat/.csv) → `.clusters` (BJ/IR) |
| `/data/arch3d/tools/process_bj_hic.sh` | BJ raw reads (SRA) → 5 kb balanced `.mcool` |
| `preprocess_hyperedge.sh` | clusters + Hi-C → counters + `pretrained_embeddings.pt` |
| `run_num_layers_sweep.sh` | train one model per task-head depth (num_layers 1..4) |

See `/data/arch3d/README.md` for the on-disk data layout on this machine.
