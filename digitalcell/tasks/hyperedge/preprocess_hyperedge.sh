#!/usr/bin/env bash
#
# End-to-end preprocessing for ONE hyperedge experiment. Turns a `.clusters` file
# (multiway Pore-C concatemers) plus a Hi-C source into the per-experiment inputs
# the training datamodule expects.
#
# Chain:
#   .clusters --process_clusters.py--> edge_list.npy  [+ optional virtual Hi-C output.mcool]
#   edge_list.npy --generate_kmers.py--> all_{3,4,5}_counter.npy + all_{3,4,5}_freq_counter.npy
#   Hi-C .mcool --toeplitz_normalize.py--> <EXP>.npz (5 kb observed/expected)
#   <EXP>.npz (+ pretrained ckpt) --generate_embeddings.py--> pretrained_embeddings.pt
#
# Output dir  $EXPERIMENTS_ROOT/$EXP/  == datamodule.data_dir[i] / embeddings_path[i].
#
# REQUIREMENTS: the `digitalcell` package importable (`python -m pip install .`) and
# its deps (torch, cooler, h5py, scipy, pandas, numpy<2). The embedding step runs the
# pretrained model -> use a GPU (it falls back to slow CPU). Intended for the cluster.
#
# Usage (real Hi-C, e.g. GM12878 / BJ):
#   EXP=GM12878_deshpande_2022 \
#   CLUSTERS=/data/arch3d/raw/porec_clusters/GM12878_deshpande_2022/GM12878_deshpande_2022.clusters \
#   HIC_SOURCE=/data/arch3d/raw/hic/GM12878_deshpande_2022/4DNFI46OLTQE.mcool \
#   bash preprocess_hyperedge.sh
#
# Usage (virtual Hi-C built from the clusters themselves, e.g. IR):
#   EXP=IR_fibroblast \
#   CLUSTERS=/data/arch3d/raw/porec_clusters/IR_fibroblast/IR_fibroblast.clusters \
#   HIC_SOURCE=virtual \
#   bash preprocess_hyperedge.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

EXP="${EXP:?set EXP (experiment name, e.g. GM12878_deshpande_2022)}"
CLUSTERS="${CLUSTERS:?set CLUSTERS (path to the <EXP>.clusters file)}"
HIC_SOURCE="${HIC_SOURCE:?set HIC_SOURCE (path to a real Hi-C .mcool, or the literal word 'virtual')}"
CKPT="${CKPT:-/data/arch3d/checkpoints/pretraining.ckpt}"
EXPERIMENTS_ROOT="${EXPERIMENTS_ROOT:-/data/arch3d/hyperedge/experiments}"
TASK_RES="${TASK_RES:-100000}"   # hyperedge node resolution (must equal dataset.resolution)
HIC_RES="${HIC_RES:-5000}"       # resolution for Toeplitz O/E normalization (model input)
PYTHON="${PYTHON:-python}"

CL_DIR="$(cd "$(dirname "$CLUSTERS")" && pwd)"
CL_FILE="$(basename "$CLUSTERS")"
OUT="$EXPERIMENTS_ROOT/$EXP"
mkdir -p "$OUT"

# process_clusters derives its save subdir from basename(parent_dir); that must be $EXP
# so its outputs land in $OUT (== $EXPERIMENTS_ROOT/$EXP).
if [ "$(basename "$CL_DIR")" != "$EXP" ]; then
  echo "ERROR: the .clusters file must live in a directory named '$EXP' (got '$(basename "$CL_DIR")')." >&2
  echo "       e.g. .../porec_clusters/$EXP/$EXP.clusters" >&2
  exit 1
fi

if [ "$HIC_SOURCE" = "virtual" ]; then BUILD_VIRTUAL=True; else BUILD_VIRTUAL=False; fi

echo "===== [$EXP] preprocessing start $(date) ====="
echo "  clusters     : $CL_DIR/$CL_FILE"
echo "  hic source   : $HIC_SOURCE  (build virtual Hi-C: $BUILD_VIRTUAL)"
echo "  output dir   : $OUT"

# 1) clusters -> edge_list.npy (+ virtual Hi-C output.mcool when HIC_SOURCE=virtual)
echo "[1/5] process_clusters ..."
"$PYTHON" - <<PYEOF
from digitalcell.tasks.hyperedge.process_clusters import main
main(parent_dir="$CL_DIR", parent_save_dir="$EXPERIMENTS_ROOT",
     fname="$CL_FILE", resolution=$TASK_RES, create_mcool_files=$BUILD_VIRTUAL)
PYEOF

# 2) edge_list.npy -> all_{k}_counter.npy / all_{k}_freq_counter.npy
echo "[2/5] generate_kmers (k=3,4,5) ..."
"$PYTHON" - <<PYEOF
from pathlib import Path
from digitalcell.tasks.hyperedge.generate_kmers import Config, main
main(Config(max_cluster_size=25, k_list=[3, 4, 5], temp_dir=Path("$OUT"),
            min_freq_cutoff=2, resolution=$TASK_RES))
PYEOF

# 3) choose the Hi-C mcool (real, or the virtual one just built)
if [ "$HIC_SOURCE" = "virtual" ]; then HIC_MCOOL="$OUT/output.mcool"; else HIC_MCOOL="$HIC_SOURCE"; fi
[ -e "$HIC_MCOOL" ] || { echo "ERROR: Hi-C mcool not found: $HIC_MCOOL" >&2; exit 1; }
echo "[3/5] Hi-C mcool: $HIC_MCOOL"

# 4) Toeplitz observed/expected normalization at $HIC_RES -> $OUT/$EXP.npz
echo "[4/5] toeplitz_normalize ..."
"$PYTHON" digitalcell/data/toeplitz_normalize.py "$HIC_MCOOL" "$OUT" --save_name "$EXP" --resolution "$HIC_RES"

# 5) embeddings from the pretrained model at $TASK_RES; rename to the name the config expects
echo "[5/5] generate_embeddings ..."
"$PYTHON" digitalcell/scripts/generate_embeddings.py \
  --ckpt_path "$CKPT" --data_file "$OUT/$EXP.npz" --resolution "$TASK_RES" --save_dir "$OUT"
mv -f "$OUT/embeddings.pt" "$OUT/pretrained_embeddings.pt"

echo "===== [$EXP] done $(date). Outputs in $OUT: ====="
ls -lh "$OUT"
