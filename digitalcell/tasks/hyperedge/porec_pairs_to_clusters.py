#!/usr/bin/env python3
"""
Convert a PAIRWISE Pore-C contact table into a `.clusters` file for process_clusters.py.

Input (CSV, or .mat read via mat-io) with columns:
    read_id, chrA, posA, chrB, posB        (BJ: GSM6505166)
    expr, read_id, chrA, posA, chrB, posB   (IR: GSM6505165 -- `expr` = replicate 1..4)

Each row is ONE pairwise contact between locus A and locus B of the same concatemer
(the table is the all-vs-all expansion of each multi-way read: a read with n monomers
contributes C(n,2) rows). We rebuild the concatemer (= one hyperedge) by grouping rows
by read id (and `expr` when present) and taking the UNION of distinct loci over the A and
B columns -- this recovers all n monomers.

Chromosomes may be integers (1..22, 23=X, 24=Y) or 'chrN' strings. Only autosomes
chr1..chr22 are emitted, matching process_clusters.read_clusters' regex ^chr([0-9]+):(\\d+)$.

Output: one line per concatemer with >= 2 distinct autosomal loci:
    <cluster_id>\\tchr{c}:{pos}\\tchr{c}:{pos} ...

Requires numpy + pandas (+ mat-io if reading a .mat directly).
"""
import argparse
import sys

import numpy as np
import pandas as pd

EXPR_MULT = 10_000_000_000  # > max read_id, so expr*MULT + read_id is a unique group key


def load_table(path, var=None):
    if path.endswith(".mat"):
        from matio import load_from_mat
        data = load_from_mat(path, raw_data=False)
        keys = [k for k in data if not k.startswith("__")]
        return data[var or keys[0]]
    return pd.read_csv(path)


def chrom_to_int(series):
    """Map chromosome column to integer codes (X->23, Y->24, M->25); non-autosomes drop later."""
    if np.issubdtype(series.dtype, np.number):
        return series.to_numpy().astype("int64")
    s = series.astype(str).str.replace("chr", "", regex=False)
    s = s.replace({"X": "23", "Y": "24", "M": "25", "MT": "25"})
    return pd.to_numeric(s, errors="coerce").fillna(-1).astype("int64").to_numpy()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="pairwise table: .csv or .mat")
    ap.add_argument("-o", "--output", required=True, help="output .clusters file")
    ap.add_argument("--var", default=None, help="variable name if reading a .mat")
    args = ap.parse_args()

    df = load_table(args.input, args.var)
    cols = {c.lower(): c for c in df.columns}
    for r in ("read_id", "chra", "posa", "chrb", "posb"):
        if r not in cols:
            raise SystemExit(f"missing column '{r}'; have {list(df.columns)}")

    read_id = df[cols["read_id"]].to_numpy().astype("int64")
    has_expr = "expr" in cols
    if has_expr:
        expr = df[cols["expr"]].to_numpy().astype("int64")
        key = expr * EXPR_MULT + read_id
    else:
        key = read_id

    chrA = chrom_to_int(df[cols["chra"]]); posA = df[cols["posa"]].to_numpy().astype("int64")
    chrB = chrom_to_int(df[cols["chrb"]]); posB = df[cols["posb"]].to_numpy().astype("int64")

    # Stack the two sides into one long (key, chrom, pos) table.
    key2 = np.concatenate([key, key])
    chr2 = np.concatenate([chrA, chrB])
    pos2 = np.concatenate([posA, posB])

    # Autosomes only.
    m = (chr2 >= 1) & (chr2 <= 22)
    key2, chr2, pos2 = key2[m], chr2[m], pos2[m]

    # Sort by (key, chrom, pos); drop duplicate loci within a concatemer.
    order = np.lexsort((pos2, chr2, key2))
    key2, chr2, pos2 = key2[order], chr2[order], pos2[order]
    dup = np.zeros(len(key2), dtype=bool)
    dup[1:] = (key2[1:] == key2[:-1]) & (chr2[1:] == chr2[:-1]) & (pos2[1:] == pos2[:-1])
    keep = ~dup
    key2, chr2, pos2 = key2[keep], chr2[keep], pos2[keep]

    # Build "chrC:pos" tokens (only on the kept, deduped loci).
    tok = np.char.add(np.char.add(np.char.add("chr", chr2.astype("U2")), ":"), pos2.astype("U10"))

    # Group consecutive equal keys.
    bounds = np.flatnonzero(key2[1:] != key2[:-1]) + 1
    starts = np.concatenate(([0], bounds))
    ends = np.concatenate((bounds, [len(key2)]))

    n_emit = 0
    with open(args.output, "w") as out:
        for s, e in zip(starts.tolist(), ends.tolist()):
            if e - s < 2:
                continue
            k = int(key2[s])
            if has_expr:
                label = f"{k // EXPR_MULT}_{k % EXPR_MULT}"
            else:
                label = str(k)
            out.write(label + "\t" + "\t".join(tok[s:e]) + "\n")
            n_emit += 1

    print(f"{args.input}: {n_emit:,} concatemers (>=2 autosomal loci) -> {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
