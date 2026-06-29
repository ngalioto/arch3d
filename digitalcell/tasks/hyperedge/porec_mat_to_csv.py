#!/usr/bin/env python3
"""
Read a MATLAB `.mat` that stores a MATLAB **table** (an MCOS class object) into a CSV.

This is the working replacement for `scipy.io.loadmat(...)` + `pd.DataFrame(...)`:
scipy/pymatreader cannot decode MATLAB `table` objects (they return only an opaque
wrapper; the real columns live in a hidden `__function_workspace__` subsystem). The
`mat-io` package decodes that subsystem and returns the table as a pandas DataFrame.

    pip install mat-io

Usage:
    python porec_mat_to_csv.py input.mat -o output.csv [--var VARNAME]

Used for the BJ/IR fibroblast Pore-C tables (GSM6505166 / GSM6505165), whose columns
are [read_id, chrA, posA, chrB, posB] (BJ) or [expr, read_id, chrA, posA, chrB, posB] (IR),
chromosomes encoded as integers 1..22, 23=X, 24=Y.
"""
import argparse

from matio import load_from_mat


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mat", help="input .mat file")
    ap.add_argument("-o", "--output", required=True, help="output CSV path")
    ap.add_argument("--var", default=None, help="table variable name (default: the file's single non-meta var)")
    args = ap.parse_args()

    data = load_from_mat(args.mat, raw_data=False)
    keys = [k for k in data if not k.startswith("__")]
    var = args.var or (keys[0] if len(keys) == 1 else None)
    if var is None:
        raise SystemExit(f"Multiple variables {keys}; choose one with --var")

    df = data[var]
    df.to_csv(args.output, index=False)
    print(f"{var}: {len(df):,} rows, columns={list(df.columns)} -> {args.output}")


if __name__ == "__main__":
    main()
