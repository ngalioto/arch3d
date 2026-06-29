#!/usr/bin/env python3
"""
Convert pore_c `fragment_alignments.csv(.gz)` files into a `.clusters` text file
for process_clusters.py (used for the GM12878 Pore-C data, GSM4490689).

Each concatemer read -> one line:
    <read_name>\tchr{N}:{pos}\tchr{N}:{pos} ...
where pos = fragment midpoint (bp). Only autosomes chr1..chr22 are kept, because
process_clusters.read_clusters matches tokens with the regex ^chr([0-9]+):(\\d+)$
(so chrX/chrY/chrM and *_random/_alt contigs are dropped here too).

Rows for a read are contiguous within a pore_c file (sorted by read_idx); we group
on read_idx and use read_name (a globally-unique UUID) as the cluster id, so several
input files can simply be concatenated into one .clusters file.

Filtering: keep alignments with pass_filter == True (pore_c's own QC) and, optionally,
mapping_quality >= --min-mapq. Only concatemers with >= 2 autosomal fragments are emitted.
"""
import argparse
import csv
import gzip
import re
import sys
from pathlib import Path

AUTOSOME = re.compile(r'^chr([0-9]+)$')


def _open(p):
    return gzip.open(p, 'rt') if str(p).endswith('.gz') else open(p, 'rt')


def iter_clusters(path, min_mapq):
    with _open(path) as fh:
        reader = csv.reader(fh)
        header = next(reader)
        col = {name: i for i, name in enumerate(header)}
        c_ridx = col['read_idx']
        c_name = col['read_name']
        c_chrom = col['chrom']
        c_start = col['start']
        c_end = col['end']
        c_pf = col['pass_filter']
        c_mq = col['mapping_quality']

        cur_ridx = None
        cur_name = None
        loci = []
        for row in reader:
            if row[c_pf] != 'True':
                continue
            if min_mapq and int(row[c_mq]) < min_mapq:
                continue
            ridx = row[c_ridx]
            if ridx != cur_ridx:
                if cur_ridx is not None and len(loci) >= 2:
                    yield cur_name, loci
                cur_ridx = ridx
                cur_name = row[c_name]
                loci = []
            m = AUTOSOME.match(row[c_chrom])
            if m and 1 <= int(m.group(1)) <= 22:
                pos = (int(row[c_start]) + int(row[c_end])) // 2
                loci.append(f"chr{int(m.group(1))}:{pos}")
        if cur_ridx is not None and len(loci) >= 2:
            yield cur_name, loci


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('inputs', nargs='+', help='one or more fragment_alignments.csv(.gz) files')
    ap.add_argument('-o', '--output', required=True, help='output .clusters file')
    ap.add_argument('--min-mapq', type=int, default=0, help='minimum mapping_quality (default 0; rely on pass_filter)')
    args = ap.parse_args()

    total = 0
    with open(args.output, 'w') as out:
        for p in args.inputs:
            n = 0
            for name, loci in iter_clusters(p, args.min_mapq):
                out.write(name + '\t' + '\t'.join(loci) + '\n')
                n += 1
            print(f"{Path(p).name}: {n} concatemers", file=sys.stderr)
            total += n
    print(f"TOTAL concatemers written -> {args.output}: {total}", file=sys.stderr)


if __name__ == '__main__':
    main()
