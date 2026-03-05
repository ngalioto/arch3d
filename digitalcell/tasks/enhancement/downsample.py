import argparse
import cooler
import numpy as np
import os

def main(
    frac: float,
    mcool_file: str,
    save_dir: str | None = None
):

    if save_dir is None:
        # save the downsampled Hi-C in the same directory as the input mcool file
        save_dir = os.path.dirname(mcool_file)

    rng = np.random.default_rng(123)

    accession = os.path.splitext(os.path.basename(mcool_file))[0]

    c = cooler.Cooler(f'{mcool_file}::/resolutions/5000')
    bins = c.bins()[:]
    pixels = c.pixels()[:]

    # binomial thinning 
    counts = pixels['count'].to_numpy(dtype=int)
    pixels['count'] = rng.binomial(counts, frac)

    # drop zero-count pixels
    pixels = pixels[pixels['count'] > 0]

    # re-aggregate & sort
    pixels = (
        pixels.groupby(['bin1_id', 'bin2_id'], as_index=False)['count']
        .sum()
        .sort_values(['bin1_id', 'bin2_id'])
    )

    cooler.create_cooler(
        f"{accession}_{int(100*frac)}pct.cool",
        bins=bins,
        pixels=pixels[['bin1_id', 'bin2_id', 'count']],
        ordered=True,
        dtypes={'count': 'int32'},
    )

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--frac", type=float)
    parser.add_argument("--mcool_file")
    parser.add_argument("--save_dir", default=None)
    args = parser.parse_args()

    main(
        args.frac,
        args.mcool_file,
        args.save_dir
    )
