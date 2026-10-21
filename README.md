# Reading a public reference trajectory changes the score

Code and result tables for the paper. The datasets themselves are not
here: each release has its own download page and its own terms, and the
scripts read them where you put them.

## What this computes

For one public reference trajectory, resampling it under a different
reading convention moves every segment by some distance. The weighted
sum of those distances, D, bounds how far a weighted-mean pose-distance
score can move for a fixed estimate. The scripts measure D on
10 releases under the two segment rules that official leaderboards
use, and compare it against the gaps between leaderboard entries.

## Layout

    scripts/       measurement and table code (49 files)
    tests/         the test suite (46 files); run it first
    results/next/  the tables this code produced (85 files)

## Running the tests

    python3 -m pytest tests/ -q

The tests need no dataset. They build small synthetic trajectories and
check the pieces the measurements rely on.

## Reproducing a table

Point the release roots in `scripts/kitti_segment_D.py` at your copies
of the data, then

    python3 scripts/kitti_segment_D.py --release kitti
    python3 scripts/kitti_segment_D.py --release kitti --rpe1m
    python3 scripts/release_bound_table.py

`results/next/dataset_register.csv` lists, for every release, the file that
was read, the time it publishes, the axis we measured on, the rate we
measured, and the terms page. The published time and the measured axis
are separate columns: on KITTI-360 they differ.

## Requirements

Python 3, numpy, scipy, matplotlib.
