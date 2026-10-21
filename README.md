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

## Regenerating the numbers from the stored tables

The tables in `results/next/` are enough on their own; no dataset is
needed for this step.

    python3 scripts/make_paper_macros.py --out-dir <directory>

This reads `results/next/` and writes the macro definitions that carry
every measured value into the text. `results/next/environment.json`
records the Python, numpy and scipy versions the stored tables were
produced with.

## Rerunning the official KITTI program

The comparison against the official evaluation code needs that code,
which is not redistributed here. Fetch and check it:

    https://s3.eu-central-1.amazonaws.com/avg-kitti/devkit_odometry.zip
    sha256 c4703456de60cdeaaec672381fbe169155ba4c2fcded102335b42cb07ff76758

Unpack it and pass the `cpp` directory:

    python3 scripts/kitti_official_binary_check.py \
        --devkit <path>/devkit/cpp --kitti-root <path>/kitti/dataset

`evaluate_odometry.cpp` ships with a `main` that does not match the
`mail.h` beside it, so the script copies the file up to the line where
`main` starts and compiles that copy with `matrix.cpp`. The part it keeps
is byte for byte the original.

## Requirements

Python 3, numpy, scipy, matplotlib, and a C++ compiler for the section
above. If scipy is installed in your user site directory, do not run
these scripts with `python3 -I`: that flag drops it from the path.
