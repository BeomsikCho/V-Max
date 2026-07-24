"""Create a waymax @N sharded symlink split from a tree of tfrecords.

Symlinks every <src>/<site>/<date>/*.tfrecord (sorted) into <out_dir> as
  <basename(out_dir)>.tfrecord-XXXXX-of-YYYYY
following waymax generate_sharded_filenames (index width max(5, digits),
'of' part %05d), plus a manifest.csv (shard_index, shard_name, source_path).

Usage:
  uv run python make_waymax_shards.py <src_root> <out_dir>
"""

import csv
import math
import os
import sys


def main():
    src, out_dir = sys.argv[1], sys.argv[2]
    base = os.path.basename(out_dir.rstrip("/"))
    assert not os.path.exists(out_dir) or not os.listdir(out_dir), f"{out_dir} not empty"

    files = []
    for site in sorted(os.listdir(src)):
        if not os.path.isdir(os.path.join(src, site)):
            continue
        for date in sorted(os.listdir(os.path.join(src, site))):
            ddir = os.path.join(src, site, date)
            files += [os.path.join(ddir, fn) for fn in sorted(os.listdir(ddir))
                      if fn.endswith(".tfrecord")]
    n = len(files)
    width = max(5, int(math.log10(n) + 1))
    fmt = f"{base}.tfrecord-%0{width}d-of-%05d"
    print(f"{n} files -> {out_dir} ({fmt % (0, n)} ...)")

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "manifest.csv"), "w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["shard_index", "shard_name", "source_path"])
        for i, srcf in enumerate(files):
            name = fmt % (i, n)
            os.symlink(srcf, os.path.join(out_dir, name))
            w.writerow([i, name, srcf])
    print(f"done: {n} symlinks + manifest.csv")
    print(f"waymax path: {os.path.join(out_dir, base)}.tfrecord@{n}")


if __name__ == "__main__":
    main()
