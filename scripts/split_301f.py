"""Split a RideFlux 301-frame tree into stratified train/validation/test trees.

The expected source layout is::

    <source>/<site>/<date>/*.tfrecord

All files from the same ``<site>/<date>`` group stay in the same split, so the
three 91-frame windows later produced from one source record can never leak
across train, validation, and test.  Groups are allocated independently inside
each site, which keeps every sufficiently large site represented in all splits.

The output contains symbolic links rather than copies::

    <target>/train_301f/<site>/<date>/*.tfrecord
    <target>/validation_301f/<site>/<date>/*.tfrecord
    <target>/test_301f/<site>/<date>/*.tfrecord

A ``split_manifest.csv`` is also written under ``<target>``.  The script is
deterministic for a fixed source tree, ratio, and seed.  It refuses to write to
a non-empty target directory.

Examples::

    uv run python scripts/split_301f.py \
        /data/Motion_Planning_and_Prediction/train \
        /data/Motion_Planning_and_Prediction/splits_raw

    uv run python scripts/split_301f.py SOURCE TARGET --ratios 85 10 5 --seed 7
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
from dataclasses import dataclass
from pathlib import Path


SPLIT_DIRS = ("train_301f", "validation_301f", "test_301f")


@dataclass(frozen=True)
class Group:
    """One indivisible ``site/date`` group of source TFRecords."""

    site: str
    name: str
    files: tuple[Path, ...]
    tie_breaker: str

    @property
    def size(self) -> int:
        return len(self.files)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "source_root",
        type=Path,
        help="301-frame source tree (<site>/<date>/*.tfrecord)",
    )
    parser.add_argument(
        "target_root",
        type=Path,
        help="destination containing train_301f, validation_301f, and test_301f",
    )
    parser.add_argument(
        "--ratios",
        nargs=3,
        type=float,
        metavar=("TRAIN", "VALIDATION", "TEST"),
        default=(80.0, 10.0, 10.0),
        help="split weights or percentages (default: 80 10 10)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="seed used for deterministic group tie-breaking (default: 42)",
    )
    return parser.parse_args()


def validate_args(source_root: Path, target_root: Path, ratios: tuple[float, ...]) -> None:
    if not source_root.is_dir():
        raise ValueError(f"Source directory does not exist: {source_root}")
    if any(ratio < 0 for ratio in ratios) or sum(ratios) <= 0:
        raise ValueError(f"Ratios must be non-negative and have a positive sum: {ratios}")

    source_resolved = source_root.resolve()
    target_resolved = target_root.resolve()
    if target_resolved == source_resolved or source_resolved in target_resolved.parents:
        raise ValueError("Target directory must not be the source directory or inside it")

    if target_root.exists() and any(target_root.iterdir()):
        raise ValueError(f"Target directory is not empty: {target_root}")


def stable_digest(seed: int, site: str, group: str) -> str:
    value = f"{seed}\0{site}\0{group}".encode()
    return hashlib.sha256(value).hexdigest()


def discover_groups(source_root: Path, seed: int) -> dict[str, list[Group]]:
    """Read the expected two-level tree without following unrelated files."""
    groups_by_site: dict[str, list[Group]] = {}

    for site_dir in sorted(path for path in source_root.iterdir() if path.is_dir()):
        site_groups = []
        for group_dir in sorted(path for path in site_dir.iterdir() if path.is_dir()):
            files = tuple(
                sorted(
                    path
                    for path in group_dir.iterdir()
                    if path.is_file() and path.name.endswith(".tfrecord")
                )
            )
            if files:
                site_groups.append(
                    Group(
                        site=site_dir.name,
                        name=group_dir.name,
                        files=files,
                        tie_breaker=stable_digest(seed, site_dir.name, group_dir.name),
                    )
                )
        if site_groups:
            groups_by_site[site_dir.name] = site_groups

    if not groups_by_site:
        raise ValueError(
            f"No TFRecords found under {source_root}/<site>/<date>/*.tfrecord"
        )
    return groups_by_site


def allocate_site_groups(
    groups: list[Group], ratios: tuple[float, ...]
) -> list[list[Group]]:
    """Allocate one site's groups while approximating ratios by file count.

    If the site has at least as many groups as non-zero splits, one group is
    seeded into every active split.  Remaining groups are greedily placed into
    the split with the largest file-count deficit relative to its target.
    """
    ratio_sum = sum(ratios)
    total_files = sum(group.size for group in groups)
    targets = [total_files * ratio / ratio_sum for ratio in ratios]
    active = [index for index, ratio in enumerate(ratios) if ratio > 0]
    assignments: list[list[Group]] = [[] for _ in SPLIT_DIRS]
    counts = [0] * len(SPLIT_DIRS)

    # Large groups are hardest to place; the digest randomizes equal-size ties.
    remaining = sorted(groups, key=lambda group: (-group.size, group.tie_breaker))

    # Represent every active split when the site has enough independent groups.
    if len(remaining) >= len(active):
        seeded_splits = sorted(active, key=lambda index: (-ratios[index], index))
        for split_index in seeded_splits:
            group = remaining.pop(0)
            assignments[split_index].append(group)
            counts[split_index] += group.size

    for group in remaining:
        # Largest normalized deficit wins.  Ratio and fixed index make ties stable.
        split_index = max(
            active,
            key=lambda index: (
                (targets[index] - counts[index]) / max(targets[index], 1.0),
                ratios[index],
                -index,
            ),
        )
        assignments[split_index].append(group)
        counts[split_index] += group.size

    return assignments


def build_plan(
    groups_by_site: dict[str, list[Group]], ratios: tuple[float, ...]
) -> dict[str, list[tuple[int, Group]]]:
    plan: dict[str, list[tuple[int, Group]]] = {site: [] for site in groups_by_site}
    for site, groups in groups_by_site.items():
        assignments = allocate_site_groups(groups, ratios)
        for split_index, assigned_groups in enumerate(assignments):
            plan[site].extend((split_index, group) for group in assigned_groups)
    return plan


def materialize_plan(
    source_root: Path,
    target_root: Path,
    plan: dict[str, list[tuple[int, Group]]],
) -> None:
    """Create the split symlink trees and a source-to-split manifest."""
    target_root.mkdir(parents=True, exist_ok=True)
    for split_dir in SPLIT_DIRS:
        (target_root / split_dir).mkdir()

    manifest_path = target_root / "split_manifest.csv"
    with manifest_path.open("w", newline="") as manifest_file:
        writer = csv.writer(manifest_file, lineterminator="\n")
        writer.writerow(
            ["split", "site", "group", "relative_path", "source_path", "link_path"]
        )

        for site in sorted(plan):
            for split_index, group in sorted(
                plan[site], key=lambda item: (item[0], item[1].name)
            ):
                split_dir = SPLIT_DIRS[split_index]
                output_group = target_root / split_dir / site / group.name
                output_group.mkdir(parents=True, exist_ok=True)

                for source_file in group.files:
                    link_path = output_group / source_file.name
                    link_path.symlink_to(source_file.resolve())
                    writer.writerow(
                        [
                            split_dir,
                            site,
                            group.name,
                            source_file.relative_to(source_root),
                            source_file.resolve(),
                            link_path,
                        ]
                    )


def print_summary(
    plan: dict[str, list[tuple[int, Group]]], ratios: tuple[float, ...]
) -> None:
    split_files = [0] * len(SPLIT_DIRS)
    split_groups = [0] * len(SPLIT_DIRS)
    total_files = 0

    print("\nPer-site allocation (files / groups):")
    print(f"{'site':<24} {'train':>16} {'validation':>16} {'test':>16}")
    for site in sorted(plan):
        site_files = [0] * len(SPLIT_DIRS)
        site_groups = [0] * len(SPLIT_DIRS)
        for split_index, group in plan[site]:
            site_files[split_index] += group.size
            site_groups[split_index] += 1
        total_files += sum(site_files)
        for index in range(len(SPLIT_DIRS)):
            split_files[index] += site_files[index]
            split_groups[index] += site_groups[index]
        cells = [f"{site_files[i]} / {site_groups[i]}" for i in range(len(SPLIT_DIRS))]
        print(f"{site:<24} {cells[0]:>16} {cells[1]:>16} {cells[2]:>16}")

    print("\nOverall allocation:")
    ratio_sum = sum(ratios)
    for index, split_dir in enumerate(SPLIT_DIRS):
        actual = 100.0 * split_files[index] / total_files
        requested = 100.0 * ratios[index] / ratio_sum
        print(
            f"- {split_dir:<16}: {split_files[index]:>8} files, "
            f"{split_groups[index]:>5} groups, {actual:6.2f}% "
            f"(requested {requested:.2f}%)"
        )


def main() -> None:
    args = parse_args()
    source_root = args.source_root.expanduser()
    target_root = args.target_root.expanduser()
    ratios = tuple(args.ratios)

    validate_args(source_root, target_root, ratios)
    groups_by_site = discover_groups(source_root, args.seed)
    plan = build_plan(groups_by_site, ratios)
    materialize_plan(source_root, target_root, plan)
    print_summary(plan, ratios)
    print(f"\nDone. Manifest: {target_root / 'split_manifest.csv'}")


if __name__ == "__main__":
    main()
