"""One-off: give Stage 2's saved outer results the simplified harness's file names.

Until 31 October, search.py saved each run as data/cache/nested/{name}-{advisor}-{key16}
.npy and .json, key16 being the start of a hash of its code. Now each is
{name}-{advisor}.npy and .json (search.load_outer). This copies each old pair to its new
name, keeps every field of the old record, and adds "context" (the data and fold settings
load_outer checks), computed from the current code. Old files stay where they are.

    python migrate_results.py      # prints one row per result

Refuses, before copying anything, if two old files would get one new name, or if an old
record's data and fold fields disagree with the current code. Safe to run twice.
"""

import json
import os
import re
import shutil

import search

OLD_NAME = re.compile(r"(.+)-([a-z]+)-([0-9a-f]{16})\.json")  # name, advisor, key16


def _as_old_fields(c):
    """The current context, laid out as an old record's fields."""
    folds = [c["cv_folds"], c["cv_seeds"], c["inner_folds"], c["inner_seeds"]]
    old = {"folds": folds, "scoring": [c["hp_levels"], c["threshold"]]}
    return old | {"course_pin": c["course_pin"], "lockbox_seed": c["lockbox_seed"]}


def migrate():
    """Copy each old result to its new name. Returns rows (name, advisor, key16, status)."""
    context, found = search.run_context(), {}  # new path: (old file, key16, record)
    for file in sorted(os.listdir(search.NESTED_DIR)):
        if not (match := OLD_NAME.fullmatch(file)):
            continue
        with open(os.path.join(search.NESTED_DIR, file)) as f:
            record = json.load(f)
        new = search._path(record["name"], record["advisor"])
        if new in found:
            raise ValueError(f"{file} and {found[new][0]} would both become {new}")
        if any(record[k] != v for k, v in _as_old_fields(context).items()):
            raise ValueError(f"{file}: its data or folds differ from the current code")
        found[new] = (file, match[3], record)
    rows = []
    for new, (file, key, record) in found.items():
        status = "already there"
        if not os.path.exists(new + ".json"):  # the record is written last
            old = os.path.join(search.NESTED_DIR, file.removesuffix(".json"))
            shutil.copyfile(old + ".npy", new + ".npy")
            with open(new + ".json", "x") as f:
                json.dump({**record, "context": context}, f, indent=1)
            status = "copied"
        rows.append((record["name"], record["advisor"], key, status))
    return rows


if __name__ == "__main__":
    print(f"{'name':<22} {'advisor':<15} {'old key':<17} status")
    for name, advisor, key, status in migrate():
        print(f"{name:<22} {advisor:<15} {key:<17} {status}")
    print(f"Old files left in place, in {search.NESTED_DIR}")
