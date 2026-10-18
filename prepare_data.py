"""Verify official archive checksums, then safely extract the training data."""

import argparse
import hashlib
import shutil
import zipfile
from pathlib import Path

ARCHIVES = {
    "labeled": "1f60df2d4c1459c307e444741056e17a5ec537614cf577331ed04eb94d0765c1",
    "unlabeled": "4b4adf44827d2f62b87c8ce5dae805562964304f73ab40609b5a9efee83c373a",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--split", choices=ARCHIVES, nargs="+", default=list(ARCHIVES))
    args = parser.parse_args()
    root = args.root.resolve()
    for split in args.split:
        archive = root / f"FutureNetwork-data-{split}.zip"
        with archive.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != ARCHIVES[split]:
            raise ValueError(f"Checksum mismatch: {archive}: {digest}")
        print(f"{split}: SHA-256 OK", flush=True)
        with zipfile.ZipFile(archive) as source:
            members = source.infolist()
            for member in members:
                if not (root / member.filename).resolve().is_relative_to(root):
                    raise ValueError(f"Unsafe archive path: {member.filename}")
            required = sum(member.file_size for member in members)
            if shutil.disk_usage(root).free < required + 1024**3:
                raise RuntimeError(f"Insufficient disk space; need {required:,} bytes plus reserve")
            print(f"{split}: extracting {len(members)} entries, {required / 1e9:.2f} GB", flush=True)
            source.extractall(root)
        if not (root / split / "samples.csv").is_file():
            raise FileNotFoundError(f"Missing {split}/samples.csv after extraction")
        print(f"{split}: extraction complete", flush=True)


if __name__ == "__main__":
    main()
