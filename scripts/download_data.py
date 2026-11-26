"""Fetch the recipe pool and the other large files that are not in git.

Run from anywhere:
    python scripts/download_data.py                        # download from the GitHub release
    python scripts/download_data.py --zip nibble_data.zip  # use a zip you were sent
    python scripts/download_data.py --force                # replace local copies that differ

The zip holds data/ and usda_sr/. Files you already have are left alone (so files tracked by git are never touched
without --force). The three large generated files are checked against SHA-256 sums, so everyone works with the same
recipe_ids and fields. Standard library only.
"""
import argparse
import hashlib
import shutil
import sys
import tempfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

URL = "https://github.com/IORosuOI/nibble-agent/releases/download/data-v1/nibble_data.zip"
CHECKSUMS = {  # update these when a new data version is released
    "data/recipes_merged.jsonl": "52ab5b405da1d339ca13a972c22f30f185f68960ae7117ea10e44326e8a060ce",
    "data/train_examples.jsonl": "e37c54abf27154901e5c78671f6d65554faac081ae3bd56ad889d500db6b817e",
    "data/val_examples.jsonl": "5e92efdc1f4ed48aceb55520731bfb2199246f63714c803d31e68728fa5d5c75",
}
ALLOWED_TOP = {"data", "usda_sr"}


def sha256(stream):
    h = hashlib.sha256()
    for chunk in iter(lambda: stream.read(1 << 20), b""):
        h.update(chunk)
    return h.hexdigest()


def file_sha256(path):
    with open(path, "rb") as f:
        return sha256(f)


def download(url, dest):
    print("downloading", url)
    try:
        with urllib.request.urlopen(url, timeout=60) as r, open(dest, "wb") as f:
            shutil.copyfileobj(r, f, 1 << 20)
    except (urllib.error.URLError, TimeoutError) as e:
        sys.exit(f"download failed: {e}\nGet nibble_data.zip from a teammate and run again with --zip <path>.")


def extract(zip_path, root, force):
    with zipfile.ZipFile(zip_path) as z:
        members = [m for m in z.infolist() if not m.is_dir()]
        names = {m.filename for m in members}
        # check everything before writing anything
        for m in members:
            p = PurePosixPath(m.filename)
            if p.is_absolute() or ".." in p.parts or ":" in m.filename or "\\" in m.filename or p.parts[0] not in ALLOWED_TOP:
                sys.exit(f"unsafe or unexpected path in zip: {m.filename}")
        for name, want in CHECKSUMS.items():
            if name not in names:
                sys.exit(f"the zip does not contain {name}")
            with z.open(name) as f:
                if sha256(f) != want:
                    sys.exit(f"checksum mismatch for {name}: this zip is a different data version")
        written = kept = 0
        for m in members:
            target = root / m.filename
            if target.exists() and not force:
                kept += 1
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            part = target.with_name(target.name + ".part")
            with z.open(m) as src, open(part, "wb") as dst:
                shutil.copyfileobj(src, dst, 1 << 20)
            part.replace(target)
            written += 1
    print(f"written {written}, already present {kept}")


def verify(root):
    bad = [n for n, want in CHECKSUMS.items() if not (root / n).is_file() or file_sha256(root / n) != want]
    for n in bad:
        print("DIFFERENT OR MISSING:", n)
    return not bad


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zip", help="use a local nibble_data.zip instead of downloading")
    ap.add_argument("--url", default=URL)
    ap.add_argument("--force", action="store_true", help="overwrite existing files with the zip's copies")
    ap.add_argument("--root", default=Path(__file__).resolve().parent.parent, type=Path, help="project root")
    a = ap.parse_args()
    if not a.force and (a.root / "usda_sr").is_dir() and all((a.root / n).is_file() for n in CHECKSUMS) and verify(a.root):
        print("data already up to date")
        return
    with tempfile.TemporaryDirectory() as tmp:
        zip_path = Path(a.zip) if a.zip else Path(tmp) / "nibble_data.zip"
        if not a.zip:
            download(a.url, zip_path)
        extract(zip_path, a.root, a.force)
    if not verify(a.root):
        sys.exit("some local files differ from this data version; run again with --force to replace them")
    print("done: data matches the expected version")


if __name__ == "__main__":
    main()
