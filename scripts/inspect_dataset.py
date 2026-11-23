"""Inspect the official dataset BEFORE writing dataset-specific code. Usage: python scripts/inspect_dataset.py"""
import collections
import pathlib
from huggingface_hub import list_repo_files, snapshot_download

REPO = "Tarakeshwaran/CIT_Hackathon_PS03_rag"
files = list_repo_files(REPO, repo_type="dataset")
print(f"{len(files)} files in {REPO}")
print(collections.Counter(pathlib.Path(f).suffix.lower() or "<none>" for f in files))
for f in files[:60]:
    print(" ", f)
if input("Download to data/official/? [y/N] ").lower() == "y":
    p = snapshot_download(REPO, repo_type="dataset", local_dir="data/official")
    print("Downloaded to", p)
    import pymupdf
    for pdf in sorted(pathlib.Path(p).rglob("*.pdf")):
        d = pymupdf.open(pdf)
        imgs = sum(len(pg.get_images()) for pg in d)
        txt = sum(len(pg.get_text().strip()) > 30 for pg in d)
        print(f"{pdf.name}: {d.page_count} pages, {txt} with text layer, {imgs} embedded images")
