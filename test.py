from pathlib import Path
import hashlib

def file_hash(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()

root = Path("SOS_REPORTS")
for p in sorted(root.rglob("*.tar.xz")):
    print(p.stat().st_size, file_hash(p), p)