"""Verify released file digests without loading record-level data."""
from pathlib import Path
import hashlib

root = Path(__file__).resolve().parent
failures = []
entries = (root / 'MANIFEST_SHA256.txt').read_text().splitlines()
for entry in entries:
    expected, name = entry.split('  ', 1)
    path = root / name
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        failures.append(name)
if failures:
    raise SystemExit('Digest check failed: ' + ', '.join(failures))
print(f'All {len(entries)} released file digests match.')
