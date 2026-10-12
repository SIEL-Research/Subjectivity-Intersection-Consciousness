"""Verify released file identity without requiring a private repository."""
from pathlib import Path
import hashlib
ROOT=Path(__file__).resolve().parents[1]
def main():
 count=0
 for line in (ROOT/'MANIFEST.sha256').read_text().splitlines():
  expected,name=line.split('  ',1);p=ROOT/name
  assert p.is_file() and hashlib.sha256(p.read_bytes()).hexdigest()==expected,name
  count+=1
 print('PASS SHA-256 manifest:',count,'files')
if __name__=='__main__':main()
