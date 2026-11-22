"""Fetch only the WLASL clips for a vocabulary from the Kaggle 'wlasl-processed'
mirror, via HTTP Range requests on the remote zip (no 5 GB download).

Needs ~/.kaggle/kaggle.json. Respect the WLASL data-use agreement: the
videos are for research use and must never be committed to a public repo.

  python scripts/fetch_wlasl_kaggle.py ../WLASL/start_kit/WLASL_v0.3.json \
      --vocab data/vocab.json --out ../wlasl_raw
"""
import argparse
import base64
import io
import json
import struct
import urllib.request
import zipfile
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

DATASET = "https://www.kaggle.com/api/v1/datasets/download/risangbaskoro/wlasl-processed"


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


def signed_url():
    cred = json.loads((Path.home()/".kaggle/kaggle.json").read_text())
    token = base64.b64encode(f"{cred['username']}:{cred['key']}".encode()).decode()
    req = urllib.request.Request(DATASET, headers={"Authorization": f"Basic {token}"})
    try:
        urllib.request.build_opener(_NoRedirect).open(req)
    except urllib.error.HTTPError as e:
        if e.code in (301, 302, 303, 307):
            return e.headers["Location"]
        raise
    raise RuntimeError("Kaggle did not redirect to a download URL")


class RemoteZip(io.RawIOBase):
    def __init__(self, url):
        self.url, self.pos = url, 0
        head = urllib.request.urlopen(urllib.request.Request(url, method="HEAD"))
        self.size = int(head.headers["Content-Length"])

    def get(self, start, end):
        error = None
        for _ in range(6):
            try:
                req = urllib.request.Request(self.url, headers={"Range": f"bytes={start}-{end}"})
                with urllib.request.urlopen(req, timeout=120) as r:
                    return r.read()
            except Exception as e:  # retried
                error = e
        raise error

    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.pos

    def seek(self, off, whence=0):
        self.pos = {0: off, 1: self.pos+off, 2: self.size+off}[whence]
        return self.pos

    def read(self, n=-1):
        n = self.size-self.pos if n < 0 else n
        if n == 0 or self.pos >= self.size:
            return b""
        data = self.get(self.pos, min(self.size, self.pos+n)-1)
        self.pos += len(data)
        return data


def main():
    p = argparse.ArgumentParser()
    p.add_argument("annotations"); p.add_argument("--vocab", required=True)
    p.add_argument("--out", required=True); p.add_argument("--threads", type=int, default=12)
    a = p.parse_args()
    vocab = set(json.loads(Path(a.vocab).read_text()))
    ids = {i["video_id"] for e in json.loads(Path(a.annotations).read_text())
           if e["gloss"].upper() in vocab for i in e["instances"]}
    remote = RemoteZip(signed_url())
    infos = zipfile.ZipFile(remote).infolist()
    want = [i for i in infos if not i.is_dir() and
            (not i.filename.startswith("videos/") or Path(i.filename).stem in ids)]
    out = Path(a.out)
    print(f"{len(want)} files, {sum(i.compress_size for i in want)/1e6:.0f} MB", flush=True)

    def fetch(info):
        dest = out/info.filename
        if dest.exists() and dest.stat().st_size == info.file_size:
            return
        raw = remote.get(info.header_offset, info.header_offset+30+1024+info.compress_size)
        n, e = struct.unpack("<HH", raw[26:30])
        data = raw[30+n+e:30+n+e+info.compress_size]
        if info.compress_type == zipfile.ZIP_DEFLATED:
            data = zlib.decompressobj(-15).decompress(data)
        if len(data) != info.file_size or zlib.crc32(data) != info.CRC:
            raise ValueError(f"Corrupt member {info.filename}")
        dest.parent.mkdir(parents=True, exist_ok=True); dest.write_bytes(data)

    with ThreadPoolExecutor(a.threads) as pool:
        for k, _ in enumerate(pool.map(fetch, want), 1):
            if k % 100 == 0:
                print(f"{k}/{len(want)}", flush=True)
    print(f"Done: {out}")


if __name__ == "__main__":
    main()
