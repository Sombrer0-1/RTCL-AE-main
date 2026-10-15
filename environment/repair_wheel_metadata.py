"""Fix cuSPARSELt 0.8.1's SBSA WHEEL tag after verifying its AArch64 ELF."""
import base64
import csv
import hashlib
from importlib.metadata import distribution
from pathlib import Path
import platform
import struct


def main():
    package = distribution("nvidia-cusparselt-cu13")
    metadata = package.read_text("WHEEL")
    if "manylinux2014_sbsa" not in metadata:
        return
    if platform.machine() != "aarch64" or package.version != "0.8.1":
        raise RuntimeError("Unexpected cuSPARSELt platform/version; refusing metadata repair")
    binaries = [Path(package.locate_file(f)) for f in package.files if ".so" in f.name]
    if not binaries:
        raise RuntimeError("cuSPARSELt ELF library missing")
    for binary in binaries:
        with binary.open("rb") as stream:
            header = stream.read(20)
        if header[:4] != b"\x7fELF" or header[5] != 1 or struct.unpack("<H", header[18:20])[0] != 183:
            raise RuntimeError(f"Not a little-endian AArch64 ELF: {binary}")
    wheel_entry = next(f for f in package.files if f.name == "WHEEL")
    record_entry = next(f for f in package.files if f.name == "RECORD")
    wheel_path = Path(package.locate_file(wheel_entry))
    record_path = Path(package.locate_file(record_entry))
    contents = metadata.replace("manylinux2014_sbsa", "manylinux2014_aarch64").encode()
    wheel_path.write_bytes(contents)
    with record_path.open(newline="") as stream:
        rows = list(csv.reader(stream))
    digest = base64.urlsafe_b64encode(hashlib.sha256(contents).digest()).decode().rstrip("=")
    for row in rows:
        if row[0] == str(wheel_entry):
            row[1:] = ["sha256=" + digest, str(len(contents))]
    with record_path.open("w", newline="") as stream:
        csv.writer(stream).writerows(rows)
    print("Repaired cuSPARSELt WHEEL tag; AArch64 ELF and package RECORD verified")


if __name__ == "__main__":
    main()
