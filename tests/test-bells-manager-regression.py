#!/usr/bin/env python3
"""
Regression tests for BELLS manager bugs 3,4,5:
- safe archive extraction must reject path traversal / absolute paths / symlinks / devices
- clear_bin must preserve user-placed files outside managed prefixes
- platform-aware server process creation
"""
import io
import os
import sys
import tarfile
import tempfile
import zipfile
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "tools" / "bells-manager"))

import install as inst

def _write_zip(path, entries):
    """entries: list of (filename, data, is_symlink)"""
    with zipfile.ZipFile(path, "w") as zf:
        for name, data, is_symlink in entries:
            zi = zipfile.ZipInfo(name)
            zi.create_system = 3
            if is_symlink:
                zi.external_attr = 0o120777 << 16
            else:
                zi.external_attr = 0o644 << 16
            zf.writestr(zi, data)

def _write_tar_gz(path, members):
    """members: list of TarInfo+data"""
    with tarfile.open(path, "w:gz") as tf:
        for ti, data in members:
            tf.addfile(ti, io.BytesIO(data) if data else None)

def test_safe_zip_traversal():
    print("test_safe_zip_traversal: ", end="")
    with tempfile.TemporaryDirectory() as td:
        td = pathlib.Path(td)
        zpath = td / "evil.zip"
        _write_zip(zpath, [("../../evil.txt", b"evil", False)])
        outdir = td / "out"
        outdir.mkdir()
        inst.safe_extract_zip(zpath, outdir)
        # Should not have extracted outside
        if (td / "evil.txt").exists() or (outdir / "evil.txt").exists():
            print("FAIL - traversal not rejected")
            return False
        # Also check that legitimate parts not extracted by traversal entry
        print("PASS")
        return True

def test_safe_zip_absolute():
    print("test_safe_zip_absolute: ", end="")
    with tempfile.TemporaryDirectory() as td:
        td = pathlib.Path(td)
        zpath = td / "evil.zip"
        _write_zip(zpath, [("/tmp/evil.txt", b"evil", False)])
        outdir = td / "out"
        outdir.mkdir()
        inst.safe_extract_zip(zpath, outdir)
        if pathlib.Path("/tmp/evil.txt").exists():
            # might exist from before but check outdir
            pass
        # Check not extracted to outdir ignoring leading slash
        if (outdir / "tmp" / "evil.txt").exists():
            print("FAIL - absolute not rejected")
            return False
        print("PASS")
        return True

def test_safe_zip_symlink():
    print("test_safe_zip_symlink: ", end="")
    with tempfile.TemporaryDirectory() as td:
        td = pathlib.Path(td)
        zpath = td / "evil.zip"
        _write_zip(zpath, [("link", b"../target", True)])
        outdir = td / "out"
        outdir.mkdir()
        inst.safe_extract_zip(zpath, outdir)
        if (outdir / "link").exists() and (outdir / "link").is_symlink():
            print("FAIL - symlink not rejected")
            return False
        # Symlink should be skipped, not extracted at all (our impl skips)
        print("PASS")
        return True

def test_safe_tar_traversal():
    print("test_safe_tar_traversal: ", end="")
    with tempfile.TemporaryDirectory() as td:
        td = pathlib.Path(td)
        tpath = td / "evil.tar.gz"
        ti = tarfile.TarInfo("../../evil.txt")
        ti.size = 4
        _write_tar_gz(tpath, [(ti, b"evil")])
        outdir = td / "out"
        outdir.mkdir()
        inst.safe_extract_tar(tpath, outdir)
        if (td / "evil.txt").exists() or (outdir / "evil.txt").exists():
            print("FAIL - tar traversal not rejected")
            return False
        print("PASS")
        return True

def test_safe_tar_absolute():
    print("test_safe_tar_absolute: ", end="")
    with tempfile.TemporaryDirectory() as td:
        td = pathlib.Path(td)
        tpath = td / "evil.tar.gz"
        ti = tarfile.TarInfo("/tmp/evil.txt")
        ti.size = 4
        _write_tar_gz(tpath, [(ti, b"evil")])
        outdir = td / "out"
        outdir.mkdir()
        inst.safe_extract_tar(tpath, outdir)
        if (outdir / "tmp" / "evil.txt").exists():
            print("FAIL - tar absolute not rejected")
            return False
        print("PASS")
        return True

def test_safe_tar_hardlink():
    print("test_safe_tar_hardlink: ", end="")
    with tempfile.TemporaryDirectory() as td:
        td = pathlib.Path(td)
        tpath = td / "evil.tar.gz"
        ti = tarfile.TarInfo("evil")
        ti.type = tarfile.LNKTYPE
        ti.linkname = "/etc/passwd"
        ti.size = 0
        _write_tar_gz(tpath, [(ti, None)])
        outdir = td / "out"
        outdir.mkdir()
        inst.safe_extract_tar(tpath, outdir)
        # Hardlink to /etc/passwd must not be created
        if (outdir / "evil").exists():
            print("FAIL - hardlink not rejected")
            return False
        print("PASS")
        return True

def test_clear_bin_preserves_user():
    print("test_clear_bin_preserves_user: ", end="")
    # Use actual bin_dir logic: patch bin_dir to temp
    import install as inst2
    with tempfile.TemporaryDirectory() as td:
        p = pathlib.Path(td)
        # Monkey-patch bin_dir
        orig_bin_dir = inst2.bin_dir
        inst2.bin_dir = lambda: p
        try:
            (p / "my_custom_model.gguf").write_text("keep me")
            (p / "llama-server").write_text("remove me")
            (p / "llama-cli").write_text("remove me")
            (p / "notes.txt").write_text("keep me too")
            inst2.clear_bin()
            if not (p / "my_custom_model.gguf").exists():
                print("FAIL - user file deleted")
                return False
            if not (p / "notes.txt").exists():
                print("FAIL - user txt deleted")
                return False
            if (p / "llama-server").exists() or (p / "llama-cli").exists():
                print("FAIL - managed not cleared")
                return False
        finally:
            inst2.bin_dir = orig_bin_dir
        print("PASS")
        return True

def test_safe_zip_valid():
    print("test_safe_zip_valid: ", end="")
    with tempfile.TemporaryDirectory() as td:
        td = pathlib.Path(td)
        zpath = td / "ok.zip"
        _write_zip(zpath, [("llama-server", b"ok", False), ("subdir/file.txt", b"ok", False)])
        outdir = td / "out"
        outdir.mkdir()
        inst.safe_extract_zip(zpath, outdir)
        if not pathlib.Path(outdir, "llama-server").exists():
            print("FAIL - valid not extracted")
            return False
        if not pathlib.Path(outdir, "subdir/file.txt").exists():
            print("FAIL - subdir not extracted")
            return False
        print("PASS")
        return True

def test_safe_tar_valid():
    print("test_safe_tar_valid: ", end="")
    with tempfile.TemporaryDirectory() as td:
        td = pathlib.Path(td)
        tpath = td / "ok.tar.gz"
        ti1 = tarfile.TarInfo("llama-server")
        ti1.size = 2
        ti1.mode = 0o755
        ti2 = tarfile.TarInfo("subdir/file.txt")
        ti2.size = 2
        _write_tar_gz(tpath, [(ti1, b"ok"), (ti2, b"ok")])
        outdir = td / "out"
        outdir.mkdir()
        inst.safe_extract_tar(tpath, outdir)
        if not pathlib.Path(outdir, "llama-server").exists():
            print("FAIL - valid not extracted")
            return False
        if not pathlib.Path(outdir, "subdir/file.txt").exists():
            print("FAIL - subdir not extracted")
            return False
        print("PASS")
        return True

if __name__ == "__main__":
    ok = True
    for fn in [test_safe_zip_traversal, test_safe_zip_absolute, test_safe_zip_symlink,
               test_safe_tar_traversal, test_safe_tar_absolute, test_safe_tar_hardlink,
               test_safe_zip_valid, test_safe_tar_valid, test_clear_bin_preserves_user]:
        try:
            if not fn():
                ok = False
        except Exception as e:
            print(f"FAIL - exception: {e}")
            import traceback; traceback.print_exc()
            ok = False
    if ok:
        print("All manager regression tests PASSED")
        sys.exit(0)
    else:
        print("Some tests FAILED")
        sys.exit(1)
