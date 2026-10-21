#!/usr/bin/env python3
"""BELLS installer — detects GPU, builds from source or downloads precompiled binaries."""

import json
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path

REPO = "https://github.com/DGuckert/llama.cpp-BELLS"
RELEASE_TAG = "latest"
RELEASE_BASE = f"{REPO}/releases/download"

def log(msg: str):
    print(f"[BELLS] {msg}")


def error(msg: str):
    print(f"[BELLS] ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    log(f"  $ {' '.join(cmd)}")
    result = subprocess.run(cmd, **kwargs)
    # Correctly detect subprocess failures: callers that need to handle
    # errors check returncode; callers that want to abort pass check=True
    # which raises CalledProcessError. Never silently ignore a non-zero
    # returncode when check=True is not set — the caller must inspect it.
    return result


def check_python():
    v = sys.version_info
    if v.major < 3 or (v.major == 3 and v.minor < 10):
        error(f"Python 3.10+ required, got {v.major}.{v.minor}")
    log(f"Python {v.major}.{v.minor}.{v.micro}")


def detect_gpu() -> tuple[str, bool, bool]:
    """Returns (vendor, has_cuda_toolkit, has_vulkan_sdk)."""
    vendor = "none"
    has_cuda = bool(shutil.which("nvcc"))
    has_vulkan = bool(shutil.which("vulkaninfo"))

    if shutil.which("nvidia-smi"):
        vendor = "nvidia"
    elif has_vulkan:
        try:
            out = subprocess.run(["vulkaninfo", "--summary"], capture_output=True, text=True, timeout=10)
            text = out.stdout.lower()
            if "amd" in text or "radeon" in text:
                vendor = "amd"
            elif "intel" in text:
                vendor = "intel"
            else:
                vendor = "vulkan"
        except Exception:
            vendor = "vulkan"
    elif platform.system() == "Windows":
        try:
            out = subprocess.run(
                ["powershell", "-Command",
                 "(Get-CimInstance Win32_VideoController).Name"],
                capture_output=True, text=True, timeout=10,
            )
            text = out.stdout.lower()
            if "nvidia" in text:
                vendor = "nvidia"
            elif "amd" in text or "radeon" in text:
                vendor = "amd"
            elif "intel" in text:
                vendor = "intel"
        except Exception:
            pass

    return vendor, has_cuda, has_vulkan


def install_python_deps():
    log("Installing Python dependencies...")
    deps = ["textual>=0.40.0", "huggingface_hub"]
    # Propagate failures: a broken pip install must not be silently ignored.
    result = run([sys.executable, "-m", "pip", "install", "--quiet"] + deps, check=False)
    if result.returncode != 0:
        log(f"  pip install failed with exit code {result.returncode} — continuing anyway (offline?)")
        # Do not abort the whole installer for missing Python deps; bench and
        # manager will fail later with a clear message. But the failure is now
        # visible and not silently swallowed.


RELEASES_API = "https://api.github.com/repos/DGuckert/llama.cpp-BELLS/releases/latest"
MARKER = ".bells-release"


def bin_dir() -> Path:
    return Path(__file__).resolve().parent.parent.parent / "bin"


def server_binary() -> Path:
    return bin_dir() / ("llama-server.exe" if platform.system() == "Windows" else "llama-server")


def latest_release() -> dict:
    """The latest GitHub release, or {} when it cannot be read."""
    try:
        req = urllib.request.Request(RELEASES_API)
        req.add_header("Accept", "application/vnd.github.v3+json")
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read())
    except Exception as e:
        log(f"  Could not read the latest release: {e}")
        return {}


def installed_release() -> str:
    try:
        return (bin_dir() / MARKER).read_text().strip()
    except OSError:
        return ""


def precompiled_assets(vendor: str, has_cuda: bool) -> list[str]:
    """Release assets worth trying on this machine, best first."""
    system = platform.system()
    machine = platform.machine().lower()
    if machine not in ("x86_64", "amd64"):
        return []
    if system == "Windows":
        # the CUDA build needs the CUDA runtime DLLs, which come with the toolkit
        if vendor == "nvidia" and has_cuda:
            return ["bells-win-cuda-vulkan.zip", "bells-win-vulkan.zip"]
        return ["bells-win-vulkan.zip"]
    if system == "Linux":
        if vendor == "nvidia":
            return ["bells-linux-cuda.tar.gz", "bells-linux-vulkan.tar.gz"]
        return ["bells-linux-vulkan.tar.gz"]
    return []


def binaries_run() -> bool:
    """True when the installed llama-server starts, i.e. its shared libraries all resolve."""
    exe = server_binary()
    if not exe.is_file():
        return False
    try:
        r = subprocess.run([str(exe), "--version"], capture_output=True, text=True, timeout=60)
        return r.returncode == 0
    except Exception:
        return False


def _is_safe_path(base: Path, target: Path) -> bool:
    """True if target is inside base (prevents path traversal)."""
    try:
        base_resolved = base.resolve()
        target_resolved = (base / target).resolve()
        return str(target_resolved).startswith(str(base_resolved) + os.sep) or target_resolved == base_resolved
    except Exception:
        return False


def _is_safe_zip_info(info: zipfile.ZipInfo) -> bool:
    """Reject unsafe zip entries: absolute paths, traversal, symlinks, devices."""
    name = info.filename
    # Reject absolute paths and traversal
    if name.startswith("/") or name.startswith("\\"):
        return False
    if ".." in Path(name).parts:
        return False
    # zipfile exposes external_attr; check symlink bit (0xA000)
    # and directory traversal already handled; reject symlinks
    is_symlink = (info.external_attr >> 16) & 0o170000 == 0o120000
    if is_symlink:
        return False
    # Reject excessively large entries to avoid zip bombs (warn, not abort)
    # 2 GiB per file is a reasonable upper bound for binaries
    if info.file_size > 2 * 1024 * 1024 * 1024:
        return False
    return True


def safe_extract_zip(zip_path: Path, dest_dir: Path):
    """Safely extract a zip file, rejecting traversal and unsafe entries."""
    with zipfile.ZipFile(zip_path, "r") as zf:
        for info in zf.infolist():
            if not _is_safe_zip_info(info):
                log(f"  Skipping unsafe zip entry: {info.filename}")
                continue
            # Extra check: resolved path must stay inside dest_dir
            target = dest_dir / info.filename
            if not _is_safe_path(dest_dir, Path(info.filename)):
                log(f"  Skipping traversal zip entry: {info.filename}")
                continue
            zf.extract(info, dest_dir)


def _is_safe_tar_info(member: tarfile.TarInfo) -> bool:
    """Reject unsafe tar entries: absolute paths, traversal, symlinks outside, devices."""
    name = member.name
    if name.startswith("/") or name.startswith("\\"):
        return False
    if ".." in Path(name).parts:
        return False
    # Reject special file types: character/block devices, FIFOs
    if member.ischr() or member.isblk() or member.isfifo() or member.isdev():
        return False
    # Symlinks/hardlinks must not escape destination: check link target
    if member.issym() or member.islnk():
        # linkname must also be safe and not absolute/traversal
        if member.linkname.startswith("/") or ".." in Path(member.linkname).parts:
            return False
    return True


def safe_extract_tar(tar_path: Path, dest_dir: Path):
    """Safely extract a tar.gz file, rejecting traversal and unsafe entries."""
    with tarfile.open(tar_path, "r:gz") as tf:
        members = tf.getmembers()
        safe_members = []
        for m in members:
            if not _is_safe_tar_info(m):
                log(f"  Skipping unsafe tar entry: {m.name}")
                continue
            target = dest_dir / m.name
            if not _is_safe_path(dest_dir, Path(m.name)):
                log(f"  Skipping traversal tar entry: {m.name}")
                continue
            # For symlinks, also validate the resolved link target stays inside
            if m.issym() or m.islnk():
                link_target = (target.parent / m.linkname).resolve()
                try:
                    if not str(link_target).startswith(str(dest_dir.resolve()) + os.sep):
                        log(f"  Skipping symlink escaping dest: {m.name} -> {m.linkname}")
                        continue
                except Exception:
                    continue
            safe_members.append(m)
        # Use filter='data' on Python 3.12+ for extra safety if available
        try:
            tf.extractall(dest_dir, members=safe_members, filter='data')
        except TypeError:
            tf.extractall(dest_dir, members=safe_members)


def clear_bin():
    """Remove only BELLS-managed binaries, preserving user files.

    Previously this deleted every file in bin/, destroying user data on
    failed installs. Now it only removes known BELLS outputs and the marker.
    """
    d = bin_dir()
    if not d.exists():
        return
    # Known BELLS binaries and artifacts to clean before fresh install.
    # Preserve any other user files (models, configs, etc.).
    managed_prefixes = ("llama-", "bells", "ggml", "libggml", "libbells")
    managed_names = {".bells-release", "bells.bat", "bells.sh"}
    for f in d.iterdir():
        if f.is_file() or f.is_symlink():
            name_lower = f.name.lower()
            if name_lower in managed_names or any(name_lower.startswith(p) for p in managed_prefixes):
                try:
                    f.unlink()
                except OSError:
                    pass
            # else: preserve user file


def try_precompiled(vendor: str, has_cuda: bool, release: dict | None = None) -> bool:
    """Try to download precompiled binaries from the latest GitHub release."""
    assets = precompiled_assets(vendor, has_cuda)
    if not assets:
        return False

    data = release if release else latest_release()
    urls = {a["name"]: a["browser_download_url"] for a in data.get("assets", [])}
    dest_dir = bin_dir()

    for asset in assets:
        log(f"Trying precompiled binary: {asset}")
        if asset not in urls:
            log(f"  Asset {asset} not found in latest release")
            continue
        archive = None
        try:
            dest_dir.mkdir(parents=True, exist_ok=True)
            archive = dest_dir / asset

            log(f"  Downloading {urls[asset]}...")
            urllib.request.urlretrieve(urls[asset], archive)

            log(f"  Extracting to {dest_dir}...")
            # Clear only managed files, preserving user data
            clear_bin()
            if asset.endswith(".zip"):
                safe_extract_zip(archive, dest_dir)
            else:
                safe_extract_tar(archive, dest_dir)
            archive.unlink()
            archive = None

            if binaries_run():
                (dest_dir / MARKER).write_text(data.get("tag_name", "") + "\n")
                log(f"  Installed {data.get('tag_name', 'release')} to {dest_dir}")
                return True
            log(f"  {asset} does not start on this machine (missing GPU runtime libraries?)")
            # Don't leave a broken half-extract on failure — clean managed files only
            clear_bin()
        except Exception as e:
            log(f"  Download failed: {e}")
            if archive and archive.exists():
                try:
                    archive.unlink()
                except OSError:
                    pass
            # On failure, clean only managed files, not user data.
            # Do not call clear_bin() again with touched logic that would wipe user files.

    return False


def build_from_source(vendor: str, has_cuda: bool, has_vulkan: bool) -> bool:
    """Clone and build from source."""
    repo_dir = Path(__file__).resolve().parent.parent.parent

    if not (repo_dir / "CMakeLists.txt").exists():
        log(f"Cloning repository to {repo_dir}...")
        result = run(["git", "clone", REPO, str(repo_dir)], check=False)
        if result.returncode != 0:
            log(f"  git clone failed with exit code {result.returncode}")
            return False

    if not shutil.which("cmake"):
        error("cmake not found. Install CMake: https://cmake.org/download/")

    build_dir = repo_dir / "build"

    cmake_args = ["-B", str(build_dir)]
    if vendor == "nvidia" and has_cuda:
        cmake_args.append("-DGGML_CUDA=ON")
        log("Building with CUDA support")
    if has_vulkan:
        cmake_args.append("-DGGML_VULKAN=ON")
        log("Building with Vulkan support")
    if not has_cuda and not has_vulkan:
        log("Building CPU-only (no GPU SDK found)")

    if platform.system() == "Windows":
        cmake_args += ["-G", "Ninja"] if shutil.which("ninja") else []

    log("Configuring...")
    r = run(["cmake"] + cmake_args, cwd=str(repo_dir))
    if r.returncode != 0:
        log(f"  cmake configure failed with exit code {r.returncode}")
        return False

    log("Building (this may take a while)...")
    r = run(["cmake", "--build", str(build_dir), "--config", "Release",
             "-j", str(os.cpu_count() or 4)], cwd=str(repo_dir))
    if r.returncode != 0:
        log(f"  build failed with exit code {r.returncode}")
        return False
    return True


def create_shortcut():
    """Create a 'bells' command available system-wide."""
    bells_py = Path(__file__).resolve().parent / "bells.py"
    py_exe = sys.executable

    if platform.system() == "Windows":
        py_cmd = shutil.which("py") or py_exe
        bells_dir = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")) / "BELLS" / "bin"
        bells_dir.mkdir(parents=True, exist_ok=True)

        bat = bells_dir / "bells.bat"
        bat.write_text(f'@echo off\n"{py_cmd}" "{bells_py}" %*\n')
        log(f"Created {bat}")

        repo_bat = Path(__file__).resolve().parent.parent.parent / "bells.bat"
        repo_bat.write_text(f'@echo off\n"{py_cmd}" "{bells_py}" %*\n')

        _add_to_path_windows(str(bells_dir))
    else:
        local_bin = Path.home() / ".local" / "bin"
        local_bin.mkdir(parents=True, exist_ok=True)
        sh = local_bin / "bells"
        sh.write_text(f'#!/bin/sh\nexec "{py_exe}" "{bells_py}" "$@"\n')
        sh.chmod(0o755)
        log(f"Created {sh}")

        if str(local_bin) not in os.environ.get("PATH", ""):
            log(f"Add to PATH: export PATH=\"{local_bin}:$PATH\"")
            for rc in [Path.home() / ".bashrc", Path.home() / ".zshrc"]:
                if rc.exists():
                    content = rc.read_text()
                    line = f'\nexport PATH="{local_bin}:$PATH"  # BELLS\n'
                    if "# BELLS" not in content:
                        rc.write_text(content + line)
                        log(f"  Added to {rc}")


def _add_to_path_windows(dir_path: str):
    """Add a directory to the user's PATH on Windows, persistently."""
    try:
        result = subprocess.run(
            ["powershell", "-Command",
             f"[Environment]::GetEnvironmentVariable('PATH', 'User')"],
            capture_output=True, text=True, timeout=10,
        )
        user_path = result.stdout.strip()
        if result.returncode != 0:
            log(f"  Could not read user PATH: {result.stderr.strip()}")
            return

        if dir_path.lower() in user_path.lower():
            log(f"  Already on PATH: {dir_path}")
            return

        new_path = f"{user_path};{dir_path}" if user_path else dir_path
        result = subprocess.run(
            ["powershell", "-Command",
             f"[Environment]::SetEnvironmentVariable('PATH', '{new_path}', 'User')"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            log(f"  Could not update user PATH: {result.stderr.strip()}")
            return
        os.environ["PATH"] = os.environ.get("PATH", "") + os.pathsep + dir_path
        log(f"  Added to user PATH: {dir_path}")
        log(f"  Restart your terminal for 'bells' to work everywhere")
    except Exception as e:
        log(f"  Could not add to PATH automatically: {e}")
        log(f"  Manually add to PATH: {dir_path}")


def main():
    log("BELLS Installer")
    log("=" * 50)

    check_python()

    vendor, has_cuda, has_vulkan = detect_gpu()
    log(f"GPU vendor: {vendor}")
    log(f"CUDA toolkit: {'Yes' if has_cuda else 'No'}")
    log(f"Vulkan SDK: {'Yes' if has_vulkan else 'No'}")
    print()

    install_python_deps()
    print()

    repo_dir = Path(__file__).resolve().parent.parent.parent
    built = bool(list(repo_dir.glob("build*/bin/Release/llama-server*")) or list(repo_dir.glob("build*/bin/llama-server*")))
    downloaded = server_binary().is_file()

    release = {}
    if downloaded:
        # downloaded binaries follow the releases; a copy with no marker predates them
        release = latest_release()
        latest = release.get("tag_name", "")
        if latest and installed_release() != latest:
            log(f"Updating binaries to {latest}...")
            if not try_precompiled(vendor, has_cuda, release):
                log("Update not available for this machine.")
                # Preserve existing binaries — don't mark as not downloaded.
                # Previously this set downloaded=False which forced a rebuild
                # even though the old binaries were still valid.
                pass
        else:
            log("Binaries are up to date.")

    if not downloaded and not built:
        log("No binaries found. Attempting to get them...")
        if not try_precompiled(vendor, has_cuda, release):
            log("Precompiled not available, building from source...")
            if not build_from_source(vendor, has_cuda, has_vulkan):
                error("Build failed. Check the output above for errors.")
    elif not downloaded:
        log("Using the binaries built from source.")

    print()
    create_shortcut()
    print()

    log("Installation complete!")
    log("")
    log("Quick start:")
    log("  bells system       — Show system info")
    log("  bells models       — List local models")
    log("  bells bench        — Run benchmark")
    log("  bells estimate run -g \"3060\" -m qwen3.6-35b -q Q4_K_M — Estimate performance")


if __name__ == "__main__":
    main()
