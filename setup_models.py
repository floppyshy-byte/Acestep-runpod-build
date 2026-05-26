import os
import shutil
import sys
from pathlib import Path


def _log_tree(path: Path, prefix: str = "", max_depth: int = 3, _depth: int = 0) -> None:
    """Print a small tree of a directory (up to max_depth)."""
    if _depth > max_depth:
        return
    if not path.exists():
        print(f"{prefix}(does not exist)")
        return
    if path.is_file():
        print(f"{prefix}{path.name} ({path.stat().st_size})")
        return
    try:
        children = sorted(path.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
    except PermissionError:
        print(f"{prefix}(permission denied)")
        return
    for i, child in enumerate(children):
        is_last = i == len(children) - 1
        branch = "└── " if is_last else "├── "
        print(f"{prefix}{branch}{child.name}")
        if child.is_dir() and _depth < max_depth:
            next_prefix = prefix + ("    " if is_last else "│   ")
            _log_tree(child, next_prefix, max_depth, _depth + 1)


def _find_hf_cache_snapshot(repo_id: str, hf_home: Path | None = None) -> Path | None:
    """Find the on-disk snapshot directory for a cached HF repo."""
    if hf_home is None:
        hf_home = Path(os.environ.get("HF_HOME", "/runpod-volume/huggingface-cache/hub"))
    sanitized = repo_id.replace("/", "--")
    repo_cache = hf_home / f"models--{sanitized}"
    if not repo_cache.exists():
        # Case-insensitive fallback (RunPod may lowercase repo IDs)
        target_name = f"models--{sanitized}"
        for child in hf_home.iterdir():
            if child.is_dir() and child.name.lower() == target_name.lower():
                repo_cache = child
                break
        else:
            return None

    refs_dir = repo_cache / "refs"
    snapshots_dir = repo_cache / "snapshots"
    if not refs_dir.exists() or not snapshots_dir.exists():
        return None

    for ref_file in refs_dir.iterdir():
        commit_hash = ref_file.read_text().strip()
        snapshot = snapshots_dir / commit_hash
        if snapshot.exists():
            return snapshot

    # Fallback: pick any snapshot directory
    for child in snapshots_dir.iterdir():
        if child.is_dir():
            return child
    return None


def _has_weights(path: Path) -> bool:
    """Check if a directory contains model weight files."""
    if not path.is_dir():
        return False
    weight_names = (
        "model.safetensors",
        "model.safetensors.index.json",
        "pytorch_model.bin",
        "pytorch_model.bin.index.json",
        "diffusion_pytorch_model.safetensors",
        "diffusion_pytorch_model.safetensors.index.json",
    )
    if any((path / fname).exists() for fname in weight_names):
        return True
    # Sharded safetensors without index (e.g. model-00001-of-00004.safetensors)
    return any(f.name.startswith("model-") and f.name.endswith(".safetensors") for f in path.iterdir() if f.is_file())


def _validate_component(name: str, path: Path) -> None:
    """Log the contents of a component directory and whether it has weights."""
    if not path.exists():
        print(f"[Validate] {name}: NOT FOUND at {path}")
        return
    if not path.is_dir():
        print(f"[Validate] {name}: exists but is not a directory ({path})")
        return

    files = sorted([f.name for f in path.iterdir() if f.is_file()])
    dirs = sorted([d.name for d in path.iterdir() if d.is_dir()])
    has_w = _has_weights(path)
    print(f"[Validate] {name}: {'OK' if has_w else 'NO_WEIGHTS'} | files={files} dirs={dirs}")


def _is_weight_file(path: Path) -> bool:
    """Return True if the file is a large model weight file."""
    return path.suffix.lower() in (".safetensors", ".bin", ".pt", ".pth", ".ckpt")


def _mirror_component(src: Path, dst: Path) -> None:
    """
    Mirror src directory to dst.
    Weight files are symlinked; everything else is copied.
    """
    if dst.exists() or dst.is_symlink():
        if dst.is_symlink():
            dst.unlink()
        elif dst.is_dir():
            pass
        else:
            dst.unlink()

    dst.mkdir(parents=True, exist_ok=True)

    for child in src.iterdir():
        dst_child = dst / child.name
        if child.is_dir():
            _mirror_component(child, dst_child)
        elif _is_weight_file(child):
            if not dst_child.exists():
                try:
                    dst_child.symlink_to(child)
                except OSError as exc:
                    print(f"[Setup] ERROR linking weight {child.name}: {exc}")
            else:
                print(f"[Setup] SKIP: weight {child.name} already exists")
        else:
            if not dst_child.exists():
                try:
                    shutil.copy2(child, dst_child)
                except PermissionError:
                    shutil.copyfile(child, dst_child)
                    shutil.copymode(child, dst_child)
                except OSError as exc:
                    print(f"[Setup] ERROR copying {child.name}: {exc}")
            else:
                print(f"[Setup] SKIP: file {child.name} already exists")


def _mirror_standalone_repo(repo_id: str, dst: Path, hf_home: Path) -> bool:
    """
    Find a cached HF repo by repo_id and mirror its root contents into dst.
    Returns True if successful.
    """
    snapshot = _find_hf_cache_snapshot(repo_id, hf_home)
    if snapshot is None:
        return False

    print(f"[Setup] Found standalone repo snapshot for {repo_id}: {snapshot}")
    _mirror_component(snapshot, dst)
    return True


def _download_hf_raw(repo_id: str, filepath: str, dst: Path) -> bool:
    """Download a single file from HuggingFace via raw CDN URL."""
    import urllib.request
    url = f"https://huggingface.co/{repo_id}/resolve/main/{filepath}"
    print(f"[Setup] Downloading missing file from HF: {url}")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "acestep-setup/1.0"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            if resp.status != 200:
                print(f"[Setup] FAILED to download {filepath}: HTTP {resp.status}")
                return False
            dst.parent.mkdir(parents=True, exist_ok=True)
            with open(dst, "wb") as f:
                f.write(resp.read())
        print(f"[Setup] DOWNLOADED {dst}")
        return True
    except Exception as exc:
        print(f"[Setup] FAILED to download {filepath}: {exc}")
        return False


def _fix_missing_index_json(checkpoint_dir: Path, repo_id: str, component: str) -> None:
    """
    If a component has sharded safetensors but no index.json, download it from HF.
    """
    comp_dir = checkpoint_dir / component
    if not comp_dir.is_dir():
        return
    has_shards = any(
        f.name.startswith("model-") and f.name.endswith(".safetensors")
        for f in comp_dir.iterdir() if f.is_file()
    )
    index_file = comp_dir / "model.safetensors.index.json"
    if has_shards and not index_file.exists():
        _download_hf_raw(repo_id, f"{component}/model.safetensors.index.json", index_file)


def setup_checkpoints_from_cache() -> None:
    """
    Bridge RunPod's HF Model Cache to ACE-Step's expected checkpoint layout.
    Supports both bundled repos (components inside subdirs) and standalone
    repos (files at root, e.g. acestep-v15-xl-turbo, acestep-5Hz-lm-4B).
    """
    print("=" * 60)
    print("[Setup] Starting checkpoint setup from HF cache")
    print("=" * 60)

    hf_home = Path(os.environ.get("HF_HOME", "/runpod-volume/huggingface-cache/hub"))
    checkpoint_dir = Path(os.environ.get("ACESTEP_CHECKPOINTS_DIR", "/runpod-volume/checkpoints"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    repo_id = os.environ.get("ACESTEP_MAIN_MODEL_REPO", "ACE-Step/Ace-Step1.5")
    components_env = os.environ.get("ACESTEP_MAIN_MODEL_COMPONENTS", "")
    components = [c.strip() for c in components_env.split(",") if c.strip()]

    print(f"[Setup] ACESTEP_CHECKPOINTS_DIR = {checkpoint_dir}")
    print(f"[Setup] HF_HOME = {hf_home}")
    print(f"[Setup] ACESTEP_MAIN_MODEL_REPO = {repo_id}")
    print(f"[Setup] Expected components = {components}")

    # Log what's on /runpod-volume
    runpod_vol = Path("/runpod-volume")
    if runpod_vol.exists():
        print("\n[Setup] /runpod-volume contents:")
        _log_tree(runpod_vol, max_depth=2)
    else:
        print("\n[Setup] /runpod-volume does NOT exist")

    # Log HF cache tree
    if hf_home.exists():
        print(f"\n[Setup] HF cache tree ({hf_home}):")
        _log_tree(hf_home, max_depth=3)
    else:
        print(f"\n[Setup] HF cache dir does NOT exist: {hf_home}")

    # If checkpoints already look good, nothing to do
    all_present = True
    for comp in components:
        if not _has_weights(checkpoint_dir / comp):
            all_present = False
            break
    if all_present and components:
        print("\n[Setup] All checkpoints already present. Nothing to do.")
        return

    # Try to find cached snapshot for the main repo
    main_snapshot = _find_hf_cache_snapshot(repo_id, hf_home)
    if main_snapshot is None:
        print(f"\n[Setup] WARNING: No HF cache snapshot found for {repo_id}")
        print(f"[Setup] Checked: {hf_home / ('models--' + repo_id.replace('/', '--'))}")
    else:
        print(f"\n[Setup] Found main HF cache snapshot: {main_snapshot}")
        print("[Setup] Snapshot contents:")
        _log_tree(main_snapshot, max_depth=2)

    # Validate each component in the main snapshot before linking
    if main_snapshot:
        print("\n[Setup] Validating main snapshot components:")
        for comp in components:
            _validate_component(comp, main_snapshot / comp)

    # Mirror components: try main snapshot first, fallback to standalone repo
    print("\n[Setup] Mirroring components (symlink weights, copy code/config):")
    for comp in components:
        dst = checkpoint_dir / comp
        src_in_main = main_snapshot / comp if main_snapshot else None

        if src_in_main and src_in_main.exists():
            _mirror_component(src_in_main, dst)
            print(f"[Setup] MIRRORED {dst} <- {src_in_main}")
        else:
            # Fallback: look for a standalone cached repo named ACE-Step/<comp>
            standalone_repo = f"ACE-Step/{comp}"
            print(f"[Setup] Component '{comp}' not in main snapshot. Trying standalone repo {standalone_repo} ...")
            ok = _mirror_standalone_repo(standalone_repo, dst, hf_home)
            if ok:
                print(f"[Setup] MIRRORED {dst} <- standalone repo {standalone_repo}")
            else:
                print(f"[Setup] SKIP: component '{comp}' not found in main snapshot or standalone repo {standalone_repo}")

        # Fix stale cache: download missing index.json for sharded models
        _fix_missing_index_json(checkpoint_dir, repo_id, comp)

    # Also mirror LM model for LLMHandler
    lm_model = os.environ.get("ACESTEP_LM_MODEL_PATH", "acestep-5Hz-lm-4B")
    lm_dst = Path("/app/models") / lm_model
    lm_src_in_main = main_snapshot / lm_model if main_snapshot else None

    if lm_src_in_main and lm_src_in_main.exists():
        _validate_component(lm_model, lm_src_in_main)
        _mirror_component(lm_src_in_main, lm_dst)
        print(f"[Setup] MIRRORED LM model {lm_dst} <- {lm_src_in_main}")
    else:
        standalone_repo = f"ACE-Step/{lm_model}"
        print(f"[Setup] LM model '{lm_model}' not in main snapshot. Trying standalone repo {standalone_repo} ...")
        ok = _mirror_standalone_repo(standalone_repo, lm_dst, hf_home)
        if ok:
            print(f"[Setup] MIRRORED LM model {lm_dst} <- standalone repo {standalone_repo}")
        else:
            print(f"[Setup] WARNING: LM model '{lm_model}' not found in main snapshot or standalone repo {standalone_repo}")

    # Final validation
    print("\n[Setup] Final checkpoint_dir validation:")
    for comp in components:
        _validate_component(comp, checkpoint_dir / comp)
    _validate_component(lm_model, Path("/app/models") / lm_model)

    print("=" * 60)


if __name__ == "__main__":
    setup_checkpoints_from_cache()
