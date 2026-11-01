"""Workspace discovery and isolated access to the existing scientific modules."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import platform
import shutil
from pathlib import Path
import sys
import threading

_IMPORT_LOCK = threading.RLock()


def runtime_info() -> dict:
    release = platform.release()
    system = platform.system()
    return {"platform": "WSL" if system == "Linux" and "microsoft" in release.lower() else system,
            "python": sys.executable}


def discover_workspace(value: str | Path | None = None) -> Path:
    if value or os.environ.get("TOMOGUI_WORKSPACE"):
        root = Path(value or os.environ["TOMOGUI_WORKSPACE"]).expanduser().resolve()
        if not (root / "Scripts/prepare_files.py").is_file():
            raise ValueError(f"Not a TomoATT workflow workspace: {root}")
        return root
    for start in (Path.cwd(), Path(__file__).resolve().parent):
        for root in (start, *start.parents):
            if (root / "Scripts/prepare_files.py").is_file():
                return root
    raise ValueError("Set --workspace to the folder containing Scripts and RESULTS/TEST.")


def load_module(path: Path):
    """Import a workspace file without executing its CLI or changing cwd."""
    path = path.resolve()
    name = "_tomogui_legacy_" + hashlib.sha256(str(path).encode()).hexdigest()[:16]
    with _IMPORT_LOCK:
        if name in sys.modules:
            return sys.modules[name]
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise ValueError(f"Cannot load workspace module: {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            sys.modules.pop(name, None)
            raise
        return module


def viewer_modules(root: Path):
    """Use the original headless viewer modules without importing Qt."""
    package_path = root / "h5_viewer"
    name = "_tomogui_viewer_" + hashlib.sha256(str(root).encode()).hexdigest()[:12]
    with _IMPORT_LOCK:
        if name not in sys.modules:
            spec = importlib.util.spec_from_file_location(
                name, package_path / "__init__.py", submodule_search_locations=[str(package_path)]
            )
            if spec is None or spec.loader is None:
                raise ValueError("The workspace must provide h5_viewer.")
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
        return tuple(importlib.import_module(name + "." + leaf) for leaf in
                     ("readers", "data_model", "expression_engine"))


class Workspace:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.state_dir = self.root / ".tomogui"
        self.lock = threading.RLock()

    def path(self, value: str | Path, *, exists: bool = True) -> Path:
        path = Path(value).expanduser()
        path = (self.root / path).resolve() if not path.is_absolute() else path.resolve()
        if not path.is_relative_to(self.root):
            raise ValueError("Select a path inside this project's workspace.")
        if exists and not path.exists():
            raise ValueError(f"Path does not exist: {path}")
        return path

    def read_state(self) -> dict:
        with self.lock:
            path = self.state_dir / "project.json"
            return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

    def update_state(self, **changes) -> dict:
        with self.lock:
            self.state_dir.mkdir(parents=True, exist_ok=True)
            state = self.read_state()
            state.update(changes)
            temp = self.state_dir / "project.json.tmp"
            temp.write_text(json.dumps(state, indent=2, allow_nan=False), encoding="utf-8")
            temp.replace(self.state_dir / "project.json")
            return state

    def artifact(self, category: str) -> Path:
        import uuid
        path = self.state_dir / category / uuid.uuid4().hex[:12]
        path.mkdir(parents=True, exist_ok=False)
        return path

    def browse(self, value: str = "") -> dict:
        folder = self.path(value or self.root)
        if folder.is_file():
            folder = folder.parent
        entries = []
        for path in folder.iterdir():
            if path.name.startswith(".") or path.name == "__pycache__":
                continue
            try:
                if not path.resolve().is_relative_to(self.root):
                    continue
                # Windows cannot follow Linux symlinks on a WSL-mounted drive.
                # One inaccessible entry must not prevent selecting other files.
                path.stat()
                directory = path.is_dir()
            except (OSError, RuntimeError):
                continue
            entries.append({"name": path.name, "path": str(path), "directory": directory})
        entries.sort(key=lambda entry: (not entry["directory"], entry["name"].lower()))
        return {"path": str(folder), "parent": str(folder.parent) if folder != self.root else None,
                "entries": entries[:600]}


def initialize_workspace(value: str | Path) -> Path:
    """Create a new portable project using packaged, data-free workflow helpers."""
    root = Path(value).expanduser().resolve()
    resources = Path(__file__).parent / 'resources/workflow'
    if not (resources / 'Scripts/prepare_files.py').is_file():
        raise ValueError('The installed package is missing its workflow resources. Reinstall TomoGUI.')
    root.mkdir(parents=True, exist_ok=False)
    shutil.copytree(resources, root, dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '*.pyo'))
    (root / 'data').mkdir()
    (root / 'data/README.txt').write_text(
        'Copy your arrival catalogs, src_rec files, station metadata and HDF5 models into this project.\n'
        'Open the project with: tomogui --workspace /path/to/this/project\n', encoding='ascii')
    return root
