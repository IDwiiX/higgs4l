"""Sample access, local file cache and selected-event storage."""

import hashlib
import shutil
import urllib.request
from pathlib import Path

import atlasopenmagic as atom
import awkward as ak
import numpy as np
import uproot
import yaml

ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = ROOT / "data" / "cache"
SELECTED_DIR = ROOT / "data" / "selected"

LEPTON_BRANCHES = [
    "lep_pt",
    "lep_eta",
    "lep_phi",
    "lep_e",
    "lep_type",
    "lep_charge",
    "lep_isTrigMatched",
    "lep_isLooseID",
    "lep_isMediumID",
    "lep_isLooseIso",
]
EVENT_BRANCHES = [
    "trigE",
    "trigM",
    "eventNumber",
    "runNumber",
    "sum_of_weights",
    "xsec",
    "filteff",
    "kfac",
    "mcWeight",
    "ScaleFactor_PILEUP",
    "ScaleFactor_ELE",
    "ScaleFactor_MUON",
    "ScaleFactor_LepTRIGGER",
]
# What data/selected keeps per event; leptons stay in their stored order.
STORED_DTYPES = {
    "lep_pt": np.float32,
    "lep_eta": np.float32,
    "lep_phi": np.float32,
    "lep_e": np.float32,
    "lep_type": np.int8,
    "lep_charge": np.int8,
    "eventNumber": np.uint64,
    "runNumber": np.uint32,
}


def load_config(path: Path = ROOT / "config" / "analysis.yaml") -> dict:
    """Read the analysis configuration."""
    return yaml.safe_load(Path(path).read_text())


def file_urls(did, release: str, skim: str) -> list[str]:
    """Plain https URLs of one dataset (no fsspec simplecache, which is not persistent)."""
    atom.set_release(release)
    return atom.get_urls(str(did), skim=skim, protocol="https", cache=False)


def fetch(url: str, cache_dir: Path = CACHE_DIR) -> Path:
    """Download a file once into the cache and return its local path."""
    path = Path(cache_dir) / url.rsplit("/", 1)[1]
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    with urllib.request.urlopen(url, timeout=600) as response, open(part, "wb") as f:
        size = int(response.headers["Content-Length"])
        shutil.copyfileobj(response, f, 1 << 20)
    if part.stat().st_size != size:
        raise OSError(f"incomplete download of {url}")
    part.rename(path)
    return path


def read_events(path: Path) -> dict[str, np.ndarray]:
    """Analysis branches of one file; per-lepton branches become (n, 4) arrays."""
    arrays = uproot.open(path)["analysis"].arrays(
        LEPTON_BRANCHES + EVENT_BRANCHES, library="ak"
    )
    # The exactly4lep skim guarantees four leptons; to_regular fails loudly otherwise.
    events = {b: ak.to_numpy(ak.to_regular(arrays[b], axis=1)) for b in LEPTON_BRANCHES}
    events.update({b: ak.to_numpy(arrays[b]) for b in EVENT_BRANCHES})
    return events


def file_record(path: Path, url: str, events: dict) -> dict:
    """Provenance of one input file: identity, size, checksum and normalization branches."""
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            sha.update(block)
    record = {
        "file": path.name,
        "url": url,
        "bytes": path.stat().st_size,
        "sha256": sha.hexdigest(),
        "entries": len(events["eventNumber"]),
    }
    for branch in ("sum_of_weights", "xsec", "filteff", "kfac"):
        values = np.unique(events[branch])
        record[branch] = values.tolist() if len(values) > 1 else float(values[0])
    return record


def selected_path(did) -> Path:
    return SELECTED_DIR / ("data.npz" if did == "data" else f"mc_{did}.npz")


def save_selected(did, events: dict) -> None:
    """Write the stored columns (and the MC weight, if present) of one dataset."""
    columns = {k: events[k].astype(t) for k, t in STORED_DTYPES.items()}
    if "weight" in events:
        columns["weight"] = events["weight"].astype(np.float64)
    SELECTED_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(selected_path(did), **columns)


def load_selected(dids) -> dict[str, np.ndarray]:
    """Stored events of several datasets, concatenated, with a 'did' column."""
    parts = []
    for did in dids:
        with np.load(selected_path(did)) as f:
            part = {k: f[k] for k in f.files}
        part["did"] = np.full(len(part["eventNumber"]), 0 if did == "data" else did)
        parts.append(part)
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
