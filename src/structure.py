import io
from pathlib import Path
from typing import Dict, Tuple, Union

import numpy as np
from Bio.PDB import PDBParser
from Bio.PDB.SASA import ShrakeRupley
from Bio.SeqUtils import seq1

STRUCT_DIR: Path = Path(__file__).resolve().parents[1] / "data" / "raw" / "structures" / "ProteinGym_AF2_structures"

MAXASA: Dict[str, float] = dict(
    A=129, R=274, N=195, D=193, C=167, Q=225, E=223, G=104,
    H=224, I=197, L=201, K=236, M=224, F=240, P=159, S=155,
    T=172, W=285, Y=263, V=174
)


def compute_rsa_plddt(pdb_source: Union[str, Path]) -> Tuple[Dict[int, float], Dict[int, float]]:
    """Returns {resid: rsa} and {resid: plddt} dicts (1-based residue numbers).

    pdb_source can be a Path, a file path string, or raw PDB text.
    """
    if isinstance(pdb_source, Path) and pdb_source.is_file():
        struct = PDBParser(QUIET=True).get_structure("x", pdb_source)
    elif isinstance(pdb_source, str) and (pdb_source.endswith(".pdb") or "\n" not in pdb_source):
        p = Path(pdb_source)
        if not p.is_file():
            p = STRUCT_DIR / pdb_source
        struct = PDBParser(QUIET=True).get_structure("x", p)
    else:
        struct = PDBParser(QUIET=True).get_structure("x", io.StringIO(str(pdb_source)))

    model = struct[0]
    ShrakeRupley().compute(model, level="R")

    chain = next(model.get_chains())
    rsa: Dict[int, float] = {}
    plddt: Dict[int, float] = {}
    for r in chain:
        if r.id[0] != " ":
            continue
        aa = seq1(r.resname)
        if aa not in MAXASA:
            continue
        rsa[r.id[1]] = float(r.sasa / MAXASA[aa])
        if "CA" in r:
            plddt[r.id[1]] = float(r["CA"].get_bfactor())
    return rsa, plddt


def extract_ca_coords(pdb_source: Union[str, Path]) -> Dict[int, np.ndarray]:
    """Extract C-alpha atom 3D coordinates for all residues.

    pdb_source can be a Path, a file path string, or raw PDB text.
    Returns {residue_number (1-based): np.ndarray([x, y, z])}.
    """
    if isinstance(pdb_source, Path) and pdb_source.is_file():
        struct = PDBParser(QUIET=True).get_structure("x", pdb_source)
    elif isinstance(pdb_source, str) and (pdb_source.endswith(".pdb") or "\n" not in pdb_source):
        p = Path(pdb_source)
        if not p.is_file():
            p = STRUCT_DIR / pdb_source
        struct = PDBParser(QUIET=True).get_structure("x", p)
    else:
        struct = PDBParser(QUIET=True).get_structure("x", io.StringIO(str(pdb_source)))

    model = struct[0]
    chain = next(model.get_chains())
    ca_coords: Dict[int, np.ndarray] = {}
    for r in chain:
        if r.id[0] != " ":
            continue
        if "CA" in r:
            ca_coords[r.id[1]] = np.asarray(r["CA"].get_coord(), dtype=np.float32)
    return ca_coords


def compute_ca_distance(coord1: np.ndarray, coord2: np.ndarray) -> float:
    """Euclidean distance between two 3D coordinates in Angstroms."""
    return float(np.linalg.norm(coord1 - coord2))

