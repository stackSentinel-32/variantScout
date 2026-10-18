from pathlib import Path

from Bio.PDB import PDBParser
from Bio.PDB.SASA import ShrakeRupley
from Bio.SeqUtils import seq1

STRUCT_DIR = Path(__file__).resolve().parents[1] / "data" / "raw" / "structures" / "ProteinGym_AF2_structures"

MAXASA = dict(A=129, R=274, N=195, D=193, C=167, Q=225, E=223, G=104,
              H=224, I=197, L=201, K=236, M=224, F=240, P=159, S=155,
              T=172, W=285, Y=263, V=174)


def compute_rsa_plddt(pdb_file):
    """Returns {resid: rsa} and {resid: plddt} dicts (1-based residue numbers)."""
    struct = PDBParser(QUIET=True).get_structure("x", STRUCT_DIR / pdb_file)
    model = struct[0]
    ShrakeRupley().compute(model, level="R")

    chain = next(model.get_chains())
    rsa, plddt = {}, {}
    for r in chain:
        if r.id[0] != " ":
            continue
        aa = seq1(r.resname)
        if aa not in MAXASA:
            continue
        rsa[r.id[1]] = r.sasa / MAXASA[aa]
        plddt[r.id[1]] = r["CA"].get_bfactor()
    return rsa, plddt
