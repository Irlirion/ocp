import math
from pickle import dumps

import lmdb
import numpy as np
import pymatgen.analysis.local_env as pmg_le
import torch
from ase import Atoms
from matminer.featurizers.structure.matrix import (
    ANG_TO_BOHR,
    OrbitalFieldMatrix,
)
from mpire import WorkerPool
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from torch_sparse import SparseTensor

from ocpmodels.datasets.lmdb_dataset import LmdbDataset
from ocpmodels.models.utils.descriptors import (
    add_coulomb_features,
    add_ewaldsum_features,
    add_sine_features,
)


def get_atom_ofms(ofm, struct, symm=False, catoff=10):
    ofms = []
    vnn = pmg_le.VoronoiNN(allow_pathological=True, cutoff=catoff)

    if symm:
        symm_struct = SpacegroupAnalyzer(struct).get_symmetrized_structure()
        indices = [lst[0] for lst in symm_struct.equivalent_indices]
        counts = [len(lst) for lst in symm_struct.equivalent_indices]
    else:
        indices = list(range(len(struct.sites)))
    for index in indices:
        try:
            site_dict = vnn.get_nn_info(struct, index)
        except ValueError:
            site_dict = {}

        ofms.append(get_single_ofm(ofm, struct.sites[index], site_dict))
    if symm:
        return ofms, counts
    return ofms


def get_single_ofm(ofm, site, site_dict):
    """
    Gets the orbital field matrix for a single chemical environment,
    where site is the center atom whose environment is characterized and
    site_dict is a dictionary of site : weight, where the weights are the
    Voronoi Polyhedra weights of the corresponding coordinating sites.
    Args:
        site (Site): center atom
        site_dict (dict of Site:float): chemical environment
    Returns:
        atom_ofm (size X size numpy matrix): ofm for site
    """
    ohvs = ofm.ohvs
    atom_ofm = np.matrix(np.zeros((ofm.size, ofm.size)))
    ref_atom = ohvs[site.specie.Z]
    for other_site in site_dict:
        scale = other_site["weight"]
        other_atom = ohvs[other_site["site"].specie.Z]
        d = site.distance(other_site["site"])
        if not math.isclose(d, 0.0, abs_tol=1e-08, rel_tol=1e-05):
            atom_ofm += other_atom.T * ref_atom * scale / d / ANG_TO_BOHR
    return atom_ofm


def add_ofms(data):
    atoms = Atoms(
        positions=data.pos, numbers=data.atomic_numbers, cell=data.cell[0]
    )
    pmg_atoms = AseAtomsAdaptor().get_structure(atoms)

    ofm = OrbitalFieldMatrix()
    atoms_ofms = get_atom_ofms(ofm, pmg_atoms, symm=False, catoff=13)
    atoms_ofms = np.array(atoms_ofms).reshape(-1, 32 * 32)
    # atoms_ofms = sparse.csr_matrix(atoms_ofms)
    data.ofms = SparseTensor.from_dense(
        torch.tensor(atoms_ofms, dtype=torch.float32)
    )
    return data


def get_descriptors(shared_objects, i):
    ds = shared_objects
    data = ds[i]
    data = add_coulomb_features(data)
    data = add_sine_features(data)
    data = add_ewaldsum_features(data)
    data = add_ofms(data)

    return dumps(data)


def preprocess_chanks(ds, f, out_src, n_jobs):
    ds_len = len(ds)
    map_size = 214748364800 * 4
    with WorkerPool(
        n_jobs=n_jobs, shared_objects=ds, keep_alive=False
    ) as pool:
        train_mols_chank = pool.map(
            f,
            range(ds_len),
            progress_bar=True,
        )

    env = lmdb.open(out_src, map_size=map_size)
    with env.begin(write=True) as txn:
        for k, j in enumerate(range(ds_len)):
            # All key-value pairs need to be Strings
            value = train_mols_chank[k]
            key = f"{j}"
            txn.put(key.encode("ascii"), value)
    env.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-lmdb",
        "-i",
        help="Path to `data.lmdb`.",
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        help="Output dir for processed files.",
        required=True,
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        help="Number of parallel processes.",
        default=1,
    )

    args = parser.parse_args()

    data_src = args.input_lmdb
    out_src = args.output_dir
    n_jobs = args.n_jobs
    ds = LmdbDataset({"src": data_src})
    print(f"Total: {len(ds)}")
    preprocess_chanks(ds, get_descriptors, out_src, n_jobs=n_jobs)
    print("Done")
