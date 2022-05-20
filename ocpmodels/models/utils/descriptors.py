from typing import List

import torch
from ase import Atoms
from dscribe.descriptors import CoulombMatrix, EwaldSumMatrix, SineMatrix
from torch_geometric.data import Data
from torch_sparse import SparseTensor


def add_ewaldsum_features(
    data: Data,
    accuracy=1e-5,
    w=1,
    rcut=None,
    gcut=None,
    a=None,
) -> Data:
    """Add the EwaldSum matrix for the given data of a homogeneous graph.
    Args:
        data_list (list of :class:`torch_geometric.data.Data`): One or
            many data graph.
        accuracy (float): The accuracy to which the sum is converged to.
            Corresponds to the variable :math:`A` in
            https://doi.org/10.1080/08927022.2013.840898. Used only if
            gcut, rcut and a have not been specified. Provide either one
            value or a list of values for each system.
        w (float): Weight parameter that represents the relative
            computational expense of calculating a term in real and
            reciprocal space. This has little effect on the total energy,
            but may influence speed of computation in large systems. Note
            that this parameter is used only when the cutoffs and a are set
            to None. Provide either one value or a list of values for each
            system.
        rcut (float): Real space cutoff radius dictating how many terms are
            used in the real space sum. Provide either one value or a list
            of values for each system.
        gcut (float): Reciprocal space cutoff radius. Provide either one
            value or a list of values for each system.
        a (float): The screening parameter that controls the width of the
            Gaussians.
    Returns:
        torch_geometric.data: Updated data object with EwaldSum matrix `.ewaldsum` for the given data.
    """

    assert (
        hasattr(data, "atomic_numbers")
        and hasattr(data, "pos")
        and hasattr(data, "cell")
    )

    n_atoms = len(data.atomic_numbers)
    esm = EwaldSumMatrix(
        n_atoms_max=n_atoms, flatten=False, permutation="none"
    )
    atoms = Atoms(
        positions=data.pos,
        numbers=data.atomic_numbers,
        cell=data.cell[0],
        pbc=True,
    )

    ewaldsum_matrix = esm.create_single(
        atoms, accuracy, w, rcut, gcut, a
    ).reshape((n_atoms, n_atoms))

    data.ewaldsum = SparseTensor.from_dense(
        torch.tensor(ewaldsum_matrix, dtype=torch.float32)
    )

    return data


def add_sine_features(
    data: List[Data],
) -> List[Data]:
    """Add the Sine matrix for the given data of a homogeneous graph.
    Args:
        data (:class:`torch_geometric.data.Data`): data graph.
    Returns:
        torch_geometric.data: Updated data object with Sine matrix `.sine` for the given datas.
    """

    assert (
        hasattr(data, "atomic_numbers")
        and hasattr(data, "pos")
        and hasattr(data, "cell")
    )

    n_atoms = len(data.atomic_numbers)
    sm = SineMatrix(n_atoms_max=n_atoms, flatten=False, permutation="none")
    atoms = Atoms(
        positions=data.pos,
        numbers=data.atomic_numbers,
        cell=data.cell[0],
        pbc=True,
    )

    # Combine input arguments
    sine = sm.create_single(atoms).reshape((n_atoms, n_atoms))

    data.sine = SparseTensor.from_dense(
        torch.tensor(sine, dtype=torch.float32)
    )

    return data


def add_coulomb_features(
    data: Data,
) -> Data:
    """Add the Coulomb matrix for the given data of a homogeneous graph.
    Args:
        data (:class:`torch_geometric.data.Data`): data graph.
    Returns:
        torch_geometric.data: Updated data object with Coulomb matrix `.coulomb` for the given datas.
    """

    assert (
        hasattr(data, "atomic_numbers")
        and hasattr(data, "pos")
        and hasattr(data, "cell")
    )

    n_atoms = len(data.atomic_numbers)
    cm = CoulombMatrix(n_atoms_max=n_atoms, flatten=False, permutation="none")
    atoms = Atoms(
        positions=data.pos,
        numbers=data.atomic_numbers,
        cell=data.cell[0],
        pbc=True,
    )

    # Combine input arguments
    coulomb = cm.create_single(atoms).reshape((n_atoms, n_atoms))

    data.coulomb = SparseTensor.from_dense(
        torch.tensor(coulomb, dtype=torch.float32)
    )

    return data
