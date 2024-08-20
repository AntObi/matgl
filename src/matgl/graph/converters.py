"""Tools to convert materials representations from Pymatgen and other codes to DGLGraphs."""

from __future__ import annotations

import abc

import dgl
import numpy as np
import torch

import matgl


class GraphConverter(metaclass=abc.ABCMeta):
    """Abstract base class for converters from input crystals/molecules to graphs."""

    @abc.abstractmethod
    def get_graph(self, structure) -> tuple[dgl.DGLGraph, torch.Tensor, list]:
        """Args:
        structure: Input crystals or molecule.

        Returns:
        DGLGraph object, state_attr
        """

    def get_graph_from_processed_structure(
        self,
        structure,
        src_id,
        dst_id,
        images,
        lattice_matrix,
        element_types,
        frac_coords,
        species_types=None,
        is_atoms: bool = False,
    ) -> tuple[dgl.DGLGraph, torch.Tensor, list]:
        """Construct a dgl graph from processed structure and bond information.

        Args:
            structure: Input crystals or molecule of pymatgen structure or molecule types.
            src_id: site indices for starting point of bonds.
            dst_id: site indices for destination point of bonds.
            images: the periodic image offsets for the bonds.
            lattice_matrix: lattice information of the structure.
            element_types: Element symbols of all atoms in the structure.
            frac_coords: Fractional coordinates of all atoms in the structure. Note: Cartesian coordinates for molecule
            species_types: Species symbols of all ions in the structure.
                Note: Only one of element_types or species_types can be provided.
            is_atoms: whether the input structure object is ASE atoms object or not.

        Returns:
            DGLGraph object, state_attr

        """
        # Check that only one of element_types or species_types is provided
        assert (element_types is not None) ^ (species_types is not None)

        u, v = torch.tensor(src_id), torch.tensor(dst_id)
        g = dgl.graph((u, v), num_nodes=len(structure))
        # TODO: Need to check if the variable needs to be double or float, now use float
        pbc_offset = torch.tensor(images, dtype=matgl.float_th)
        g.edata["pbc_offset"] = pbc_offset
        # TODO: Need to check if the variable needs to be double or float, now use float
        lattice = torch.tensor(np.array(lattice_matrix), dtype=matgl.float_th)
        # Note: pbc_ offshift and pos needs to be float64 to handle cases where bonds are exactly at cutoff
        if element_types:
            element_to_index = {elem: idx for idx, elem in enumerate(element_types)}
            node_type = (
                np.array([element_types.index(site.specie.symbol) for site in structure])
                if is_atoms is False
                else np.array([element_to_index[elem] for elem in structure.get_chemical_symbols()])
            )
        else:
            species_to_index = {specie: idx for idx, specie in enumerate(species_types)}
            node_type = (
                np.array([species_types.index(site.specie.to_pretty_string()) for site in structure])
                if is_atoms is False
                else np.array([species_to_index[specie] for specie in structure.get_atomic_numbers()])
            )
        g.ndata["node_type"] = torch.tensor(node_type, dtype=matgl.int_th)
        # TODO: Need to check if the variable needs to be double or float, now use float
        g.ndata["frac_coords"] = torch.tensor(frac_coords, dtype=matgl.float_th)
        state_attr = np.array([0.0, 0.0]).astype(matgl.float_np)
        return g, lattice, state_attr
