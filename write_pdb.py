import numpy as np

from csb.bio.structure import Atom, ProteinResidue, Chain, Structure, Ensemble
from csb.bio.utils import fit_transform


def write_ensemble(X, filename, center=True, align=True):
    """Writes a structure ensemble to a PDB file.

    :param X: coordinates of a structure ensemble. Shape: (n_samples, n_beads, 3)
    :type X: :class:`numpy.ndarray`

    :param filename: file name of output PDB file
    :type filename: str

    :param center: if True, aligns the centers of mass of all structures
    :type center: bool

    :param align: if True, align all structures to the last one
    """

    if center:
        X -= X.mean(1)[:,None,:]

    if align:
        X = np.array([fit_transform(X[0], x) for x in X])

    mol_ranges = np.array([0, X.shape[1]])
    ensemble = Ensemble()

    for i, x in enumerate(X):
        structure = Structure('')
        structure.model_id = i + 1

        mol_coordinates = np.array([x[start:end]
                                    for start, end in zip(mol_ranges[:-1],
                                                          mol_ranges[1:])])
        for j, mol in enumerate(mol_coordinates):
            structure.chains.append(Chain(chr(65 + j)))

            for k, y in enumerate(mol):
                atom = Atom(k+1, 'CA', 'C', y)
                residue = ProteinResidue(k, 'ALA')
                residue.atoms.append(atom)
                structure.chains[chr(65 + j)].residues.append(residue)

        ensemble.models.append(structure)
    ensemble.to_pdb(filename)


def write_VMD_script(ensemble_pdb_file, bead_radius, n_beads,  output_file):
    """Writes a VMD script to show structures

    This writes a VMD script loading a structure ensemble PDB file, setting
    bead radii to given values and showing the structures as a chain of beads.

    :param ensemble_pdb_file: path to PDB file
    :type ensemble_pdb_file: str

    :param bead_radius: bead radius
    :type bead_radius: float

    :param n_beads: number of beads / monomers
    :type n_beads: int

    :param output_file: output file name
    :type output_file: str
    """

    lines = ['color Display Background white',
             'menu main on',
             'menu graphics on',
             'mol load pdb {}'.format(ensemble_pdb_file),
             'mol color Index',
             'mol delrep 0 0',
             'mol representation VDW',
             'mol addrep 0'
             'mol representation trace'
             'mol addrep 0'
            ]

    indices = ' '.join(map(str, range(n_beads)))
    lines.append('set sel [atomselect top "index '+indices+'"]')
    lines.append('$sel set radius {}'.format(bead_radius))
    with open(output_file,'w') as opf:
        [opf.write(line + '\n') for line in lines]
