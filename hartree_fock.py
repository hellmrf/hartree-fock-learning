from pprint import pp
from typing import Sequence, TypeAlias

import numpy as np


class PrimitiveGaussian:

    def __init__(self,
                 alpha: float,
                 coeff: float,
                 coords: Sequence[int | float],
                 l1=0,
                 l2=0,
                 l3=0):
        self.alpha: float = alpha
        self.coeff: float = coeff
        self.A: float = (2 * alpha /
                         np.pi)**0.75  # TODO: terms for l1, l2, l3 > 0
        self.coords: np.ndarray = np.array(coords)
        self.l1: float = l1
        self.l2: float = l2
        self.l3: float = l3

    def __repr__(self):
        return f'PrimitiveGaussian(alpha = {self.alpha}, coeff = {self.coeff}, coords = {self.coords})'

    # def __call__(self, r):
    #     return self.A * np.exp(-self.alpha * np.dot(r - self.a, r - self.a)) #  Copilot suggestion


Atom: TypeAlias = Sequence[PrimitiveGaussian]
Molecule: TypeAlias = Sequence[Atom]


def overlap_atoms(atom1: Atom, atom2: Atom) -> float:
    """Returns the overlap between two Atoms."""

    overlap = 0
    for gaussian_i in atom1:
        for gaussian_j in atom2:
            # TODO: read about Gaussian product theorem
            """
            https://en.wikipedia.org/wiki/Gaussian_function#Product_of_two_Gaussian_functions
            https://en.wikipedia.org/wiki/Overlap_integral
            https://en.wikipedia.org/wiki/Slater-type_orbital
            """

            # N = normalization constant
            N = gaussian_i.A * gaussian_j.A

            # XXX: alphas normalized ratio?
            p = gaussian_i.alpha + gaussian_j.alpha
            q = gaussian_i.alpha * gaussian_j.alpha / p

            # Q = distance between the two atoms
            Q = gaussian_i.coords - gaussian_j.coords
            Q2 = np.dot(Q, Q)

            # computing the overlap between two gaussians for s orbitals
            # if needed to compute for p orbitals, we need to add the terms for l1, l2, l3 > 0
            overlap += N * gaussian_i.coeff * gaussian_j.coeff * np.exp(
                -q * Q2) * (np.pi / p)**(3 / 2)

    return overlap


def overlap_matrix(molecule: Sequence[Atom]) -> np.ndarray:
    """
    Passa pelos átomos da molécula e calcula a matriz de sobreposição S entre eles.
    2 átomos -> matriz 2x2. 3 átomos -> matriz 3x3. E assim por diante.
    """
    n_basis = len(molecule)
    S = np.zeros((n_basis, n_basis))

    for i in range(n_basis):
        for j in range(n_basis):
            """cada átomo é um conjunto de gaussianas (funções de base).

            no nosso caso, temos que cada átomo é uma lista de 3 PrimitiveGaussians. Então 
            faremos a superposição desses dois átomos.
            Entretanto, como cada átomo é uma lista de gaussianas, precisamos fazer a 
            superposição das gaussianas independentemente.

            overlap_atoms(atom1, atom2) vai resolver isso.
            """

            S[i, j] = overlap_atoms(molecule[i], molecule[j])

    return S


def sto_3g():
    """   
            H2 molecule, with two H atoms at (0, 0, 0) and (1.2, 0, 0). 
    """
    # Basis set from https://www.basissetexchange.org/basis/sto-3g/format/gaussian94/?version=1&elements=1
    STO_3G = [[0.3425250914e1, 0.1543289673], [0.6239137298, 0.5353281423],
              [0.1688554040, 0.4446345422]]

    # First H, centered at (0, 0, 0)
    H1_1s = [
        PrimitiveGaussian(STO_3G[i][0], STO_3G[i][1], [0, 0, 0])
        for i in range(3)
    ]
    # Second H, centered at (1.2, 0, 0)
    H2_1s = [
        PrimitiveGaussian(STO_3G[i][0], STO_3G[i][1], [1.2, 0, 0])
        for i in range(3)
    ]

    molecule = [H1_1s, H2_1s]

    S = overlap_matrix(molecule)

    print("H1_1s: ")
    pp(H1_1s)
    print("H2_1s: ")
    pp(H2_1s)
    print("\nOverlap matrix:")
    pp(S)


def _6_31G():
    """   
            H2 molecule, with two H atoms at (0, 0, 0) and (1.2, 0, 0). 
    """
    # Basis set from https://www.basissetexchange.org/basis/6-31g/format/gaussian94/?version=1&elements=1

    _6_31G_1s = [[0.1873113696e2, 0.3349460434e-1],
                 [0.2825394365e1, 0.2347269535], [0.6401216923, 0.8137573261]]

    _6_31G_2s = [[0.1612777588, 1]]

    # First H, centered at (0, 0, 0)

    molecule = []
    N_ATOMS = 2

    for i in range(N_ATOMS):
        _1s = [
            PrimitiveGaussian(_6_31G_1s[j][0], _6_31G_1s[j][1],
                              [i * 1.2, 0, 0]) for j in range(3)
        ]
        molecule.append(_1s)

        _2s = [PrimitiveGaussian(_6_31G_2s[0][0], 1, [i * 1.2, 0, 0])]
        molecule.append(_2s)

    # molecule = [H1_1s, H2_1s]

    S = overlap_matrix(molecule)

    print("molecule: ")
    pp(molecule)
    # print("H2_1s: ")
    # pp(H2_1s)
    print("\nOverlap matrix:")
    pp(S)


if __name__ == '__main__':

    print("\n", "=" * 50, "\n", "=" * 50, "\n", " " * 10, f"STO-3G", "\n",
          "=" * 50, "\n", "=" * 50)

    sto_3g()

    print("\n", "=" * 50, "\n", "=" * 50, "\n", " " * 10, f"6-31G", "\n",
          "=" * 50, "\n", "=" * 50)

    _6_31G()
