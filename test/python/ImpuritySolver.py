# Copyright (c) 2022 Simons Foundation
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You may obtain a copy of the License at
#     https:#www.gnu.org/licenses/gpl-3.0.txt
#
# Authors: Jonathan Karp, Alexander Hampel, Nils Wentzell, Hugo U. R. Strand, Olivier Parcollet

import unittest
import functools
import numpy as np
from triqs_hartree_fock import ImpuritySolver, LatticeSolver
from triqs_hartree_fock.utils import flatten
from triqs.gfs import *
from triqs.mesh import MeshImFreq, MeshImTime, MeshDLRImFreq
from triqs.operators import *
from h5 import HDFArchive
from triqs.lattice.tight_binding import TBLattice
from triqs.sumk import *
from triqs.lattice import *


# Fixed problem definition shared by all tests
t = 1
tp = 0.1
hop = {(1, 0):  [[t]],
       (-1, 0):  [[t]],
       (0, 1):  [[t]],
       (0, -1):  [[t]],
       (1, 1):  [[tp]],
       (-1, -1):  [[tp]],
       (1, -1):  [[tp]],
       (-1, 1):  [[tp]]}

TBL = TBLattice(units=[(1, 0, 0), (0, 1, 0)], hoppings=hop, orbital_positions=[(0., 0., 0.)]*1)
nk = 10
beta = 40
h_int = 3*n('up', 0)*n('down', 0)
gf_struct = [('up', 1), ('down', 1)]


def solve_impurity(mesh):
    """Run the self-consistent impurity solve for a given mesh and return Sigma_HF
    after an h5 round-trip (so storage/reload is exercised for that mesh)."""
    SK = SumkDiscreteFromLattice(lattice=TBL, n_points=nk)
    # External Sigma/Gloc use MeshImFreq for compatibility with SumkDiscreteFromLattice
    sigma = GfImFreq(beta=beta, n_points=1025, target_shape=[1, 1])
    Sigma = BlockGf(name_list=['up', 'down'], block_list=(sigma, sigma), make_copies=True)
    Gloc = Sigma.copy()
    mu = 0
    S = ImpuritySolver(gf_struct=gf_struct, mesh=mesh)

    converged = False
    while not converged:
        for name, _ in gf_struct:
            Sigma[name] << S.Sigma_HF[name]
        Gloc << SK(mu=mu, Sigma=Sigma)
        S.set_G0_iw(Gloc)
        Sigma_old = S.Sigma_HF.copy()
        S.solve(h_int=h_int, one_shot=False, tol=1e-4)
        if np.allclose(flatten(Sigma_old), flatten(S.Sigma_HF), rtol=0, atol=1e-6):
            converged = True

    # test storing to and loading from h5
    with HDFArchive('impurity_results.h5', 'w') as ar:
        ar['solver'] = S
    with HDFArchive('impurity_results.h5', 'r') as ar:
        S = ar['solver']

    return S.Sigma_HF


@functools.lru_cache(maxsize=None)
def lattice_reference():
    """Sigma_HF from the LatticeSolver; depends only on the fixed parameters, so solve once."""
    BL = BravaisLattice(units=[(1, 0, 0), (0, 1, 0)])
    BZ = BrillouinZone(BL)
    mk = MeshBrZone(BZ, nk)
    ekup = Gf(mesh=mk, target_shape=[1, 1])
    ekdn = Gf(mesh=mk, target_shape=[1, 1])
    ekup << TBL.fourier(mk)
    ekdn << TBL.fourier(mk)
    h0_k = BlockGf(name_list=['up', 'down'], block_list=(ekup, ekdn))
    S = LatticeSolver(h0_k=h0_k, gf_struct=gf_struct, beta=beta)
    S.solve(h_int=h_int)
    return S.Sigma_HF


class test_impurity_solver(unittest.TestCase):

    # test that lattice and impurity solvers agree, for both the DLR and the
    # full Matsubara mesh of the ImpuritySolver
    def test_agreement_dlr(self):
        mesh = MeshDLRImFreq(beta, 'Fermion', w_max=10.0, eps=1e-10, symmetrize=True)
        np.testing.assert_allclose(flatten(lattice_reference()), flatten(solve_impurity(mesh)), rtol=0, atol=1e-4)

    def test_agreement_imfreq(self):
        mesh = MeshImFreq(beta=beta, statistic='Fermion', n_iw=1025)
        np.testing.assert_allclose(flatten(lattice_reference()), flatten(solve_impurity(mesh)), rtol=0, atol=1e-4)

    def test_mesh_validation(self):
        with self.assertRaises(TypeError):  # not a mesh (e.g. a pre-4.0 positional beta)
            ImpuritySolver(gf_struct, beta)
        with self.assertRaises(TypeError):  # unsupported mesh type
            ImpuritySolver(gf_struct, MeshImTime(beta=beta, statistic='Fermion', n_tau=101))
        with self.assertRaises(ValueError):  # wrong statistic
            ImpuritySolver(gf_struct, MeshImFreq(beta=beta, statistic='Boson', n_iw=128))


if __name__ == '__main__':
    unittest.main()
