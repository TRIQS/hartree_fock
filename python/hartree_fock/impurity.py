import numpy as np 
from scipy.optimize import root
from triqs.gf import *
import triqs.utility.mpi as mpi
from h5.formats import register_class
from .utils import *

class ImpuritySolver(object):
    
    """ Hartree-Fock Impurity solver.

    Parameters
    ----------

    gf_struct : list of pairs [ (str,int), ...]
        Structure of the Green's functions. It must be a
        list of pairs, each containing the name of the
        Green's function block as a string and the size of that block.
        For example: ``[ ('up', 3), ('down', 3) ]``.

    beta : float
        inverse temperature

    n_iw: integer, optional.
        Number of matsubara frequencies in the Matsubara Green's function. Default is 1025. 

    symmeties : optional, list of functions
        symmetry functions acting on self energy at each consistent step

    force_real : optional, bool
        True if the self energy is forced to be real

    """
    def __init__(self, gf_struct, beta, n_iw=1025, symmetries=[], force_real=False):

        self.gf_struct = gf_struct
        self.beta = beta
        self.n_iw = n_iw
        self.symmetries = symmetries
        self.force_real = force_real

        if force_real:
            self.Sigma_HF = {bl: np.zeros((bl_size, bl_size)) for bl, bl_size in gf_struct}
        else:
            self.Sigma_HF = {bl: np.zeros((bl_size, bl_size), dtype=complex) for bl, bl_size in gf_struct}

        name_list = []
        block_list = []
        for bl_name, bl_size in self.gf_struct:
            name_list.append(bl_name)
            block_list.append(GfImFreq(beta=beta, n_points=n_iw, target_shape=[bl_size, bl_size]))
        self.G0_iw = BlockGf(name_list=name_list, block_list=block_list)
        self.G_iw = self.G0_iw.copy()
        self.git_hash = "@PROJECT_GIT_HASH@"

    def solve(self, h_int, with_fock=True, one_shot=False):

        """ Solve for the Hartree Fock self energy using a root finder method.
        The self energy is stored in the ``Sigma_HF`` object of the ImpuritySolver instance.
        The Green's function is stored in the ``G_iw`` object of the ImpuritySolver instance. 

        Parameters
        ----------

        h_int : TRIQS Operator instance
            Local interaction Hamiltonian

        with_fock : optional, bool
            True if the fock terms are included in the self energy. Default is True

        one_shot : optional, bool
            True if the calcualtion is just one shot and not self consistent. Default is False

        """
        self.last_solve_params = {key: val for key, val in locals().items() if key != 'self'}
        mpi.report(logo())
        mpi.report('Running Impurity Solver')
        mpi.report('beta = %.4f' %self.beta)
        mpi.report('h_int =', h_int)
        if one_shot:
            mpi.report('mode: one shot')
        else:
            mpi.report('mode: self-consistent')
        mpi.report('Including Fock terms:', with_fock)

        def f(Sigma_HF_flat):

            Sigma_HF = {bl: np.zeros((bl_size, bl_size), dtype=complex) for bl, bl_size in self.gf_struct}
            G_iw = self.G0_iw.copy()
            G_dens = {}
            Sigma_unflattened = unflatten(Sigma_HF_flat, self.gf_struct, self.force_real)
            for bl, G0_bl in self.G0_iw:
                G_iw[bl] << inverse(inverse(G0_bl) - Sigma_unflattened[bl])
                G_dens[bl] = G_iw[bl].density()
                if self.force_real:
                    max_imag = G_dens[bl].imag.max()
                    if max_imag > 1e-10:
                        mpi.report('Warning! Discarding imaginary part of density matrix. Largest imaginary part: %f'%max_imag)
                    G_dens[bl] = G_dens[bl].real

            for term, coef in h_int:
                bl1, u1 = term[0][1]
                bl2, u2 = term[3][1]
                bl3, u3 = term[1][1]
                bl4, u4 = term[2][1]

                assert(bl1 == bl2 and bl3 == bl4)

                Sigma_HF[bl1][u2, u1] += coef * G_dens[bl3][u4, u3]
                Sigma_HF[bl3][u4, u3] += coef * G_dens[bl1][u2, u1]

                if bl1 == bl3 and with_fock:
                    Sigma_HF[bl1][u4, u1] -= coef * G_dens[bl3][u2, u3]
                    Sigma_HF[bl3][u2, u3] -= coef * G_dens[bl1][u4, u1]
        
            for function in self.symmetries:
                Sigma_HF = function(Sigma_HF)

            if one_shot:
                return Sigma_HF
            return Sigma_HF_flat - flatten(Sigma_HF, self.force_real)

        Sigma_HF_init = self.Sigma_HF

        if one_shot:
            self.Sigma_HF = f(Sigma_HF_init)
            for bl, G0_bl in self.G0_iw:
                self.G_iw[bl] << inverse(inverse(G0_bl) - self.Sigma_HF[bl])

        
        else: #self consistent Hartree-Fock
            root_finder = root(f, flatten(Sigma_HF_init, self.force_real), method='broyden1')
            if root_finder['success']:
                mpi.report('Self Consistent Hartree-Fock converged successfully')
                self.Sigma_HF = unflatten(root_finder['x'], self.gf_struct, self.force_real)
                mpi.report('Calculated self energy:')
                with np.printoptions(suppress=True, precision=4):
                    for name, bl in self.Sigma_HF.items():
                        mpi.report('Sigma_HF[\'%s\']:'%name)
                        mpi.report(bl)
                for bl, G0_bl in self.G0_iw:
                    self.G_iw[bl] << inverse(inverse(G0_bl) - self.Sigma_HF[bl])

            else:
                mpi.report('Hartree-Fock solver did not converge successfully.')
                mpi.report(root_finder['message'])


    def interaction_energy(self):

        """ Calculate the interaction energy

        """
        E = 0        
        for bl, gbl in self.G_iw:
            E += 0.5 * np.trace(self.Sigma_HF[bl].dot(gbl.density()))
        return E

    def __reduce_to_dict__(self):
        print(type(self.git_hash))
        store_dict = {'n_iw': self.n_iw, 'G0_iw': self.G0_iw, 'G_iw': self.G_iw,
                      'gf_struct': self.gf_struct, 'beta': self.beta,
                      'symmetries': self.symmetries, 'Sigma_HF': self.Sigma_HF,
                      'git_hash': self.git_hash}
        return store_dict
        if hasattr(self, 'last_solve_params'):
            store_dict['last_solve_params'] = self.last_solve_params

    @classmethod
    def __factory_from_dict__(cls,name,D) :

        instance = cls(D['gf_struct'], D['beta'], D['n_iw'], D['symmetries'])
        instance.Sigma_HF = D['Sigma_HF']
        instance.G0_iw = D['G0_iw']
        instance.G_iw = D['G_iw']
        instance.git_hash = D['git_hash']
        if 'last_solve_params' in D:
            instance.last_solve_params = D['last_solve_params']

        return instance

register_class(ImpuritySolver)
