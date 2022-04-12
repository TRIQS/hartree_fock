import numpy as np 
from scipy.optimize import root, brentq

class LatticeSolver(object):

    """ Hartree-Fock Lattice solver for local interactions.

    Parameters
    ----------

    e_k : TRIQS BlockGF on a Brillouin zone mesh
        Single-particle dispersion.

    h_int : TRIQS Operator instance
        Local interaction Hamiltonian

    gf_struct : list of pairs [ (str,int), ...]
        Structure of the Green's functions. It must be a
        list of pairs, each containing the name of the
        Green's function block as a string and the size of that block.
        For example: ``[ ('up', 3), ('down', 3) ]``.

    beta : float
        inverse temperature

    """

    def __init__(self, e_k, h_int, gf_struct, beta):

        self.e_k = e_k.copy()
        self.e_k_MF = e_k.copy()
        self.n_k = len(self.e_k.mesh)
        # self.mu = 0

        self.h_int = h_int
        self.gf_struct = gf_struct
        self.beta = beta

        self.Sigma_HF = {bl: np.zeros((bl_size, bl_size), dtype=complex) for bl, bl_size in gf_struct}
        self.rho = {bl: np.zeros((bl_size, bl_size)) for bl, bl_size in gf_struct}

    def solve(self, N_target=None, mu=None, with_fock=True):

        """ Solve for the Hartree Fock self energy using a root finder method.

        Parameters
        ----------

        N_target : optional, float
            target density per site. Can only be provided if mu is not provided

        mu: optional, float
            chemical potential. Can only be provided if N_target is not provided

        with_fock : bool
            True if the fock terms are included in the self energy

        """
        # if mu is None and N_target is None:
        #     raise ValueError('Either mu or N_target must be provided')
        if mu is not None and N_target is not None:
            raise ValueError('Only provide either mu or N_target, not both')
        
        if not N_target is None:
            self.mode = 'fixed_density'
            self.N_target = N_target
        else:
            self.mode = 'fixed_mu'
            if not mu is none:
                self.mu = mu
            else:
                self.mu = 0

        if self.mode == 'fixed_density':
            print('Running Solver at fixed density of %.4f' %self.N_target)
        else:
            print('Running Solver at fixed chemical potential of %.4f' %self.mu)

        #function to pass to root finder
        def f(Sigma_HF_flat):
            self.update_mean_field_dispersion(unflatten(Sigma_HF_flat, self.gf_struct))
            if self.mode == 'fixed_density':
                self.update_mu(self.N_target)
            rho = self.update_rho()
            # print(self.mu)
            Sigma_HF = {bl: np.zeros((bl_size, bl_size), dtype=complex) for bl, bl_size in self.gf_struct}
            for term, coef in self.h_int:
                bl1, u1 = term[0][1]
                bl2, u2 = term[3][1]
                bl3, u3 = term[1][1]
                bl4, u4 = term[2][1]

                assert(bl1 == bl2 and bl3 == bl4)
                Sigma_HF[bl1][u2, u1] += coef * rho[bl3][u4, u3]
                Sigma_HF[bl3][u4, u3] += coef * rho[bl1][u2, u1]

                if bl1 == bl3 and with_fock:
                    Sigma_HF[bl1][u4, u1] -= coef * rho[bl3][u2, u3]
                    Sigma_HF[bl3][u2, u3] -= coef * rho[bl1][u4, u1]
            return Sigma_HF_flat - flatten(Sigma_HF)

        Sigma_HF_init = self.Sigma_HF

        root_finder = root(f, flatten(Sigma_HF_init), method='broyden1')
        self.Sigma_HF = unflatten(root_finder['x'], self.gf_struct)

    def update_mean_field_dispersion(self, Sigma_HF):
        for bl, size in self.gf_struct:
            self.e_k_MF[bl].data[:] = self.e_k[bl].data + Sigma_HF[bl][None, ...]

    def update_rho(self):

        for bl, size in self.gf_struct:
            e, V = np.linalg.eigh(self.e_k_MF[bl].data)
            e -= self.mu

            # density matrix = Sum fermi_function*|psi><psi|
            self.rho[bl] = np.einsum('kab,kb,kcb->ac', V, fermi(e, self.beta), V.conj())/self.n_k
        
        return self.rho

    
    def update_mu(self, N_target):

        energies = {}
        e_min = np.inf
        e_max = -np.inf
        for bl, size in self.gf_struct:
            energies[bl] = np.linalg.eigvalsh(self.e_k_MF[bl].data)
            bl_min = energies[bl].min()
            bl_max = energies[bl].max()
            if bl_min < e_min:
                e_min = bl_min
            if bl_max > e_max:
                e_max = bl_max

        def target_function(mu):
            n = 0
            for bl, size in self.gf_struct:
                n += np.sum(fermi(energies[bl] - mu, self.beta)) / self.n_k
            return n - N_target
        mu = brentq(target_function, e_min, e_max)
        self.mu = mu
        return mu             

def flatten(Sig_HF):
    return np.array([Sig_bl.flatten().view(float) for bl, Sig_bl in Sig_HF.items()]).flatten()

def unflatten(Sig_HF_flat, gf_struct):
    offset = 0
    Sig_HF = {}
    for bl, bl_size in gf_struct:
        Sig_HF[bl] =  Sig_HF_flat[list(range(offset, offset + 2*bl_size**2))].view(complex).reshape((bl_size, bl_size))
        offset = offset + 2*bl_size**2
    return Sig_HF

def fermi(e, beta):
    #numerically stable version
    return np.exp(-beta * e *(e>0))/(1 + np.exp(-beta*np.abs(e)))