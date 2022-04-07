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

        self.h_int = h_int
        self.gf_struct = gf_struct
        self.beta = beta

        self.Sigma_HF = {bl: np.zeros((bl_size, bl_size), dtype=np.complex_) for bl, bl_size in gf_struct}
        self.rho = {bl: np.zeros((bl_size, bl_size)) for bl, bl_size in gf_struct}

    def solve(self, N_target, with_fock=True):

        """ Solve for the Hartree Fock self energy using a root finder method.

        Parameters
        ----------

        N_target : float
            target density per site

        with_fock : bool
            True if the fock terms are included in the self energy

        """

        self.N_target = N_target

        #function to pass to root finder
        def f(Sigma_HF_flat):
            Sigma_HF = unflatten(Sigma_HF_flat, self.gf_struct)
            self.update_mean_field_dispersion(Sigma_HF)
            self.update_mu(self.N_target)
            rho = self.update_rho()
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
            print(flatten(Sigma_HF))
            return Sigma_HF_flat - flatten(Sigma_HF)

        Sigma_HF_init = self.Sigma_HF

        root_finder = root(f, flatten(Sigma_HF_init), method='lm')
        self.Sigma_HF = unflatten(root_finder['x'], self.gf_struct)
        print(root_finder['success'])
        print(root_finder['message'])

    def update_mean_field_dispersion(self, Sigma_HF):
        for bl, size in self.gf_struct:
            self.e_k_MF[bl].data[:] = self.e_k[bl].data + Sigma_HF[bl][None, ...]

    def update_rho(self):

        fermi = lambda e : 1./(np.exp(self.beta * e) + 1)
        for bl, size in self.gf_struct:
            e, V = np.linalg.eigh(self.e_k_MF[bl].data)
            e -= self.mu

            # density matrix = Sum fermi*|psi><psi|
            self.rho[bl] = np.einsum('kab,kb,kcb->ac', V, fermi(e), V.conj())/self.n_k
        
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
        fermi = lambda e : 1./(np.exp(self.beta * e) + 1)

        def target_function(mu):
            n = 0
            for bl, size in self.gf_struct:
                n += np.sum(fermi(energies[bl] - mu)) / self.n_k
            return n - N_target
        mu = brentq(target_function, e_min, e_max)
        # print('mu = ', mu)
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
