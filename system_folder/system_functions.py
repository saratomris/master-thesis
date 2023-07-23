import numpy as np
import sympy as sp
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scipy as sc
from scipy import optimize
from scipy.integrate import quad
import tqdm
import tqdm.auto
from copy import deepcopy
import time
import os
from os import listdir
from fractions import Fraction
from matplotlib.ticker import MultipleLocator
from pathlib import Path
from chebpy import chebfun

# %%
""" Variables """
# --- units ---
mili = 10**(-3)

# --- independent variables ---
n = 5
n_chains = 2
a_ = 1   ## Lattice constant
J = 1*mili   ## Exchange in magnet
t_ = 1   ## hopping amplitude
K = 0   ## Easy axis anisotropy
N_ = 1  ## Lattice sites
S_ = 1  ## Spin
J_L = 50*mili   ## Exchange at interface

decs = 15   ## Endra fra 17
lattice_angle=np.pi/3   ## Triangular lattice
a1_e = np.array([1,0])
a2_e = np.array([np.cos(lattice_angle), np.sin(lattice_angle)])

b1_electron = 4*np.pi/np.sqrt(3)*np.array([np.cos(-lattice_angle/2), np.sin(-lattice_angle/2)])
b2_electron = 4*np.pi/np.sqrt(3)*np.array([0,1])


# --- dependent variables ---
m = n*n_chains

b_mag_electron = np.linalg.norm(b1_electron)
s = b_mag_electron/(2*np.cos(np.pi/6))
AeBZ = 3*np.sqrt(3)*s**2/2

BZ1_electron = b1_electron/2
BZ2_electron = b2_electron/2

b1_magnon = 2*np.pi/n*np.array([1,0])
b2_magnon = 2*np.pi/np.sqrt(3)*np.array([0,1])
b1_mag_magnon = np.linalg.norm(b1_magnon)
b2_mag_magnon = np.linalg.norm(b2_magnon)
BZ_x = b1_mag_magnon/2
BZ_y = b2_mag_magnon/2
d_vec = np.around(np.array([
    [np.cos(lattice_angle - i*lattice_angle), 
     np.sin(lattice_angle - i*lattice_angle)] for i in range(6)]), decimals=15)   ## NB: Only valid for the stated ordering of nn_indices
                                                                                  ## NB2: Veldig viktig at decimals=15 her
J_matrix = np.array([
    [J, 0, 0],
    [0, J, 0],
    [0, 0, J]
])


# %%
""" functions related to the lattice"""

# --- create-functions ---
def create_realspace_lattice(ncols=n, nrows=n_chains, a1_e_=a1_e, a2_e_=a2_e):
    realspace_lattice = np.zeros((nrows,ncols,2))
    for i in range(1,ncols):
        realspace_lattice[0,i] = i*a1_e_
        realspace_lattice[1,i] = a2_e_ + i*a1_e_
    realspace_lattice[1,0] = a2_e_
    return realspace_lattice


def create_spin_lattice(ncols=n, nrows=n_chains):
    """ Creates spin lattice for theta in [0,pi], psi in {0,pi}.
        Gives spin angles to global z-axis in dim*dim triangular lattice. """
    spin_lattice = np.zeros((nrows,ncols,2))
    Dth=2*np.pi/ncols
    for r in range(nrows):
        toggle=0
        if r%2==1: # shift in spin for every second chain
            toggle=Dth/2
        for c in range(ncols):
            spin_lattice[r,c] = [(Dth*c+toggle)%(2*np.pi), 0]
    return spin_lattice


def create_Qnu_set(period, nrows=n_chains):
    """ Creates set of magnon reciprocal lattice vectors """
    assert ((b1_magnon == 2*np.pi/n*np.array([1,0])).all() and (b2_magnon == 2*np.pi/np.sqrt(3)*np.array([0,1])).all()), \
            "Sorry, this code is not generalised for other SPFMI reciprocal lattice vectors yet"
    assert (period == 5), f"period=5 is the relevant spiral periodicity"
    if period == 5:
        return np.array([[ 0,0],
                         [ 2*np.pi/period, 0],
                         [-2*np.pi/period, 0],
                         [ 4*np.pi/period, 0],
                         [-4*np.pi/period, 0],
                         [ 6*np.pi/period, 0],
                         [-6*np.pi/period, 0],
                         [ 0, 2*np.pi/np.sqrt(3)],
                         [ 2*np.pi/period, 2*np.pi/np.sqrt(3)],
                         [-2*np.pi/period, 2*np.pi/np.sqrt(3)]
                         ])

# --- storage dictionary to avoid recalculations ---
SL = create_spin_lattice(ncols=5, nrows=n_chains)
SL_tmp = np.zeros_like(SL)

for r in range(n_chains):
    for c in range(n):
        SL_tmp[r,c, :] = SL[r,c, :] + [0.2*np.sign(np.around(np.cos(SL[r,c,0])*np.sin(SL[r,c,0]), 10)), 0]

storage = {(5,n_chains) : {"spin lattice": SL,
                           "real space lattice" : create_realspace_lattice(ncols=5, nrows=n_chains),
                           "Qnu" : create_Qnu_set(period=5, nrows=n_chains)},
            }

# --- get-functions --- 
def get_col(idx, period=n):
    return idx%period

def get_row(idx, period=n, nrows=n_chains):
    return (idx//period)%nrows

def get_idx_from_rc(r,c, period=n, nrows=n_chains):
    return abs(r%nrows)*period+c + period*bool(c<0) - period*bool(c>=period)

def get_nearest_neighbour_indices(idx, period=n, nrows=n_chains):
    """ first element is the top right nearest neighbour, 
        going clockwise to top left nearest neighbour in last element """
    ri = get_row(idx, period, nrows)
    ci = get_col(idx, period)
    j  =[get_idx_from_rc(ri-1,ci +(ri%2),period,nrows), 
         get_idx_from_rc(ri,ci+1,period,nrows), 
         get_idx_from_rc(ri+1,ci +(ri%2),period,nrows), 
         get_idx_from_rc(ri+1,ci-1 +(ri%2),period,nrows), 
         get_idx_from_rc(ri,ci-1,period,nrows), 
         get_idx_from_rc(ri-1,ci-1 +(ri%2),period,nrows)
        ]
    return np.array(j)

def get_spin_from_lattice_idx(idx,period=n, nrows=n_chains): 
    return storage[(period,nrows)]["spin lattice"][get_row(idx, period, nrows), get_col(idx, period)]

def get_rL_from_lattice_idx(idx,period=n, nrows=n_chains): 
    return storage[(period,nrows)]["real space lattice"][get_row(idx, period, nrows), get_col(idx, period)]

def get_Qnu_set(period, nrows=n_chains):
    """ get-function for create_Qnu_set() to avoid recalculations """
    return storage[(period,nrows)].get("Qnu")

def get_DMI_magnitude(Jy=J, period=n):
    """ Gives the magnitude of the DMI vector, calculated from the ground state energy """
    Dtheta=2*np.pi/period
    return Jy*(np.sin(Dtheta) + np.sin(Dtheta/2))/np.cos(Dtheta)   ## Note: We require D-vector = (0, Dy, 0)

# %%
""" lattice-dependent quantities """

def U_rotmat(idx, period=n, nrows=n_chains):
    """ Rotation matrix to specify each spin in their local spin axes """
    if type(storage[(period,nrows)].get(("U", idx))) != np.ndarray:
        theta,psi=get_spin_from_lattice_idx(idx, period, nrows)
        storage[(period,nrows)][("U",idx)] = np.array([[np.cos(theta)*np.cos(psi), np.cos(theta)*np.sin(psi), -np.sin(theta)],
                     [-np.sin(psi), np.cos(psi), 0],
                     [np.sin(theta)*np.cos(psi), np.sin(theta)*np.sin(psi), np.cos(theta)]
                    ])
    return storage[(period,nrows)].get(("U", idx))

def H_hamblockmat(Jmat=J_matrix, ncols=n, nrows=n_chains):
    """ Interaction matrix """
    
    if type(storage[(ncols,nrows)].get("Hvxv")) != np.ndarray:
        Dx = 0
        Dy = get_DMI_magnitude(Jmat[1,1], ncols)
        Dz = 0

        n_idcs = ncols*nrows   ## no. indices of unit cell
        H_block = np.zeros((n_idcs,n_idcs,3,3)) # (n_idcs)x(n_idcs) block matrix with 3x3 matrices as elements

        for i in range(n_idcs):
            for j in get_nearest_neighbour_indices(i,ncols,nrows): 
                ## Diagonal elements
                H_block[i,j,0,0] = Jmat[0,0]
                H_block[i,j,1,1] = Jmat[1,1]
                H_block[i,j,2,2] = Jmat[2,2]

                ## Off-diagonal elements
                sgn=-1
                if get_col(j, ncols) == (get_col(i, ncols)+1)%ncols:   ## to set the sign of Dij = -Dji
                    sgn=+1
                toggle=1   ## to ensure that Dij only acts between nn on same chain (same row)
                if get_row(j, ncols, nrows) != get_row(i, ncols, nrows):
                    toggle=0
                H_block[i,j,0,1] =  Dz*toggle*sgn + Jmat[0,1]
                H_block[i,j,0,2] = -Dy*toggle*sgn + Jmat[0,2]
                H_block[i,j,1,0] = -Dz*toggle*sgn + Jmat[1,0]
                H_block[i,j,1,2] =  Dx*toggle*sgn + Jmat[1,2]
                H_block[i,j,2,0] =  Dy*toggle*sgn + Jmat[2,0]
                H_block[i,j,2,1] = -Dx*toggle*sgn + Jmat[2,1]
        storage[(ncols,nrows)]["Hvxv"] = H_block
        return H_block
    return storage[(ncols,nrows)].get("Hvxv")


def W_rotatedvxvmat(period=n,nrows=n_chains, Jmat=J_matrix):
    """ Rotated interaction matrix """
    
    if type(storage[(period,nrows)].get("W")) != np.ndarray:
        n_idcs = period*nrows   ## no. indices of unit cell
        Wij = np.zeros((n_idcs,n_idcs,3,3)) # (n_idcs)x(n_idcs) block matrix with 3x3 matrices as elements
        H_ = H_hamblockmat(Jmat, period,nrows)

        for i in range(n_idcs):
            for j in np.unique(get_nearest_neighbour_indices(i,period,nrows)): ## takes nearest neighbours into account
                Wij[i,j] = (U_rotmat(i,period,nrows)@H_[i,j])@np.transpose(U_rotmat(j,period,nrows))   ## Used to have np.around() with decimals=16
        storage[(period,nrows)]["W"] = Wij
        return Wij
    return storage[(period,nrows)].get("W")


# --- Real space coefficients of the spin-wave Hamiltonian ---
def C1_matrix(period=n,nrows=n_chains, Jmat=J_matrix):
    """ Coefficients C1 of some boson-operators in the spin-wave SPFMI Hamiltonian """
    
    if type(storage[(period,nrows)].get(("C1"))) != np.ndarray:
        n_idcs = period*nrows   ## no. indices of unit cell
        C1ij = np.zeros((n_idcs,n_idcs), dtype=np.complex_) # (dim*dim)x(dim*dim) matrix
        W = W_rotatedvxvmat(period,nrows, Jmat)
        for r in range(n_idcs):
            for c in get_nearest_neighbour_indices(r,period,nrows):
                C1ij[r,c] = -1/2*(W[r,c,0,0]-W[r,c,1,1]-1j*(W[r,c,0,1]+W[r,c,1,0]))
        storage[(period,nrows)][("C1")] = C1ij
        return C1ij
    return storage[(period,nrows)][("C1")]

def C2_matrix(period=n,nrows=n_chains, Jmat=J_matrix):
    """ Coefficients C2 of some boson-operators in the spin-wave SPFMI Hamiltonian """
    
    if type(storage[(period,nrows)].get(("C2"))) != np.ndarray:
        n_idcs = period*nrows   ## no. indices of unit cell
        C2ij = np.zeros((n_idcs,n_idcs), dtype=np.complex_) # (dim*dim)x(dim*dim) matrix
        W = W_rotatedvxvmat(period,nrows, Jmat)
        for r in range(n_idcs):
            for c in get_nearest_neighbour_indices(r,period,nrows):
                C2ij[r,c] = -1/2*(W[r,c,0,0]+W[r,c,1,1]+1j*(W[r,c,0,1]-W[r,c,1,0]))
        storage[(period,nrows)][("C2")] = C2ij
        return C2ij
    return storage[(period,nrows)][("C2")]


# --- Fourier transformed coefficients of the spin-wave Hamiltonian ---
def Gamma1_matrix(q_vec, period=n,nrows=n_chains, Jmat=J_matrix):
    """ Coefficients Gamma1 of some boson-operators in the fourier transformed spin-wave SPFMI Hamiltonian.
        The FT of coefficients C1 with some additional constant. """
    if type(storage[(period,nrows)].get(("G1"))) != dict:
        storage[(period,nrows)][("G1")] = {}
    if type(storage[(period,nrows)][("G1")].get(q_vec.tobytes())) != np.ndarray:
        n_SLs = period*nrows
        G1 = np.zeros((n_SLs,n_SLs), dtype=np.complex_) # (n_SLs)x(n_SLs) matrix
        C1 = C1_matrix(period,nrows, Jmat)
        for L in range(n_SLs):
            for T_idx, T in enumerate(get_nearest_neighbour_indices(L,period,nrows)):
                G1[L,T] += C1[L,T]*np.exp(1j*q_vec@d_vec[T_idx])
        storage[(period,nrows)][("G1")][q_vec.tobytes()] = G1
        return G1
    return storage[(period,nrows)][("G1")][q_vec.tobytes()]


def Gamma2_matrix(q_vec, period=n,nrows=n_chains, Jmat=J_matrix):
    """ Coefficients Gamma2 of some boson-operators in the fourier transformed spin-wave SPFMI Hamiltonian.
        The FT of coefficients C2 with some additional constant. """
    if type(storage[(period,nrows)].get(("G2"))) != dict:
        storage[(period,nrows)][("G2")] = {}
    if type(storage[(period,nrows)][("G2")].get(q_vec.tobytes())) != np.ndarray:
        n_SLs = period*nrows
        G2 = np.zeros((n_SLs,n_SLs), dtype=np.complex_) # (n_SLs)x(n_SLs) matrix
        C2 = C2_matrix(period,nrows, Jmat)
        for L in range(n_SLs):
            for T_idx, T in enumerate(get_nearest_neighbour_indices(L,period,nrows)):
                G2[L,T] += C2[L,T]*np.exp(1j*q_vec@d_vec[T_idx])
        storage[(period,nrows)][("G2")][q_vec.tobytes()] = G2
        return G2
    return storage[(period,nrows)][("G2")][q_vec.tobytes()]


def zeta_matrix(period=n,nrows=n_chains, Jmat=J_matrix):
    """ Coefficients zeta of the remaining boson-operators in the fourier transformed spin-wave SPFMI Hamiltonian.
        The FT of coefficients W^{zz}_{ij} with some additional constant. """
    
    if type(storage[(period,nrows)].get("zeta")) != np.ndarray:
        n_SLs = period*nrows
        zeta = np.zeros((n_SLs,n_SLs), dtype=np.complex_) # (n_SLs)x(n_SLs) matrix
        W = W_rotatedvxvmat(period,nrows, Jmat,)
        for L in range(n_SLs):
            for T in get_nearest_neighbour_indices(L,period,nrows):
                zeta[L,T] += W[L,T,2,2]
        storage[(period,nrows)]["zeta"] = zeta
        return zeta
    return storage[(period,nrows)]["zeta"]


def nu_matrix(q_vec, period=n,nrows=n_chains, Jmat=J_matrix):
    if type(storage[(period,nrows)].get(("nu"))) != dict:
        storage[(period,nrows)][("nu")] = {}
    if type(storage[(period,nrows)][("nu")].get(q_vec.tobytes())) != np.ndarray:
        n_SLs = period*nrows
        nu = np.zeros((n_SLs,n_SLs), dtype=np.complex_) # (n_SLs)x(n_SLs) matrix
        G1 = Gamma1_matrix(q_vec, period,nrows, Jmat)
        for L in range(n_SLs):
            nu[L,L] = - 1/2*K*(np.sin(get_spin_from_lattice_idx(L,period,nrows)[0]))**2
            for T in np.unique(get_nearest_neighbour_indices(L,period,nrows)):
                nu[L,T] = G1[L,T] 
        storage[(period,nrows)][("nu")][q_vec.tobytes()] = nu
        return nu
    return storage[(period,nrows)][("nu")][q_vec.tobytes()]


def eta_matrix(q_vec, period=n,nrows=n_chains, Jmat=J_matrix):
    if type(storage[(period,nrows)].get(("eta"))) != dict:
        storage[(period,nrows)][("eta")] = {}
    if type(storage[(period,nrows)][("eta")].get(q_vec.tobytes())) != np.ndarray:
        n_SLs = period*nrows
        eta = np.zeros((n_SLs,n_SLs), dtype=np.complex_) # (n_SLs)x(n_SLs) matrix
        zeta = zeta_matrix(period,nrows, Jmat)
        G2 = Gamma2_matrix(q_vec, period,nrows, Jmat)
        for L in range(n_SLs):
            eta[L,L] = -1/2*K*(np.sin(get_spin_from_lattice_idx(L,period,nrows)[0]))**2
            for T in get_nearest_neighbour_indices(L,period,nrows): 
                eta[L,L] += zeta[L,T]
                eta[L,T] = G2[L,T]
        storage[(period,nrows)][("eta")][q_vec.tobytes()] = eta
        return eta
    return storage[(period,nrows)][("eta")][q_vec.tobytes()]


def H_matrix(q_vec, period=n,nrows=n_chains, Jmat=J_matrix): 
    """ Hamiltonian matrix.
        block matrix with eta(-q), nu(-q), and their complex conjugates of q """
    
    if type(storage[(period,nrows)].get(("H"))) != dict:
        storage[(period,nrows)][("H")] = {}
    if type(storage[(period,nrows)][("H")].get(q_vec.tobytes())) != np.ndarray:
        storage[(period,nrows)][("H")][q_vec.tobytes()] = np.block([ \
        [
            np.conjugate(eta_matrix(q_vec, period,nrows, Jmat)), 
            np.conjugate(nu_matrix(q_vec, period,nrows, Jmat))
        ], [
            nu_matrix(-q_vec, period,nrows, Jmat), 
            eta_matrix(-q_vec, period,nrows, Jmat)
        ]])
    return storage[(period,nrows)][("H")][q_vec.tobytes()]


# --- para unit matrix (PUM). Want to diagonalise M = PUM@H_matrix. ---
def create_para_unit_matrix(sub_dim=n*n_chains):
    if type(storage.get(sub_dim)) != dict:
        storage[sub_dim] = {}
    if type(storage[sub_dim].get("pum")) != np.ndarray:
        storage[sub_dim]["pum"] = np.block([[np.eye(sub_dim), np.eye(sub_dim)*0], [np.eye(sub_dim)*0, np.eye(sub_dim)*(-1)]])
    return storage[sub_dim].get("pum")

# %%
""" Sorting bands: prelude """

def sort_positive_asc_negative_desc(ws, vs):
    ind = np.argsort(ws)
    ws_ascending = ws[ind]
    vs_ascending = vs[:, ind]

    pos_idcs = (np.argwhere(ws_ascending>0)).flatten()
    neg_idcs = np.flip((np.argwhere(ws_ascending<0)).flatten())
    
    return ws_ascending[(list(pos_idcs) + list(neg_idcs))], \
           vs_ascending[:, (list(pos_idcs) + list(neg_idcs))]


## ----- Gram-Schmidt ----- ##
def proj(u,v, pum):
    assert (np.abs(u.T.conj()@pum@u) >1e-10), \
           f"OBS: Denominator is null...: {u.T.conj()@pum@u}\n\
            with u: {u}"
    return u*(u.T.conj()@pum@v)/(u.T.conj()@pum@u)

def do_GramSchmidt(V,pum):
    """ Returns the orthonormalised basis set {u_k} from the given basis set {v_k}.
        v_k is a column of input matrix V. 
        In my case: v_k an eigenvalue from np.linalg.eig(). """
    U = np.zeros_like(V)
    U[:,0] = V[:,0]
    for k in range(1,V.shape[1]):
        # print(f"k: {k}")
        U[:,k] = V[:,k] - proj(U[:,0], V[:,k],pum)   ## first step (i=0)
        for i in range(1,k):
            U[:,k] = U[:,k] - proj(U[:,i], U[:,k],pum)
        U[:,k] = U[:,k]/np.linalg.norm(U[:,k])
    assert ( U-V != 0).any(), "GS evecs and pre-GS evecs are equal"
    return U

def get_GS_evecs(ws, vs, pum):
    vs_GS = deepcopy(vs)
    for gamma in range(ws.shape[-1]):
        om = ws[gamma]
        if round(om, 10) in np.around(ws[gamma+1:], 10):
            ind = np.argwhere(np.around(ws,10)==round(om,10)).flatten()
            vs_GS[:,ind] = do_GramSchmidt(vs[:,ind], pum)
    return vs_GS



## ----- Diagonalising K@PUM@K.T.conj() ----- ##
def get_omegas_Tinv_q(q_vec, period=n,nrows=n_chains, Jmat=J_matrix):
    """ Returns sorted evals and corresponding evecs: positive ascending, negative descending """
    
    if type(storage[(period,nrows)].get(("omegas"))) != dict:   ## if we don't have evals, we don't have evecs
        storage[(period,nrows)][("omegas")] = {}
        storage[(period,nrows)][("Tinv")] = {}
    
    if type(storage[(period,nrows)][("omegas")].get(q_vec.tobytes())) != np.ndarray:
        H = H_matrix(q_vec, period,nrows, Jmat)
        Kmat = np.linalg.cholesky(H).T.conj()
        PUM = create_para_unit_matrix(period*nrows)
        ws, U = np.linalg.eig(Kmat @ PUM @ (Kmat.T.conj()))
        ws = ws.real

        ws, U = sort_positive_asc_negative_desc(ws,U)
        U_GS = get_GS_evecs(ws, U, PUM)
        omegas = 2*ws@PUM
        Tinv = np.linalg.inv(Kmat) @ U_GS @ (np.diag(1/2*omegas)**(1/2))

        storage[(period,nrows)][("omegas")][q_vec.tobytes()],storage[(period,nrows)][("Tinv")][q_vec.tobytes()] = omegas, Tinv
        return omegas, Tinv
    return storage[(period,nrows)][("omegas")].get(q_vec.tobytes()),storage[(period,nrows)][("Tinv")].get(q_vec.tobytes())

# %%
""" Sorting bands: Actual sorting """

def sort_bands_wrt_neighbouring_q_point(e, psi, psiprev):
    Q = np.abs(psiprev.T.conj()@psi)**2
    assignment = sc.optimize.linear_sum_assignment(-Q)[1]
    return e[assignment], psi[:,assignment]


def get_Tinv_correct_form(q_vectors, Tinv_original, m_):
    Tinv_mani = deepcopy(Tinv_original)
    for qi_,q_ in enumerate(q_vectors):
        if (-q_[0] in q_vectors[qi_+1:, 0] and -q_[1] in q_vectors[qi_+1:,1]):
            idcs = np.argwhere( (q_vectors[qi_:, 0] == -q_[0]) & (q_vectors[qi_:, 1] == -q_[1]) )+qi_
            for idx in idcs:
                Tinv_mani[qi_, :m_, :m_] = Tinv_original[idx, m_:, m_:].conj()
                Tinv_mani[qi_, :m_, m_:] = Tinv_original[idx, m_:, :m_].conj()
                Tinv_mani[qi_, m_:, :m_] = Tinv_original[idx, :m_, m_:].conj()
                Tinv_mani[qi_, m_:, m_:] = Tinv_original[idx, :m_, :m_].conj()
    return Tinv_mani


def get_2m_bands(q_vectors, subfolder_, filenames_, disable_tqdm_qs=False, period=n,nrows=n_chains, Jmat=J_matrix, k_sign_=+1, kp_sign_=+1, recalculate=False):
    try:
        all_Mevals = np.load(f"arrays\{subfolder_}\{filenames_[0]} ({sts.get(k_sign_)}k)({sts.get(kp_sign_)}k').npy")
        all_Mevecs = np.load(f"arrays\{subfolder_}\{filenames_[1]} ({sts.get(k_sign_)}k)({sts.get(kp_sign_)}k').npy")
        if recalculate: raise Exception
    except:
        m = period*nrows
        noq = q_vectors.shape[0]
        sts = {+1 : '+', -1 : '-'}

        all_Mevals = np.zeros((noq, 2*m))
        all_Mevecs = np.zeros((noq, 2*m, 2*m), dtype=np.complex_)

        for qi_ in tqdm.tqdm(range(noq), disable=disable_tqdm_qs):
                e, psi = get_omegas_Tinv_q(q_vectors[qi_,:], period,nrows, Jmat)
                all_Mevals[qi_,:] = e
                all_Mevecs[qi_,:,:] = psi
        
        all_Mevecs = get_Tinv_correct_form(q_vectors, all_Mevecs, period*nrows)
        
        if type(filenames_) == list and type(subfolder_) == str:
            np.save(f"arrays\{subfolder_}\{filenames_[0]} ({sts.get(k_sign_)}k)({sts.get(kp_sign_)}k').npy", all_Mevals)
            np.save(f"arrays\{subfolder_}\{filenames_[1]} ({sts.get(k_sign_)}k)({sts.get(kp_sign_)}k').npy", all_Mevecs)

    return all_Mevals, all_Mevecs




