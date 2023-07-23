# %%
from system_folder.system_functions import *
from system_folder.qvalue_arrays import *
from system_folder.DoS_gap_functions import *


def g1Lns(L,Qnu,s1, period=n,nrows=n_chains, N=N_, S=S_, Jbar_L=J_L):
    N_L = N/(period*nrows)
    thetaL, psiL = get_spin_from_lattice_idx(L,period, nrows)
    return -np.sqrt(2*S*N_L)/N *Jbar_L*np.cos(psiL)*( np.cos(thetaL) - s1 ) *np.exp(1j*Qnu@get_rL_from_lattice_idx(L,period, nrows))

def g2Lns(L,Qnu,s1, period=n,nrows=n_chains, N=N_, S=S_, Jbar_L=J_L):
    N_L = N/(period*nrows)
    thetaL, _ = get_spin_from_lattice_idx(L, period,nrows)
    return np.sqrt(2*S*N_L)/N *Jbar_L*s1*np.sin(thetaL) *np.exp(1j*Qnu@get_rL_from_lattice_idx(L,period, nrows))

def gLnsts(L,Qnu,s1,s2, period=n,nrows=n_chains, N=N_, S=S_, Jbar_L=J_L):
    if s2 == -s1:
        return g1Lns(L,Qnu,s1, period,nrows,N,S, Jbar_L)
    return g2Lns(L,Qnu,s1, period,nrows,N,S, Jbar_L)


def Aq_elements(s1,s2,s3,s4, gamma,Tinv_m_,  Qnu_array_,  period=n, nrows=n_chains, disable_tqdm=False):
    A = np.zeros(Tinv_m_.shape[0], dtype=np.complex_)
    m = Tinv_m_.shape[-1]

    for L in range(m):
        for Lp in range(m):
            A +=  gLnsts(L,Qnu_array_,s1,s2, period,nrows)*gLnsts(Lp,-Qnu_array_,s3,s4, period,nrows)\
                    * Tinv_m_[:,L,gamma%m]* np.conjugate(Tinv_m_[:,Lp+m,gamma%m])\
                  + gLnsts(L,Qnu_array_,s1,s2, period,nrows)*np.conjugate(gLnsts(Lp,Qnu_array_,s4,s3, period,nrows))\
                    * Tinv_m_[:,L,gamma%m]* np.conjugate(Tinv_m_[:,Lp,gamma%m])\
                  + np.conjugate(gLnsts(L,-Qnu_array_,s2,s1, period,nrows))*gLnsts(Lp,-Qnu_array_,s3,s4, period,nrows)\
                    * Tinv_m_[:,L+m,gamma%m]* np.conjugate(Tinv_m_[:,Lp+m,gamma%m])\
                  + np.conjugate(gLnsts(L,-Qnu_array_,s2,s1, period,nrows))*np.conjugate(gLnsts(Lp,Qnu_array_,s4,s3, period,nrows))\
                    * Tinv_m_[:,L+m,gamma%m]* np.conjugate(Tinv_m_[:,Lp,gamma%m])
    return A


## ----- curly-A ----- ##
def curlyAq_elements(s1,s2,s3,s4, gamma,Tinv_m_p_, Tinv_m_n_,  Qnu_array_,  period=n, nrows=n_chains, disable_tqdm=False):
    return - Aq_elements(s1,s4,s2,s3, gamma,Tinv_m_p_,  Qnu_array_,  period, nrows, disable_tqdm) \
           - Aq_elements(s2,s3,s1,s4, gamma,Tinv_m_n_, -Qnu_array_,  period, nrows, disable_tqdm)


# %%
sts = {+1 : '+', -1 : '-'}   ### sign to symbol dictionary
sta = { -1: u"\u2193", 1 : u"\u2191" }   ## spin_to_arrows dict
spins = [-1,1]


# %%
""" For Nu matrix """

def get_Veff_fullMatrix(s1,s2,s3,s4, th_array_, k_s, kp_s, eta_, no_bar, no_, period, nrows):
    """ Veff varying with theta' and theta """
    global sta, sts

    Nth = len(th_array_)
    m = period*nrows
    subfolder_ = f"FS with  eta={eta_}"
    path_ = f"arrays/{subfolder_}/Veff_matrices"
    if not os.path.isdir(path_):
        Path(path_).mkdir(parents=True, exist_ok=True)
    tmp_Veff = np.zeros((Nth, Nth), dtype=np.complex_)
    for th0_idx in range(Nth):
        print(f"Working on th0_idx {th0_idx} of {Nth} for calculation {no_} of 4 for element {no_bar} of 16...", sep='', end='\r')
        q_FS, Qnu_array = get_q_Qnu_FS(None, None, th_array_,\
                                    th0i=th0_idx, eta=eta_, disable_tqdm=True, k_sign_=k_s, kp_sign_=kp_s, recalculate=False)
        omegas_, Tinv_p_ = get_2m_bands(q_FS, None, None, disable_tqdm_qs=True, k_sign_=k_s, kp_sign_=kp_s, recalculate=False)
        Tinv_n_ = np.zeros_like(Tinv_p_)
        Tinv_n_[:, :m,:m] = Tinv_p_[:, m:,m:].conj()
        # Tinv_n_[:, :m,m:] = Tinv_p_[:, m:,:m].conj()
        Tinv_n_[:, m:,:m] = Tinv_p_[:, :m,m:].conj()
        # Tinv_n_[:, m:,m:] = Tinv_p_[:, :m,:m].conj()
        for gamma in range(m):
            tmp_Veff[:,th0_idx] += -1/(2*omegas_[:,gamma])*curlyAq_elements(s1,s2,s3,s4,gamma,Tinv_p_[:,:,:m],Tinv_n_[:,:,:m],  Qnu_array,  period=period,nrows=nrows, disable_tqdm=True)
    np.save(f"{path_}/({sts.get(k_s)}k)({sts.get(kp_s)}k')   {sta.get(s1)}{sta.get(s2)}{sta.get(s3)}{sta.get(s4)}.npy", tmp_Veff)
        
    return tmp_Veff


def get_Veffbar_fullMatrix(s1,s2,s3,s4,  th_array_, eta_, no, recalculate, period, nrows):
    global sta

    path_ = f"arrays/FS with  eta={eta_}/Veffbar_matrices"
    if not os.path.isdir(path_): 
        Path(path_).mkdir(parents=True, exist_ok=True)
    try:
        tmp_Veffbar_= np.load(f"{path_}/{sta.get(s1)}{sta.get(s2)}{sta.get(s3)}{sta.get(s4)}.npy")
        if recalculate: raise Exception
    except:
        tmp_Veffbar_ = 1/2* (  get_Veff_fullMatrix(s1,s2,s3,s4, th_array_, k_s=+1, kp_s=+1, eta_=eta_, no_bar=no, no_=1, period=period, nrows=nrows) \
                             + get_Veff_fullMatrix(s2,s1,s4,s3, th_array_, k_s=-1, kp_s=-1, eta_=eta_, no_bar=no, no_=2, period=period, nrows=nrows) \
                             - get_Veff_fullMatrix(s1,s2,s4,s3, th_array_, k_s=-1, kp_s=+1, eta_=eta_, no_bar=no, no_=3, period=period, nrows=nrows) \
                             - get_Veff_fullMatrix(s2,s1,s3,s4, th_array_, k_s=+1, kp_s=-1, eta_=eta_, no_bar=no, no_=4, period=period, nrows=nrows))
        np.save(f"{path_}/{sta.get(s1)}{sta.get(s2)}{sta.get(s3)}{sta.get(s4)}.npy", tmp_Veffbar_)
    return tmp_Veffbar_


def get_Nu_fullMatrix(eta_, th_array_, recalculate=False, period=n, nrows=n_chains):
    Nth = len(th_array_)
    Nu_matrix = np.zeros((4, 4, Nth, Nth), dtype=np.complex_)
    Nu_matrix[0,0, :,:] = 1/2*(  get_Veffbar_fullMatrix( 1,-1,-1, 1, th_array_, eta_, no=1, recalculate=recalculate, period=period, nrows=nrows)\
                                -get_Veffbar_fullMatrix( 1,-1, 1,-1, th_array_, eta_, no=2, recalculate=recalculate, period=period, nrows=nrows)\
                                -get_Veffbar_fullMatrix(-1, 1,-1, 1, th_array_, eta_, no=3, recalculate=recalculate, period=period, nrows=nrows)\
                                +get_Veffbar_fullMatrix(-1, 1, 1,-1, th_array_, eta_, no=4, recalculate=recalculate, period=period, nrows=nrows))
    Nu_matrix[0,1, :,:] = 1/2*(  get_Veffbar_fullMatrix( 1,-1, 1, 1, th_array_, eta_, no=5, recalculate=recalculate, period=period, nrows=nrows)\
                                -get_Veffbar_fullMatrix(-1, 1, 1, 1, th_array_, eta_, no=6, recalculate=recalculate, period=period, nrows=nrows))
    Nu_matrix[0,2, :,:] = 1/2*(  get_Veffbar_fullMatrix( 1,-1,-1,-1, th_array_, eta_, no=7, recalculate=recalculate, period=period, nrows=nrows)\
                                -get_Veffbar_fullMatrix(-1, 1,-1,-1, th_array_, eta_, no=8, recalculate=recalculate, period=period, nrows=nrows))
    Nu_matrix[0,3, :,:] = 1/2*(  get_Veffbar_fullMatrix( 1,-1,-1, 1, th_array_, eta_, no=1, recalculate=False, period=period, nrows=nrows)\
                                +get_Veffbar_fullMatrix( 1,-1, 1,-1, th_array_, eta_, no=2, recalculate=False, period=period, nrows=nrows)\
                                -get_Veffbar_fullMatrix(-1, 1,-1, 1, th_array_, eta_, no=3, recalculate=False, period=period, nrows=nrows)\
                                -get_Veffbar_fullMatrix(-1, 1, 1,-1, th_array_, eta_, no=4, recalculate=False, period=period, nrows=nrows))
    
    Nu_matrix[1,0, :,:] =   get_Veffbar_fullMatrix(1, 1,-1, 1, th_array_, eta_, no=9, recalculate=recalculate, period=period, nrows=nrows) \
                          - get_Veffbar_fullMatrix(1, 1, 1,-1, th_array_, eta_, no=10, recalculate=recalculate, period=period, nrows=nrows)
    Nu_matrix[1,1, :,:] = get_Veffbar_fullMatrix(1, 1, 1, 1, th_array_, eta_, no=11, recalculate=recalculate, period=period, nrows=nrows)
    Nu_matrix[1,2, :,:] = get_Veffbar_fullMatrix(1, 1,-1,-1, th_array_, eta_, no=12, recalculate=recalculate, period=period, nrows=nrows)
    Nu_matrix[1,3, :,:] =   get_Veffbar_fullMatrix(1, 1,-1, 1, th_array_, eta_, no=9, recalculate=False, period=period, nrows=nrows) \
                          + get_Veffbar_fullMatrix(1, 1, 1,-1, th_array_, eta_, no=10, recalculate=False, period=period, nrows=nrows)

    Nu_matrix[2,0, :,:] =   get_Veffbar_fullMatrix(-1,-1,-1, 1, th_array_, eta_, no=13, recalculate=recalculate, period=period, nrows=nrows) \
                          - get_Veffbar_fullMatrix(-1,-1, 1,-1, th_array_, eta_, no=14, recalculate=recalculate, period=period, nrows=nrows)
    Nu_matrix[2,1, :,:] = get_Veffbar_fullMatrix(-1,-1, 1, 1, th_array_, eta_, no=15, recalculate=recalculate, period=period, nrows=nrows)
    Nu_matrix[2,2, :,:] = get_Veffbar_fullMatrix(-1,-1,-1,-1, th_array_, eta_, no=16, recalculate=recalculate, period=period, nrows=nrows)
    Nu_matrix[2,3, :,:] =   get_Veffbar_fullMatrix(-1,-1,-1, 1, th_array_, eta_, no=13, recalculate=False, period=period, nrows=nrows) \
                          + get_Veffbar_fullMatrix(-1,-1, 1,-1, th_array_, eta_, no=14, recalculate=False, period=period, nrows=nrows)

    Nu_matrix[3,0, :,:] = 1/2*(  get_Veffbar_fullMatrix( 1,-1,-1, 1, th_array_, eta_, no=1, recalculate=False, period=period, nrows=nrows) \
                                -get_Veffbar_fullMatrix( 1,-1, 1,-1, th_array_, eta_, no=2, recalculate=False, period=period, nrows=nrows) \
                                +get_Veffbar_fullMatrix(-1, 1,-1, 1, th_array_, eta_, no=3, recalculate=False, period=period, nrows=nrows) \
                                -get_Veffbar_fullMatrix(-1, 1, 1,-1, th_array_, eta_, no=4, recalculate=False, period=period, nrows=nrows) )
    Nu_matrix[3,1, :,:] = 1/2*(  get_Veffbar_fullMatrix( 1,-1, 1, 1, th_array_, eta_, no=5, recalculate=False, period=period, nrows=nrows) \
                                +get_Veffbar_fullMatrix(-1, 1, 1, 1, th_array_, eta_, no=6, recalculate=False, period=period, nrows=nrows))
    Nu_matrix[3,2, :,:] = 1/2*(  get_Veffbar_fullMatrix( 1,-1,-1,-1, th_array_, eta_, no=7, recalculate=False, period=period, nrows=nrows) \
                                +get_Veffbar_fullMatrix(-1, 1,-1,-1, th_array_, eta_, no=8, recalculate=False, period=period, nrows=nrows))
    Nu_matrix[3,3, :,:] = 1/2*(  get_Veffbar_fullMatrix( 1,-1,-1, 1, th_array_, eta_, no=1, recalculate=False, period=period, nrows=nrows) \
                                +get_Veffbar_fullMatrix( 1,-1, 1,-1, th_array_, eta_, no=2, recalculate=False, period=period, nrows=nrows) \
                                +get_Veffbar_fullMatrix(-1, 1,-1, 1, th_array_, eta_, no=3, recalculate=False, period=period, nrows=nrows) \
                                +get_Veffbar_fullMatrix(-1, 1, 1,-1, th_array_, eta_, no=4, recalculate=False, period=period, nrows=nrows) )

    return Nu_matrix
