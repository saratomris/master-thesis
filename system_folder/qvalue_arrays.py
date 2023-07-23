# %%
from system_folder.system_functions import *


# %%
sts = {+1 : '+', -1 : '-'}   ### sign to symbol dictionary

# %%
""" Creating array with k-values through certain high-symmetry points """

# --- Through G-M-K-G ---
def get_k_GKMG(subfolder_, k_GKMG_filename_, nok=70, b1_e=b1_electron, b2_e=b2_electron):
    Kpt = np.array([np.linalg.norm(b1_e)/np.sqrt(3), 0])
    Mpt = np.linalg.norm(b1_e)/2*np.array([np.cos(np.pi/6), np.sin(np.pi/6)])
    kx = np.hstack((np.linspace(0, Kpt[0], nok),
                    np.linspace(Kpt[0], Mpt[0], nok+1)[1:],
                    np.linspace(Mpt[0], 0, nok+1)[1:],
                   )).flatten().reshape(((3*nok), 1))
    
    ky = np.hstack((np.linspace(0, Kpt[1], nok),
                    np.linspace(Kpt[1], Mpt[1], nok+1)[1:],
                    np.linspace(Mpt[1], 0, nok+1)[1:],
                   )).flatten().reshape(((3*nok), 1))

    tmp = np.hstack((kx,ky))
    if type(subfolder_) == str and type(k_GKMG_filename_) == str:
        np.save(f"arrays/{subfolder_}/{k_GKMG_filename_}.npy", tmp)
    return tmp


def get_symmetry_pts_GMKG(subfolder_, Qs_GMKG_pts_filename_, Qs_GKMG, k_GKMG_, b1_e=b1_electron, b2_e=b2_electron):
    """ no_GX_ = no_SY and no_XS_ = no_YG """
    Kpt = np.array([np.linalg.norm(b1_e)/np.sqrt(3),0])
    Mpt = np.linalg.norm(b1_e)/2*np.array([np.cos(np.pi/6), np.sin(np.pi/6)])

    Mpt_i = np.argwhere((k_GKMG_[:,0] == Mpt[0]) & (k_GKMG_[:,1] == Mpt[1])).flatten()[0]
    Kpt_i = np.argwhere((k_GKMG_[:,0] == Kpt[0]) & (k_GKMG_[:,1] == Kpt[1])).flatten()[0]
    tmp = [Qs_GKMG[0], Qs_GKMG[Mpt_i], Qs_GKMG[Kpt_i], Qs_GKMG[-1]]
    if type(subfolder_) == str and type(Qs_GMKG_pts_filename_) == str:
        np.save(f"arrays/{subfolder_}/{Qs_GMKG_pts_filename_}.npy", tmp)
    return tmp

# %%
""" Creating array with q-values through certain high-symmetry points """

# --- Through G-X-S-Y-G --- 
def get_q_GXSYG(subfolder_, q_GXSYG_filename_, nox=100, noy=300, bz_x=BZ_x, bz_y=BZ_y):

    qx = np.hstack((np.linspace(0, bz_x, nox+1),
                    np.full(noy, bz_x),
                    np.linspace(bz_x, 0, nox+1)[1:],
                    np.full(noy, 0))).flatten().reshape(((2*nox+2*noy+1), 1))
    qy = np.hstack((np.full(nox+1, 0),
                    np.linspace(0, bz_y, noy+1)[1:],
                    np.full(nox+1, bz_y)[1:],
                    np.linspace(bz_y, 0, noy+1)[1:])).flatten().reshape(((2*nox+2*noy+1), 1))

    tmp = np.hstack((qx,qy))
    if type(subfolder_) == str and type(q_GXSYG_filename_) == str:
        np.save(f"arrays/{subfolder_}/{q_GXSYG_filename_}.npy", tmp)
    return tmp


## ----- For FS ----- ##
def eDR(k_x, k_y, eta, a=1):
    return -2*(sp.cos(k_x*a) + 2*sp.cos(1/2*k_x*a)*sp.cos(sp.sqrt(3)/2*k_y*a)) -eta
def eDR_np(k_x, k_y, eta, a=1):
    return -2*(np.cos(k_x*a) + 2*np.cos(1/2*k_x*a)*np.cos(np.sqrt(3)/2*k_y*a)) -eta

def solve_kF(theta, eta, a=1):   ## eta = mu/t
    ## finn anslag ved kx=0
    est = 2/(np.sqrt(3)*a)*np.arccos(-1/2*(eta/2+1))
    kF = sp.symbols('k_F')
    func_np = sp.lambdify(kF, eDR(kF*sp.cos(theta), kF*sp.sin(theta), eta, a), modules=['numpy'])
    return sc.optimize.fsolve(func_np, est)[0]

def get_kx_ky(theta_, eta_=-5.9, a=1):
    kF = solve_kF(theta_, eta_, a)
    return np.array([kF*np.cos(theta_), kF*np.sin(theta_)])



def get_Qnu_k_kp(k_, kp_, period=n, nrows=n_chains):
    """ Want Q_nu closest to kp - k"""
    Qnu_set = get_Qnu_set(period, nrows)
    tmp = Qnu_set - (kp_ - k_)
    nu = np.argmin(np.abs(tmp[:,0]**2 + tmp[:,1]**2))
    return Qnu_set[nu,:]


def get_q_Qnu_FS(subfolder_, filenames_q_Qnu_kp_k_, th_array_, th0i, eta=-5.9, disable_tqdm=False, k_sign_=+1, kp_sign_=+1, recalculate=False):
    path_ = f"arrays/{subfolder_}"
    if ((not os.path.isdir(path_)) and (subfolder_ != None)):
        Path(path_).mkdir(parents=True, exist_ok=True)
    try:
        q_array_   = np.load(f"arrays/{subfolder_}/{filenames_q_Qnu_kp_k_[0]} ({sts.get(k_sign_)}k)({sts.get(kp_sign_)}k').npy")  ## q
        Qnu_array_ = np.load(f"arrays/{subfolder_}/{filenames_q_Qnu_kp_k_[1]} ({sts.get(k_sign_)}k)({sts.get(kp_sign_)}k').npy")  ## Qnu
        np.load(f"arrays/{subfolder_}/{filenames_q_Qnu_kp_k_[2]} ({sts.get(kp_sign_)}k').npy")                                    ## kp_
        np.load(f"arrays/{subfolder_}/{filenames_q_Qnu_kp_k_[3]} ({sts.get(k_sign_)}k).npy")                                      ## k_
        if recalculate: raise Exception
    except:
        q_array_  = np.zeros((len(th_array_), 2))
        kp_array_ = np.zeros((len(th_array_), 2))
        Qnu_array_= np.zeros((len(th_array_), 2))
        k_ = k_sign_*get_kx_ky(th_array_[th0i], eta_=eta)
        for th_idx, th_ in enumerate(th_array_):
            kp_array_[th_idx] = kp_sign_*get_kx_ky(th_, eta_=eta)
            Qnu_array_[th_idx] = get_Qnu_k_kp(k_, kp_array_[th_idx])
            q_array_[th_idx, :] = kp_array_[th_idx] - k_ - Qnu_array_[th_idx]
        if type(subfolder_) == str and type(filenames_q_Qnu_kp_k_) == list:
            np.save(f"arrays/{subfolder_}/{filenames_q_Qnu_kp_k_[0]} ({sts.get(k_sign_)}k)({sts.get(kp_sign_)}k').npy", q_array_)
            np.save(f"arrays/{subfolder_}/{filenames_q_Qnu_kp_k_[1]} ({sts.get(k_sign_)}k)({sts.get(kp_sign_)}k').npy", Qnu_array_)
            np.save(f"arrays/{subfolder_}/{filenames_q_Qnu_kp_k_[2]} ({sts.get(kp_sign_)}k').npy", kp_array_)
            np.save(f"arrays/{subfolder_}/{filenames_q_Qnu_kp_k_[3]} ({sts.get(k_sign_)}k).npy", k_)
    return q_array_, Qnu_array_


# %%
### q_vectors_fullBZ and corresponding sorting

def get_q_vectors_fullBZ(filename=None, bz_x=BZ_x, bz_y=BZ_y, nox=150, noy=150, recalculate=False):
    try:
        qtmp = np.load(f"arrays/{filename}.npy")
        if recalculate: raise Exception
    except:
        qx = np.hstack((np.linspace(-bz_x, 0, nox+1)[1:], np.linspace(0, bz_x, nox+1)[1:]))
        qy = np.hstack((np.linspace(-bz_y, 0, noy+1)[1:], np.linspace(0, bz_y, noy+1)[1:]))
        qtmp = np.array([[[qxi, qyi] for qyi in qy] for qxi in qx])
        if type(filename)==str:
            np.save(f"arrays/{filename}.npy", qtmp)
    return qtmp


def sort_Mevals_Mevecs_initial(q0, PUM, period, nrows, Jmat):
    omegas, Mevecs = get_omegas_Tinv_q(q0, period,nrows, Jmat)
    Mevals = 0.5*omegas@PUM
    ind = np.argsort(Mevals)

    return omegas[ind], Mevecs[:, ind]


def get_2m_bands_fullBZ(q_vectors_fullBZ, filenames=None, disable_tqdm_qs=False, period=n,nrows=n_chains, Jmat=J_matrix, recalculate=False):
    try:
        omegas = np.load(f"arrays\{filenames[0]}.npy")
        Tinv   = np.load(f"arrays\{filenames[1]}.npy")
        if recalculate: raise Exception
    except:
        m = period*nrows
        PUM = create_para_unit_matrix(m)
        nox = q_vectors_fullBZ.shape[0]
        noy = q_vectors_fullBZ.shape[1]

        omegas = np.zeros((nox, noy, 2*m))
        Tinv = np.zeros((nox, noy, 2*m, 2*m), dtype=np.complex_)

        for qxi in tqdm.auto.tqdm(range(nox), position=0, leave=True, disable=disable_tqdm_qs):
            for qyi in range(noy):
                e, psi = sort_Mevals_Mevecs_initial(q_vectors_fullBZ[qxi,qyi,:], PUM, \
                                                period, nrows, Jmat)
                omegas[qxi,qyi] = e
                Tinv[qxi,qyi]   = psi
        if type(filenames) == list:
            np.save(f"arrays\{filenames[0]}.npy", omegas)
            np.save(f"arrays\{filenames[1]}.npy", Tinv)
    return omegas, Tinv


def sort_bands_fullBZ(omegas_, Tinv_, filenames=None, disable_tqdm_qs=False, recalculate=False):
    try:
        sorted_Mevals = np.load(f"arrays\{filenames[0]}.npy")
        sorted_Mevecs = np.load(f"arrays\{filenames[1]}.npy")
        if recalculate: raise Exception
    except:
        nox = omegas_.shape[0]
        noy = omegas_.shape[1]

        sorted_Mevals = deepcopy(omegas_)
        sorted_Mevecs = deepcopy(Tinv_)

        for qxi in tqdm.auto.tqdm(range(nox), disable=disable_tqdm_qs):
            for qyi in range(noy):
                e = sorted_Mevals[qxi,qyi,:]
                psi = sorted_Mevecs[qxi,qyi,:,:]
                if (qyi == 0 and qxi == 0):
                    e_sorted = e
                    psi_sorted = psi
                if (qyi == 0 and qxi > 0):
                    qnxi, qnyi = [qxi-1, qyi]
                    eprev = sorted_Mevals[qnxi, qnyi, :]
                    psiprev = sorted_Mevecs[qnxi, qnyi, :,:]
                    e_sorted,psi_sorted = sort_bands_wrt_neighbouring_q_point(e, psi, psiprev)
                elif (qyi > 0 and qxi == 0):
                    qnxi, qnyi = [qxi, qyi-1]
                    eprev = sorted_Mevals[qnxi, qnyi, :]
                    psiprev = sorted_Mevecs[qnxi, qnyi, :,:]
                    e_sorted,psi_sorted = sort_bands_wrt_neighbouring_q_point(e, psi, psiprev)
                else:
                    [qn1xi, qn1yi], [qn2xi, qn2yi] = [[qxi, qyi-1], [qxi-1, qyi]]
                    eprev1 = sorted_Mevals[qn1xi, qn1yi, :]
                    psiprev1 = sorted_Mevecs[qn1xi, qn1yi, :,:]
                    e1,psi1 = sort_bands_wrt_neighbouring_q_point(e, psi, psiprev1)
                    eprev2 = sorted_Mevals[qn2xi, qn2yi, :]
                    psiprev2 = sorted_Mevecs[qn2xi, qn2yi, :,:]
                    e_sorted,psi_sorted = sort_bands_wrt_neighbouring_q_point(e1, psi1, psiprev2)
                sorted_Mevals[qxi,qyi] = e_sorted
                sorted_Mevecs[qxi,qyi] = psi_sorted
        if type(filenames) == list:
            np.save(f"arrays\{filenames[0]}.npy", sorted_Mevals)
            np.save(f"arrays\{filenames[1]}.npy", sorted_Mevecs)
            
    return sorted_Mevals, sorted_Mevecs


def get_GXSYG_indices(q_vectors_fullBZ, bz_x=BZ_x, bz_y=BZ_y):
    ind_GX = np.argwhere((q_vectors_fullBZ[:,:,0]>=0)      & (q_vectors_fullBZ[:,:,1]==0))[:-1]
    ind_XS = np.argwhere((q_vectors_fullBZ[:,:,0] == bz_x) & (q_vectors_fullBZ[:,:,1]>=0))[:-1]
    ind_SY = np.flip(np.argwhere((q_vectors_fullBZ[:,:,0]>=0) & (q_vectors_fullBZ[:,:,1]==bz_y)), axis=0)[:-1]
    ind_YG = np.flip(np.argwhere((q_vectors_fullBZ[:,:,0]==0) & (q_vectors_fullBZ[:,:,1]>=0)), axis=0)#[:-1]
    
    ind_GXSYG = np.array([], dtype=int)
    ind_GXSYG = np.hstack((ind_GXSYG, ind_GX.flatten(), ind_XS.flatten(), ind_SY.flatten(), ind_YG.flatten()))
    ind_GXSYG = ind_GXSYG.reshape((len(ind_GXSYG)//2, 2))

    return ind_GXSYG


def get_array_GXSYG(array, ind_GXSYG_, filename=None, recalculate=False):
    try:
        lst_GXSYG = np.load(f"arrays/{filename}.npy")
        if recalculate: raise Exception
    except:
        lst_GXSYG  = np.empty((ind_GXSYG_.shape[0], array.shape[-1]), dtype=object)
        for i in range(ind_GXSYG_.shape[0]):
            lst_GXSYG[i] = array[tuple(ind_GXSYG_[i,:])]
        if type(filename)==str:
            np.save(f"arrays/{filename}.npy", lst_GXSYG)
    return lst_GXSYG

# %%

def get_Qs(q_vectors_, Qs_filename_, recalculate=False):
    try:
        Qs = np.load(f"arrays/{Qs_filename_}.npy")
        if recalculate: raise Exception
    except:
        Qs = [0]
        for i in range(1, q_vectors_.shape[0]):
            Qs.append(Qs[-1] + np.linalg.norm(q_vectors_[i]-q_vectors_[i-1]))
        if type(Qs_filename_) == str:
            np.save(f"arrays/{Qs_filename_}.npy", Qs)
    return Qs


def get_symmetry_pts_GXSYG(Qs_GXSYG_pts_filename_, Qs_GXSYG, q_GXSYG_, bz_x=BZ_x, bz_y=BZ_y, recalculate=False):
    """ no_GX_ = no_SY and no_XS_ = no_YG """
    try:
        tmp = np.load(f"arrays/{Qs_GXSYG_pts_filename_}.npy")
        if recalculate: raise Exception
    except:
        Xpt = np.argwhere((q_GXSYG_[:,0] == bz_x) & (q_GXSYG_[:,1] == 0)).flatten()[0]
        Spt = np.argwhere((q_GXSYG_[:,0] == bz_x) & (q_GXSYG_[:,1] == bz_y)).flatten()[0]
        Ypt = np.argwhere((q_GXSYG_[:,0] == 0) & (q_GXSYG_[:,1] == bz_y)).flatten()[0]
        tmp = [Qs_GXSYG[0], Qs_GXSYG[Xpt], Qs_GXSYG[Spt], Qs_GXSYG[Ypt], Qs_GXSYG[-1]]
        if type(Qs_GXSYG_pts_filename_) == str:
            np.save(f"arrays/{Qs_GXSYG_pts_filename_}.npy", tmp)
    return tmp

