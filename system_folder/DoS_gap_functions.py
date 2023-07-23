
from system_folder.system_functions import *


def c_theta(theta):
    global s
    return s* (1 + (2/np.sqrt(3)-1)/(np.sqrt(2)-1) * ( np.abs(np.sin(3*theta/2)) + np.abs(np.cos(3*theta/2)) - 1))**(-1)

def f(k_, e_, theta, eta,t,a):
    return e_ + 2*t*(sp.cos(k_*sp.cos(theta)*a) + 2*sp.cos(1/2*k_*sp.cos(theta)*a)*sp.cos(sp.sqrt(3)/2*k_*sp.sin(theta)*a)) + eta

def get_ki_array__theta(e_, eta,t,a):
    k_ = sp.Symbol('k')
    func_np = lambda theta : sp.lambdify(k_, f(k_, e_, theta, eta,t,a))
    return lambda theta : chebfun(lambda ki : func_np(theta)(ki), [0, c_theta(theta)]).roots()

def fprime__theta(ki, e_, eta,t,a):
    k_ = sp.symbols('k')
    return lambda theta : sp.lambdify(k_, f(k_, e_, theta, eta,t,a).diff(k_), modules=['numpy'])(ki)

def integrand(theta, e_, eta,t,a):
    ki_array = get_ki_array__theta(e_, eta,t,a)(theta)
    sum_ = 0
    for ki in ki_array:
        fp = fprime__theta(ki, e_, eta,t,a)(theta)
        if fp != 0:
            sum_ += ki/abs(fprime__theta(ki, e_, eta,t,a)(theta))
    return sum_

def Depsilon(e_, eta,t,a):
    global N_,AeBZ
    return 2*N_/AeBZ * quad(integrand, 0, 2*np.pi, args=(e_,eta,t,a), epsabs=1e-18, full_output=1)[0]

def Depsilon_array_GKMG(epsilons_GKMG, eta, t=t_,a=a_):
    tmp = []
    for e_ in tqdm.tqdm(epsilons_GKMG):
        tmp += [Depsilon(e_, eta,t,a)]
    return tmp

def get_lambda_Delta(eta_, th_array_, t=t_,a=a_):
    Nth = len(th_array_)
    D0 = Depsilon(e_=eta_, eta=0,t=t,a=a)
    N0 = D0/2

    Nu = np.load(f"arrays/FS with  eta={eta_}/coupling matrix.npy")
    Nu_new = np.zeros((4*Nth,4*Nth), dtype=np.complex_)
    for thi in range(Nth):
        for thj in range(Nth):
            Nu_new[4*thi:(4*thi+4), 4*thj:(4*thj+4)] = Nu[:,:,thi,thj]

    w,v = np.linalg.eig(-N0/Nth *Nu_new)
    ind_max = np.argwhere(w == np.amax(w)).flatten()[0]
    lambda_ = w[ind_max]
    Delta = v[:, ind_max]
    Delta = np.around(Delta, 12)

    return lambda_, Delta


def get_Tc_linearised(eta_, th_array_):
    wc = np.amax(np.load("arrays/omegas_fullBZ.npy"))
    EMc = float(sp.EulerGamma)
    kB = sc.constants.Boltzmann
    lambda_, _ = get_lambda_Delta(eta_, th_array_, t=t_,a=a_)

    return 2/(np.pi) *np.exp(EMc) *wc *np.exp(-1/lambda_)