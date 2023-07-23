# %%
from system_folder.system_functions import *

# %%
def check_correspondence(q_vectors_, Tinv_input, omegas_input, period, nrows, disable_tqdm=True):
    m=period*nrows
    print("")
    pum = create_para_unit_matrix(m)
    max_power=[]
    for qi_ in tqdm.tqdm(range(q_vectors_.shape[0]), disable=disable_tqdm):
        H_ = H_matrix(q_vectors_[qi_,:], period,nrows, J_matrix)
        for i in range(omegas_input[qi_, :].shape[0]):
            tmp = np.around((pum@H_-0.5*omegas_input[qi_, i]*pum)@Tinv_input[qi_, :,i],15)
            if not tmp.all() == 0:
                max_power += [np.max(np.log10(np.abs(tmp[i])))]
    print("Check for correspondence is finished! Result:")
    if max_power:
        print(f"    Inconsistency... Greatest offset of order 1e{np.round(np.max(max_power),2)}.")
    else:
        print("    Its goodie!!")


def check_diagonalisation_condition_all(q_vectors, Tinv_input, omegas_input, period, nrows, disable_tqdm=True):
    print("")
    counter=0
    idcs = []
    tmp_max = []
    for qi_ in tqdm.tqdm(range(q_vectors.shape[0]), disable=disable_tqdm):
        H_ = H_matrix(q_vectors[qi_,:], period,nrows, J_matrix)
        tmp = Tinv_input[qi_, :,:].T.conj()@H_@Tinv_input[qi_, :,:]-0.5*np.diag(omegas_input[qi_, :])
        tmp -= np.diag(np.diag(tmp))
        if np.amax(np.abs(tmp)) > 1e-8:
            counter += 1
            idcs += [qi_]
            tmp_max += [np.amax(np.abs(tmp))]
    print("Check of diagonalisation condition for all q-points is finished! Result:")
    if counter>0:
        print(f"    Dessverre: {counter} av {q_vectors.shape[0]}")
        print(f"    Største offset: {np.amax(np.abs(tmp_max))}")
        print( "    Indekser hvor ting går galt:", idcs)
    else:
        print( "    Its goodie!!")


def check_bosonic_all(q_vectors, Tinv_input, pum, disable_tqdm=True):
    print("")
    counter=0
    idcs = []
    tmp_max = []
    for qi_ in tqdm.tqdm(range(q_vectors.shape[0]), disable=disable_tqdm):
        tmp = pum - Tinv_input[qi_, :,:].T.conj()@pum@Tinv_input[qi_, :,:]
        if np.amax(np.abs(tmp)) > 1e-8:   ## Should be zero
            counter += 1
            idcs += [qi_]
            tmp_max += [np.amax(np.abs(tmp))]
    print("Check of bosonic condition is finished! Result:")
    if counter>0:
        print(f"    Dessverre: {counter} av {q_vectors.shape[0]}")
        print(f"    Største offset: {np.amax(np.abs(tmp_max))}")
        print( "    Indekser hvor ting går galt:", idcs)
    else:
        print( "    Its goodie!!")


