from system_folder.system_functions import *
from system_folder.qvalue_arrays import *
from system_folder.DoS_gap_functions import *

# %%
# --- Functions required for hovering bands ---

def get_text(l):
    return l.get_label()


def update_annot(l,ind, text_arr, annot):
    x,y = l.get_data()
    annot.xy = (x[ind["ind"][0]], y[ind["ind"][0]])
    annot.set_text("\n".join(text_arr))
    annot.get_bbox_patch().set_alpha(0.4)


def hover(event, fig,ax,annot,lines):
    vis = annot.get_visible()
    if event.inaxes == ax:
        text=[]
        for line in lines:
            cont, ind = line.contains(event)
            if cont:
                text.insert(0,get_text(line))
                update_annot(line,ind,text, annot)
                annot.set_visible(True)
                fig.canvas.draw_idle()
            else:
                if vis:
                    annot.set_visible(False)
                    fig.canvas.draw_idle()


def update_annot_coords(l,ind, annot_coords):
    x,y = l.get_data()
    xi, yi = (x[ind["ind"][0]], y[ind["ind"][0]])
    annot_coords.xy = (xi, yi)
    annot_coords.set_text(f"({round(xi,10)}, {round(yi,10)})")
    annot_coords.get_bbox_patch().set_alpha(0.4)    
    print (f'x = {round(xi,10)}, y = {round(yi,10)}                          ', end='\r')

def onclick(event, annot_coords,lines):
    for line in lines:
        cont, ind = line.contains(event)
        if cont:
            update_annot_coords(line, ind, annot_coords)
            annot_coords.set_visible(True)


def update_annot_coords_theta(l,ind, annot_coords):
    x,y = l.get_data()
    xi, yi = (x[ind["ind"][0]], y[ind["ind"][0]])
    annot_coords.xy = (xi, yi)
    xi_frac = str(Fraction(round(xi,10)/np.pi).limit_denominator())
    if "/" in xi_frac:
        num,denom = xi_frac.split('/')
        if num == '1':
            num = ""
        frac="/"
        xi_text = r"\frac{{{}}}{{{}}}".format(num+"\pi", denom)
    else:
        num=xi_frac
        if num == "1":
            num = ""
        frac=denom=""
        xi_text = f"{num}\pi"
    annot_coords.set_text(r"$({}, {})$".format(xi_text, round(yi,10)))
    annot_coords.get_bbox_patch().set_alpha(0.4)    
    print (f'x = {num}\u03C0{frac}{denom} (={round(xi,10)}),   y = {round(yi,10)}                          ', end='\r')

def onclick_theta(event, annot_coords,lines):
    for line in lines:
        cont, ind = line.contains(event)
        if cont:
            update_annot_coords_theta(line,ind, annot_coords)
            annot_coords.set_visible(True)


def onclick_labels(event,lines):
    text=[]
    for line in lines:
        cont, _ = line.contains(event)
        if cont:
            text.insert(0,get_text(line))
    for tx in text:
        print(tx)


def onclick_labels_multiple(event,all_lines):
    text=[]
    for lines in all_lines:
        for line in lines:
            cont, _ = line.contains(event)
            if cont:
                text.insert(0,get_text(line))
        for tx in text:
            print(tx)


def get_radian_ticks(x_tick_values):
    lst = []
    for i in range(len(x_tick_values)):
        el = str(Fraction(x_tick_values[i]/np.pi).limit_denominator()).split('/')
        if len(el) > 1:
            num,denum = el
            if num == "1":
                num = "\pi"
            else:
                num += "\pi"
            lst += [fr"$\frac{{{num}}}{{{denum}}}$"]
        else:
            if el[0] == "1":
                el[0] = "\pi"
            elif el[0] == "0":
                el[0] = "0"
            else:
                el[0] += "\pi"
            lst += ["$" + el[0] + "$"]
    return lst


# %%
""" For DoS """
def do_hex_plot(axi):
    global s, b_mag_electron
    th0i = 400
    thetas = np.arange(0, 2*np.pi+np.pi/th0i, np.pi/th0i)
    kx = c_theta(thetas)*np.cos(thetas)
    ky = c_theta(thetas)*np.sin(thetas)

    ## Hexagon
    axi.plot(kx, ky, lw=1.5, c='k')

    ## GK
    axi.plot([0,s], [0,0], color='r', ls='--', lw=1)
    axi.text(-1,0, r"$\mathbf{\Gamma}$", fontsize=12, verticalalignment='center')
    ## GM
    axi.plot([0,b_mag_electron/2*np.cos(np.pi/6)], [0,b_mag_electron/2*np.sin(np.pi/6)], color='r', ls='--', lw=1)
    axi.text(b_mag_electron/2*np.cos(np.pi/6)+0.15,b_mag_electron/2*np.sin(np.pi/6), r"$\mathbf{M}$", fontsize=12)
    ## MK
    axi.plot([b_mag_electron/2*np.cos(np.pi/6), s], [b_mag_electron/2*np.sin(np.pi/6), 0], color='r', ls='--', lw=1)
    axi.text(s+0.1,0, r"$\mathbf{K}$", fontsize=12, verticalalignment='center')


    axi.set_xlim(-s-0.5, s+0.5)
    axi.set_ylim(-s-0.5, s+0.5)
    axi.set_aspect('equal')
    axi.axis('off')


def create_DoS_plot(epsilon, Qs_GKMG, Qs_pts, save_fig=False, recalculate=False):
    try:
        e90 = np.load("arrays/DoS/epsilon_90pts.npy")
        De90 = np.load("arrays/DoS/density of states 90pts.npy")
        if recalculate: raise Exception
    except:
        e90 = np.linspace(np.amin(epsilon) - 0.01, np.amax(epsilon) + 0.01, 90)
        De90 = Depsilon_array_GKMG(e90, eta=0)
        np.save("arrays/DoS/epsilon_90pts.npy", e90)
        np.save("arrays/DoS/density of states 90pts.npy", De90)

    fig = plt.figure(tight_layout=True, figsize=(8,4))
    gs = gridspec.GridSpec(1,3)

    ax1 = fig.add_subplot(gs[0,:2])
    ax2 = fig.add_subplot(gs[0,2], sharey=ax1)

    ax1.plot(Qs_GKMG, epsilon, lw=1)
    ax1.axhline(2, color='k', ls='--', lw=0.8)
    ax2.axhline(2, color='k', ls='--', lw=0.8)

    ax12 = fig.add_axes([0.485, 0.6, 0.2, 0.2])
    do_hex_plot(ax12)

    for Qs_pt in Qs_pts:
        ax1.axvline(Qs_pt, lw=1, c='k')
    tick_labels = [r"$\mathbf{\Gamma}$", r"$\mathbf{M}$", r"$\mathbf{K}$", r"$\mathbf{\Gamma}$"]

    ax1.set_ylabel(r"$\epsilon_\mathbf{k}/t$", fontsize=14)
    ax1.set_xlim(0,Qs_GKMG[-1])
    ax1.set_xticks(Qs_pts,labels=tick_labels, fontsize=13)

    ax2.plot(De90, e90, lw=1, label="90")

    ax2.set_xlim(0,1)
    ax2.set_xticks([0,0.5,1])
    ax2.xaxis.set_minor_locator(MultipleLocator(0.25))
    ax2.set_xlabel(r"$\frac{t}{N}D(\epsilon)$")

    if save_fig:
        plt.savefig(f'plots svg/electron-DoS-GKMG.svg', dpi=1000, format='svg', bbox_inches='tight')
        plt.savefig(f'plots jpg/electron-DoS-GKMG.jpg', dpi=1000, format='jpg', bbox_inches='tight')
    plt.show()


# %%
def create_magnon_spectrum_GXSYG(ws_m_GXSYG, Qs, Qs_GXSYG_pts, period=n, save=False, plotLegend=True, plotBZ=True, coloured=True, shorter=False):
    # --- Helping function for plotting the magnon spectrum GXSYG ---
    global b1_electron, b2_electron
    m = ws_m_GXSYG.shape[-1]
    
    symmetry_points = "Gamma-X-S-Y-Gamma"
    sp = symmetry_points.split('-')

    names = np.array([f"band {i}" for i in range(m)])
    cmap=mpl.cm.get_cmap('hsv')

    fig = plt.figure(figsize=(8,6))
    if shorter:
        fig.set_figheight(5)
        ax  = fig.add_axes([0.1,0.085,0.7,0.8])
    else:
        ax  = fig.add_axes([0.1,0.085,0.7,0.8])

    lines = []
    for i in range(m):
        if coloured:
            l, = ax.plot(Qs, ws_m_GXSYG[:,i], label=names[i], c=cmap(1-i/m), lw=1.2)
        else:
            l, = ax.plot(Qs, ws_m_GXSYG[:,i], label=names[i], c='k', lw=1)
        lines.append(l)

    handles,labels = ax.get_legend_handles_labels()
    handles,labels = np.flip(handles),np.flip(labels)
    annot = ax.annotate("", xy=(0, 0), xytext=(-20, 20), textcoords="offset points",
                        bbox=dict(boxstyle="round", fc="w"),
                        arrowprops=dict(arrowstyle="->"))
    annot.set_visible(False)
    annot_coords = ax.annotate("", xy=(0, 0), xytext=(-20, -30), textcoords="offset points",
                        bbox=dict(boxstyle="round", fc="w"),
                        arrowprops=dict(arrowstyle="->"))
    annot_coords.set_visible(False)

    ticks=[]
    tick_labels=[]
    for i in range(len(sp)):
        ax.axvline(Qs_GXSYG_pts[i], color='k', lw=1)
        ticks += [Qs_GXSYG_pts[i]]
        if sp[i] == "Gamma":
            tick_labels += [r"$\Gamma$"]
        else:
            tick_labels += [fr"${sp[i]}$"]

# ---------------------------------------------------------------------------------- #
    if plotLegend:
        order = np.arange(m-1,-1,-1)
        legend_handles = []
        legend_labels  = []
        for i in order:
            legend_handles += [handles[i]]
            legend_labels  += [labels[i]]
        ax.legend(legend_handles, legend_labels, bbox_to_anchor=(1.227, 1.017))#(1, 1.017))
    ax.grid(c='gainsboro')
    ax.set_ylabel(r"$\omega_{\mathbf{q}\gamma}$ [eV/$\hbar$]", fontsize=14)
    ax.set_xlim(0,Qs[-1])
    ax.set_xticks(ticks,labels=tick_labels, fontsize=13)
    ax.tick_params(axis='y', labelsize=10)
# ---------------------------------------------------------------------------------- #
    if plotBZ:
        if shorter:
            axins = fig.add_axes([0.85, 0.085, 0.1, 0.3])
        else:
            axins = fig.add_axes([0.85, 0.085, 0.1, 0.3])
        axins.quiver([0,0], [0,0], [b1_magnon[0], b2_magnon[0]], [b1_magnon[1], b2_magnon[1]], scale=1, units='xy', color='C0', width=0.2)
        axins.plot([-b1_magnon[0]/2, b1_magnon[0]/2],[ b2_magnon[1]/2,  b2_magnon[1]/2], color='k')
        axins.plot([ b1_magnon[0]/2, b1_magnon[0]/2],[-b2_magnon[1]/2,  b2_magnon[1]/2], color='k')
        axins.plot([-b1_magnon[0]/2, b1_magnon[0]/2],[-b2_magnon[1]/2, -b2_magnon[1]/2], color='k')
        axins.plot([-b1_magnon[0]/2,-b1_magnon[0]/2],[ b2_magnon[1]/2, -b2_magnon[1]/2], color='k')
        axins.plot(0,b2_magnon[1]/2, color='C3', marker='.')
        axins.plot(0,0, color='C3', marker='.')
        axins.plot(b1_magnon[0]/2,b1_magnon[1]/2, color='C3', marker='.')
        axins.plot(b1_magnon[0]/2,b1_magnon[1]/2+b2_magnon[1]/2, color='C3', marker='.')

        axins.set_title('mBZ1')
        axins.text(-1.3,-0.8, r'$\Gamma$', color='C3', fontsize=12)
        axins.text(-1.3,b2_magnon[1]/2+0.3, r'$Y$', color='C3', fontsize=12)
        axins.text(b1_magnon[0]/2+0.3,-0.8, r'$X$', color='C3', fontsize=12)
        axins.text(b1_magnon[0]/2+0.3,b2_magnon[1]/2+0.3, r'$S$', color='C3', fontsize=12)

        axins.set_xlim(-2.8*b1_magnon[0]/2, 2.8*b1_magnon[0]/2)
        axins.set_ylim(-1.5*b2_magnon[1]/2, 2.1*b2_magnon[1]/2)
        axins.tick_params(labelleft=False, labelbottom=False, direction='in')
        axins.set_aspect('equal', adjustable='box')
        axins.set_xlabel('$q_x$')
        axins.set_ylabel('$q_y$')
# ---------------------------------------------------------------------------------- #
    text_shorter_dict = {True : "_shorter", False : ""}
    if save:
        plt.savefig(f'plots svg/magnon-spectrum_{symmetry_points}__n{period}_BZ{plotBZ}_legend{plotLegend}{text_shorter_dict.get(shorter)}.svg', dpi=1000, format='svg', bbox_inches='tight')
        plt.savefig(f'plots jpg/magnon-spectrum_{symmetry_points}__n{period}_BZ{plotBZ}_legend{plotLegend}{text_shorter_dict.get(shorter)}.jpg', dpi=1000, format='jpg', bbox_inches='tight')
# ---------------------------------------------------------------------------------- #
    h = lambda evnt : hover(evnt, fig,ax,annot,lines)
    oc = lambda evnt : onclick(evnt, annot_coords,lines)
    fig.canvas.mpl_connect("motion_notify_event", h)
    fig.canvas.mpl_connect('button_press_event', oc)
    plt.show()

def plot_magnon_spectrum_GXSYG(period=n, nrows=n_chains, save=False, plotLegend=False, plotBZ=True, coloured=True, shorter=True, recalculate_data=False):

    if recalculate_data:
        disable_tqdm=False
        noX,noY = (40,40)
        q_vecs_ = get_q_vectors_fullBZ(filename="q_vectors_fullBZ", bz_x=BZ_x, bz_y=BZ_y, nox=noX, noy=noY, recalculate=recalculate_data)
        print("omegas and Tinv... \t\t\t\t", end='\r')
        omegas_, Tinv_ = get_2m_bands_fullBZ(q_vecs_, filenames=["omegas_fullBZ", "Tinv_fullBZ"], disable_tqdm_qs=disable_tqdm, period=period,nrows=nrows, recalculate=recalculate_data)
            ### v
        print("omegas and Tinv sorted... \t\t\t\t", end='\r')
        omegas_sorted, Tinv_sorted = sort_bands_fullBZ(omegas_, Tinv_, filenames=["omegas_sorted_fullBZ", "Tinv_sorted_fullBZ"], disable_tqdm_qs=disable_tqdm, recalculate=recalculate_data)

        print("other... \t\t\t\t", end='\r')
        ind_GXSYG = get_GXSYG_indices(q_vecs_, bz_x=BZ_x, bz_y=BZ_y)
        q_GXSYG = get_array_GXSYG(q_vecs_, ind_GXSYG, filename="q_GXSYG", recalculate=recalculate_data)
        omegas_2m_GXSYG = get_array_GXSYG(omegas_sorted, ind_GXSYG, filename="omegas_2m_GXSYG", recalculate=recalculate_data)

        Qs = get_Qs(q_GXSYG, Qs_filename_="Qs_GXSYG", recalculate=recalculate_data)
        Qs_GXSYG_pts = get_symmetry_pts_GXSYG(Qs_GXSYG_pts_filename_="Qs_pts_GXSYG", Qs_GXSYG=Qs, q_GXSYG_=q_GXSYG, recalculate=recalculate_data)

        omegas_m_GXSYG = omegas_2m_GXSYG[:,:m]
        ind = np.argsort(omegas_m_GXSYG[0,:])

        print("plotting... \t\t\t\t\t\t\t\t\t\t", end='\r')
    else:
        omegas_m_GXSYG = np.load("arrays/omegas_2m_GXSYG.npy", allow_pickle=True)[:,:m]
        ind = np.argsort(omegas_m_GXSYG[0,:])
        Qs = np.load("arrays/Qs_GXSYG.npy", allow_pickle=True)
        Qs_GXSYG_pts = np.load("arrays/Qs_pts_GXSYG.npy", allow_pickle=True)

    create_magnon_spectrum_GXSYG(omegas_m_GXSYG[:,ind], Qs, Qs_GXSYG_pts, \
                                 period=n, save=save, plotLegend=plotLegend, plotBZ=plotBZ, shorter=True)


### For our curiousity
def create_magnon_spectrum_FS(ws_m, th_array_, theta_tick_values, theta_tick_labels, period=n, save=False, plotLegend=True, plotBZ=True, coloured=True, shorter=False):
    # --- Plotting function for magnon spectrum GXSYG ---
    global b1_electron, b2_electron
    
    symmetry_points = "Gamma-X-S-Y-Gamma"
    sp = symmetry_points.split('-')

    names = np.array([f"band {i}" for i in range(m)])
    cmap=mpl.cm.get_cmap('hsv')

    fig = plt.figure(figsize=(8,6))
    if shorter:
        fig.set_figheight(5)
        ax  = fig.add_axes([0.1,0.085,0.7,0.8])
    else:
        ax  = fig.add_axes([0.1,0.085,0.7,0.8])

    lines = []
    for i in range(m):
        if coloured:
            l, = ax.plot(th_array_, ws_m[:,i], label=names[i], c=cmap(1-i/m), lw=1.2)
        else:
            l, = ax.plot(th_array_, ws_m[:,i], label=names[i], c='k', lw=1)
        lines.append(l)

    handles,labels = ax.get_legend_handles_labels()
    handles,labels = np.flip(handles),np.flip(labels)
    annot = ax.annotate("", xy=(0, 0), xytext=(-20, 20), textcoords="offset points",
                        bbox=dict(boxstyle="round", fc="w"),
                        arrowprops=dict(arrowstyle="->"))
    annot.set_visible(False)
    annot_coords = ax.annotate("", xy=(0, 0), xytext=(-20, -30), textcoords="offset points",
                        bbox=dict(boxstyle="round", fc="w"),
                        arrowprops=dict(arrowstyle="->"))
    annot_coords.set_visible(False)

# ---------------------------------------------------------------------------------- #
    if plotLegend:
        order = np.arange(m-1,-1,-1)
        legend_handles = []
        legend_labels  = []
        for i in order:
            legend_handles += [handles[i]]
            legend_labels  += [labels[i]]
        ax.legend(legend_handles, legend_labels, bbox_to_anchor=(1.227, 1.017))#(1, 1.017))
    ax.grid(c='gainsboro')
    ax.set_ylabel(r"$\omega_{\mathbf{q}\gamma}$ [eV/$\hbar$]", fontsize=14)
    ax.set_xlim(th_array_[0], th_array_[-1])
    ax.set_xticks(theta_tick_values,   theta_tick_labels)
    ax.set_xlabel(r"$\theta$")
    ax.xaxis.set_minor_locator(MultipleLocator(np.pi/4))
    ax.set_xlim(th_array_[0],th_array_[-1])
    ax.tick_params(axis='y', labelsize=10)
# ---------------------------------------------------------------------------------- #
    if plotBZ:
        if shorter:
            axins = fig.add_axes([0.85, 0.085, 0.1, 0.3])
        else:
            axins = fig.add_axes([0.85, 0.085, 0.1, 0.3])
        axins.quiver([0,0], [0,0], [b1_magnon[0], b2_magnon[0]], [b1_magnon[1], b2_magnon[1]], scale=1, units='xy', color='C0', width=0.2)
        axins.plot([-b1_magnon[0]/2, b1_magnon[0]/2],[ b2_magnon[1]/2,  b2_magnon[1]/2], color='k')
        axins.plot([ b1_magnon[0]/2, b1_magnon[0]/2],[-b2_magnon[1]/2,  b2_magnon[1]/2], color='k')
        axins.plot([-b1_magnon[0]/2, b1_magnon[0]/2],[-b2_magnon[1]/2, -b2_magnon[1]/2], color='k')
        axins.plot([-b1_magnon[0]/2,-b1_magnon[0]/2],[ b2_magnon[1]/2, -b2_magnon[1]/2], color='k')
        axins.plot(0,b2_magnon[1]/2, color='C3', marker='.')
        axins.plot(0,0, color='C3', marker='.')
        axins.plot(b1_magnon[0]/2,b1_magnon[1]/2, color='C3', marker='.')
        axins.plot(b1_magnon[0]/2,b1_magnon[1]/2+b2_magnon[1]/2, color='C3', marker='.')

        axins.set_title('1st BZ')
        axins.text(-1.3,-0.8, r'$\Gamma$', color='C3', fontsize=12)
        axins.text(-1.3,b2_magnon[1]/2+0.3, r'$Y$', color='C3', fontsize=12)
        axins.text(b1_magnon[0]/2+0.3,-0.8, r'$X$', color='C3', fontsize=12)
        axins.text(b1_magnon[0]/2+0.3,b2_magnon[1]/2+0.3, r'$S$', color='C3', fontsize=12)

        axins.set_xlim(-2.8*b1_magnon[0]/2, 2.8*b1_magnon[0]/2)
        axins.set_ylim(-1.5*b2_magnon[1]/2, 2.1*b2_magnon[1]/2)
        axins.tick_params(labelleft=False, labelbottom=False, direction='in')
        axins.set_aspect('equal', adjustable='box')
        axins.set_xlabel('$q_x$')
        axins.set_ylabel('$q_y$')
# ---------------------------------------------------------------------------------- #
    text_shorter_dict = {True : "_shorter", False : ""}
    if save:
        plt.savefig(f'plots svg/magnon-spectrum_{symmetry_points}__n{period}_BZ{plotBZ}_legend{plotLegend}{text_shorter_dict.get(shorter)}.svg', dpi=1000, format='svg', bbox_inches='tight')
        plt.savefig(f'plots jpg/magnon-spectrum_{symmetry_points}__n{period}_BZ{plotBZ}_legend{plotLegend}{text_shorter_dict.get(shorter)}.jpg', dpi=1000, format='jpg', bbox_inches='tight')
# ---------------------------------------------------------------------------------- #

    h = lambda evnt : hover(evnt, fig,ax,annot,lines)
    oc_t = lambda evnt : onclick_theta(evnt, annot_coords,lines)
    fig.canvas.mpl_connect("motion_notify_event", h)
    fig.canvas.mpl_connect('button_press_event', oc_t)
    plt.show()



# %%
def plot_Veff_FS(th_array_, th0i, eta, nrows=n_chains, period=n, do_print=False, disable_tqdm=False, save=False, part_vals="abs_vals", see_all_lines=False):
    """ Plotting from loading Veff """
    m = period*nrows
    pts = {"abs_vals" : "Absolute value", "real_vals" : "Real part", "imag_vals" : "Imaginary part"} ### part_vals to string dictionary

    fig = plt.figure()
    if not save:
        plt.suptitle(fr"Pair potential on the Fermi sphere: $\mu/t={eta}$ for $k'$ at $\theta'={get_radian_ticks(th_array_)[th0i][1:-1]}$", y=0.98)

    ax=plt.gca()

    theta_tick_values = np.linspace(th_array_[0], th_array_[-1], 5)
    theta_tick_labels = get_radian_ticks(theta_tick_values)
    ax.set_xticks(theta_tick_values,   theta_tick_labels)
    ax.set_xlabel(r"$\theta$")
    ax.xaxis.set_minor_locator(MultipleLocator(np.pi/4))
    ax.set_xlim(th_array_[0],th_array_[-1])

    ax.tick_params(axis='y', labelsize=10)
    ax.grid(c='gainsboro', which='both')
    
    sta = { -1: u"\u2193", 1 : u"\u2191" }   ## spin_to_arrows dict
    lines=[]
    if see_all_lines:
        if do_print:
            print(u"|V(\u03C3\u2081, \u03C3\u2082, \u03C3\u2083, \u03C3\u2084)| for q=0")
        if part_vals == "real_vals":
            spin_combinations = [(1,-1,1,-1), (-1,1,-1,1),   (1,1,1,1), (-1,-1,-1,-1),      (1,1,1,-1),  (1,1,-1,1),  (1,-1,1,1),  (-1,1,1,1),  (-1,-1,-1,1), (-1,-1,1,-1),  (-1,1,-1,-1),  (1,-1,-1,-1),   (1,1,-1,-1), (-1,-1,1,1),   (1,-1,-1,1), (-1,1,1,-1)]
            colors =            ["lightsalmon", "red",       "gold", "chocolate",           "mediumspringgreen", "green",   "darkgray", "black",  "navajowhite","darkorange",   "yellowgreen", "olive",    "skyblue", "royalblue",      "plum", "darkviolet"]
            ds =                [(5, 0), (5, 10),            (5, 0), (5, 10),               (5, 0), (5, 10), (5, 15), (5, 20), (5, 25), (5, 30), (5, 35), (5, 40),                                    (5, 0), (5, 10),          (5, 0), (5,10)]
            for i, (s1,s2,s3,s4) in enumerate(spin_combinations):
                Veff = np.load(f"arrays/FS with  eta={eta}/Veff_matrices/(+k)(+k')   {sta.get(s1)}{sta.get(s2)}{sta.get(s3)}{sta.get(s4)}.npy")[th0i,:]#*J/(10*mili)/J_L**2
                l, = ax.plot(th_array_, np.real(Veff), color=colors[i], \
                        label=r"${{{}{}{}{}}}$".format(sta.get(s1),sta.get(s2),sta.get(s3),sta.get(s4)), dashes=ds[i])
                ax.set_ylabel(r"$N \mathfrak{Re}( V^{\sigma_1\sigma_2\sigma_3\sigma_4}_{\mathbf{k}\mathbf{k}^\prime} ) $", fontsize=14)
                lines.append(l)
        elif part_vals == "imag_vals":
            spin_combinations = [(1,-1,-1,-1), (-1,-1,-1,1), (1,1,-1,1), (1,-1,1,1),       (-1,1,1,-1), (1,-1,-1,1), (-1,-1,1,1), (1,1,-1,-1), (1,1,1,1), (-1,-1,-1,-1), (-1,1,-1,1), (1,-1,1,-1),   (-1,1,-1,-1), (-1,-1,1,-1), (-1,1,1,1), (1,1,1,-1)]
            colors =            ["olive", "navajowhite",       "green", "darkgray",           "darkviolet", "plum",   "royalblue", "skyblue",  "gold","chocolate",   "red", "lightsalmon",    "yellowgreen", "darkorange",      "black", "mediumspringgreen"]
            ds =                [(5, 0), (5, 10), (5, 15), (5, 20),                         (5, 0), (5, 10), (5, 15), (5, 20), (5, 25), (5, 30), (5, 35), (5, 40),                                    (5, 0), (5, 10), (5, 15), (5,20)]
            for i, (s1,s2,s3,s4) in enumerate(spin_combinations):
                Veff = np.load(f"arrays/FS with  eta={eta}/Veff_matrices/(+k)(+k')   {sta.get(s1)}{sta.get(s2)}{sta.get(s3)}{sta.get(s4)}.npy")[th0i,:]#*J/(10*mili)/J_L**2
                l, = ax.plot(th_array_, np.imag(Veff), color=colors[i], \
                        label=r"${{{}{}{}{}}}$".format(sta.get(s1),sta.get(s2),sta.get(s3),sta.get(s4)), dashes=ds[i])
                ax.set_ylabel(r"$N \mathfrak{Im}( V^{\sigma_1\sigma_2\sigma_3\sigma_4}_{\mathbf{k}\mathbf{k}^\prime} ) $", fontsize=14)
                lines.append(l)
        else:
            raise Exception
        if do_print:
            print(f"|V({sta.get(s1)}{sta.get(s2)}{sta.get(s3)}{sta.get(s4)})| = {np.round(Veff[0], 10)}")
        plt.legend(handles=lines, loc='center right', bbox_to_anchor=(1.25,0.5), ncol=1)
    else:   ## see_all_lines = False
        if do_print:
            print(u"|V(\u03C3\u2081, \u03C3\u2082, \u03C3\u2083, \u03C3\u2084)| for q=0")
        if part_vals == "real_vals":
            spin_combinations = [(1,-1,1,-1), (-1,1,-1,1),   (1,1,1,1), (-1,-1,-1,-1),   (1,1,1,-1),  (1,1,-1,1),  (1,-1,1,1),  (-1,1,1,1),  (-1,-1,-1,1), (-1,-1,1,-1),  (-1,1,-1,-1),  (1,-1,-1,-1),     (1,1,-1,-1), (-1,-1,1,1),     (1,-1,-1,1), (-1,1,1,-1)]
            colors =            ["crimson", "crimson",       "gold", "gold",             "limegreen", "limegreen", "limegreen", "limegreen", "limegreen", "limegreen", "limegreen", "limegreen",           "royalblue", "royalblue",     "purple", "purple"]
            for i, (s1,s2,s3,s4) in enumerate(spin_combinations):
                Veff = np.load(f"arrays/FS with  eta={eta}/Veff_matrices/(+k)(+k')   {sta.get(s1)}{sta.get(s2)}{sta.get(s3)}{sta.get(s4)}.npy")[:,th0i]#*J/(10*mili)/J_L**2
                l, = ax.plot(th_array_, np.real(Veff), color=colors[i], lw=1.2, \
                        label=r"${{{}{}{}{}}}$".format(sta.get(s1),sta.get(s2),sta.get(s3),sta.get(s4)))
                ax.set_ylabel(r"$N \mathfrak{Re}( V^{\sigma_1\sigma_2\sigma_3\sigma_4}_{\mathbf{k}\mathbf{k}^\prime} ) $", fontsize=14)
                lines.append(l)
                if do_print:
                    print(f"|V({sta.get(s1)}{sta.get(s2)}{sta.get(s3)}{sta.get(s4)})| = {np.round(Veff[0], 10)}")
        elif part_vals == "imag_vals":
            spin_combinations = [(1,-1,-1,-1), (-1,-1,-1,1), (1,1,-1,1), (1,-1,1,1),                     (-1,1,1,-1), (1,-1,-1,1), (-1,-1,1,1), (1,1,-1,-1), (1,1,1,1), (-1,-1,-1,-1), (-1,1,-1,1), (1,-1,1,-1),                             (-1,1,-1,-1), (-1,-1,1,-1), (-1,1,1,1), (1,1,1,-1)]
            colors =            ["lightseagreen", "lightseagreen", "lightseagreen", "lightseagreen",     "palevioletred","palevioletred","palevioletred","palevioletred","palevioletred","palevioletred","palevioletred","palevioletred",    "olivedrab",  "olivedrab",  "olivedrab","olivedrab"]
            for i, (s1,s2,s3,s4) in enumerate(spin_combinations):
                Veff = np.load(f"arrays/FS with  eta={eta}/Veff_matrices/(+k)(+k')   {sta.get(s1)}{sta.get(s2)}{sta.get(s3)}{sta.get(s4)}.npy")[:,th0i]#*J/(10*mili)/J_L**2
                l, = ax.plot(th_array_, np.imag(Veff), color=colors[i], lw=1.2, \
                        label=r"${{{}{}{}{}}}$".format(sta.get(s1),sta.get(s2),sta.get(s3),sta.get(s4)))
                ax.set_ylabel(r"$N \mathfrak{Im}( V^{\sigma_1\sigma_2\sigma_3\sigma_4}_{\mathbf{k}\mathbf{k}^\prime} ) $", fontsize=14)
                lines.append(l)
                if do_print:
                    print(f"|V({sta.get(s1)}{sta.get(s2)}{sta.get(s3)}{sta.get(s4)})| = {np.round(Veff[0], 10)}")
        else:
            raise Exception
        if part_vals=="real_vals":
            legend1 = plt.legend(handles=[lines[0], lines[1]],              loc="upper right",  bbox_to_anchor=(1.15,1.025), handletextpad=0.0, handlelength=0, framealpha=1, edgecolor=lines[1].get_color())#[lines[0].get_color(), lines[1].get_color()])
            legend2 = plt.legend(handles=[lines[i_] for i_ in range(2,4)],  loc="center right", bbox_to_anchor=(1.15,0.806), handletextpad=0.0, handlelength=0, framealpha=1, edgecolor=lines[3].get_color())#, labelcolor=[lines[i_].get_color() for i_ in range(2,6)])
            legend3 = plt.legend(handles=[lines[i_] for i_ in range(4,12)],  loc="lower right",  bbox_to_anchor=(1.15,0.19+0.053), handletextpad=0.0, handlelength=0, framealpha=1, edgecolor=lines[11].get_color())#, labelcolor=[lines[i_].get_color() for i_ in range(6,8)])
            legend4 = plt.legend(handles=[lines[i_] for i_ in range(12,14)], loc="lower right",  bbox_to_anchor=(1.15,0.05+0.06), handletextpad=0.0, handlelength=0, framealpha=1, edgecolor=lines[13].get_color())#, labelcolor=[lines[i_].get_color() for i_ in range(8,16)])
            legend5 = plt.legend(handles=[lines[i_] for i_ in range(14,16)], loc="lower right",  bbox_to_anchor=(1.15,-0.025), handletextpad=0.0, handlelength=0, framealpha=1, edgecolor=lines[15].get_color())#, labelcolor=[lines[i_].get_color() for i_ in range(8,16)])
            ax.add_artist(legend1)
            ax.add_artist(legend2)
            ax.add_artist(legend3)
            ax.add_artist(legend4)
            ax.add_artist(legend5)
        elif part_vals=="imag_vals":
            legend1 = plt.legend(handles=[lines[i_] for i_ in range(0,4)],    loc="center right", bbox_to_anchor=(1.15,0.12), handletextpad=0.0, handlelength=0, framealpha=1, edgecolor=lines[3].get_color())#, labelcolor=[lines[i_].get_color() for i_ in range(2,6)])
            legend2 = plt.legend(handles=[lines[i_] for i_ in range(4,12)],   loc="center right", bbox_to_anchor=(1.15,0.5),    handletextpad=0.0, handlelength=0, framealpha=1, edgecolor=lines[11].get_color())#, labelcolor=[lines[i_].get_color() for i_ in range(2,6)])
            legend3 = plt.legend(handles=[lines[i_] for i_ in range(12,16)],  loc="lower right",  bbox_to_anchor=(1.15,0.74),  handletextpad=0.0, handlelength=0, framealpha=1, edgecolor=lines[15].get_color())#, labelcolor=[lines[i_].get_color() for i_ in range(6,8)])
            ax.add_artist(legend1)
            ax.add_artist(legend2)
            ax.add_artist(legend3)
    if save:
        pts_save = {"abs_vals" : "abs", "real_vals" : "real", "imag_vals" : "imag"} ### part_vals to string dictionary
        type_save ={1 : "allLines", 0 : "colorMatched"}
        plt.savefig(f'plots svg/{pts_save.get(part_vals)}_{type_save.get(see_all_lines)}_Veff-plot-FS_eta{eta}__n{period}.svg', dpi=1000, format='svg', bbox_inches='tight')
        plt.savefig(f'plots jpg/{pts_save.get(part_vals)}_{type_save.get(see_all_lines)}_Veff-plot-FS_eta{eta}__n{period}.jpg', dpi=1000, format='jpg', bbox_inches='tight')

    annot = ax.annotate("", xy=(0, 0), xytext=(-20, 20), textcoords="offset points",
                        bbox=dict(boxstyle="round", fc="w"),
                        arrowprops=dict(arrowstyle="->"))
    annot.set_visible(False)
    annot_coords = ax.annotate("", xy=(0, 0), xytext=(-20, -30), textcoords="offset points",
                        bbox=dict(boxstyle="round", fc="w"),
                        arrowprops=dict(arrowstyle="->"))
    annot_coords.set_visible(False)

    h = lambda evnt : hover(evnt, fig,ax,annot,lines)
    oc_t = lambda evnt : onclick_theta(evnt, annot_coords,lines)
    fig.canvas.mpl_connect("motion_notify_event", h)
    fig.canvas.mpl_connect('button_press_event', oc_t)
    plt.show()



# %%
def plot_Veffbar_FS(th_array_, th0i, eta, nrows=n_chains, period=n, do_print=False, disable_tqdm=False, save=False, part_vals="abs_vals", see_all_lines=False):
    """ Plotting from loading Veffbar """
    m = period*nrows
    pts = {"abs_vals" : "Absolute value", "real_vals" : "Real part", "imag_vals" : "Imaginary part"} ### part_vals to string dictionary

    fig = plt.figure()
    if not save:
        plt.suptitle(fr"Pair potential on the Fermi sphere: $\mu/t={eta}$ for $k'$ at $\theta'={get_radian_ticks(th_array_)[th0i][1:-1]}$", y=0.98)

    ax=plt.gca()

    theta_tick_values = np.linspace(th_array_[0], th_array_[-1], 5)
    theta_tick_labels = get_radian_ticks(theta_tick_values)
    ax.set_xticks(theta_tick_values,   theta_tick_labels)
    ax.set_xlabel(r"$\theta$")
    ax.xaxis.set_minor_locator(MultipleLocator(np.pi/4))
    ax.set_xlim(th_array_[0],th_array_[-1])

    ax.tick_params(axis='y', labelsize=10)
    ax.grid(c='gainsboro', which='both',zorder=-1)
    
    sta = { -1: u"\u2193", 1 : u"\u2191" }   ## spin_to_arrows dict
    lines=[]
    if see_all_lines:
        if do_print:
            print(u"|Vbar(\u03C3\u2081, \u03C3\u2082, \u03C3\u2083, \u03C3\u2084)| for q=0")
        if part_vals == "real_vals":
            spin_combinations = [(1,-1,1,-1), (-1,1,-1,1),   (1,1,1,1), (-1,-1,-1,-1),     (1,1,1,-1),  (1,1,-1,1),      (1,-1,1,1), (-1,1,1,1), (-1,-1,-1,1), (-1,-1,1,-1),  (-1,1,-1,-1),  (1,-1,-1,-1),   (1,1,-1,-1), (-1,-1,1,1),    (1,-1,-1,1), (-1,1,1,-1)]
            colors =            ["lightsalmon", "red",       "skyblue", "royalblue",       "mediumspringgreen", "green", "darkgray", "black",    "navajowhite","darkorange",   "yellowgreen", "olive",     "gold", "chocolate",       "plum", "darkviolet"]
            ds =                [(5, 0), (5, 10),            (5, 0), (5, 10),               (5, 0), (5, 10), (5, 15), (5, 20), (5, 25), (5, 30), (5, 35), (5, 40),                                         (5, 0), (5, 10),              (5, 0), (5,10)]
            for i, (s1,s2,s3,s4) in enumerate(spin_combinations):
                Veffbar = np.load(f"arrays/FS with  eta={eta}/Veffbar_matrices/{sta.get(s1)}{sta.get(s2)}{sta.get(s3)}{sta.get(s4)}.npy")[th0i,:]
                l, = ax.plot(th_array_, np.real(Veffbar), color=colors[i], \
                        label=r"${{{}{}{}{}}}$".format(sta.get(s1),sta.get(s2),sta.get(s3),sta.get(s4)), dashes=ds[i])
                ax.set_ylabel(r"$N \mathfrak{Re}( \bar{V}^{\sigma_1\sigma_2\sigma_3\sigma_4}_{\mathbf{k}\mathbf{k}^\prime} ) $", fontsize=14)
                lines.append(l)
        elif part_vals == "imag_vals":
            spin_combinations = [(1,1,-1,1), (-1,-1,-1,1),     (1,-1,1,1), (1,-1,-1,-1),      (1,-1,1,-1), (-1,1,-1,1), (-1,-1,-1,-1), (1,1,1,1), (1,1,-1,-1), (-1,-1,1,1), (1,-1,-1,1), (-1,1,1,-1),        (-1,1,1,1), (-1,1,-1,-1),       (1,1,1,-1), (-1,-1,1,-1)]
            colors =            ["green",   "navajowhite",     "darkgray", "olive",           "lightsalmon",  "red",    "royalblue",    "skyblue", "gold",      "chocolate", "plum",      "darkviolet",      "black",    "yellowgreen",       "mediumspringgreen", "darkorange"]
            ds =                [(5, 0), (5, 10),              (5, 0), (5, 10),               (5, 0), (5, 10), (5, 15), (5, 20), (5, 25), (5, 30), (5, 35), (5, 40),                                         (5, 0), (5, 10),                (5, 0), (5,10)]
            for i, (s1,s2,s3,s4) in enumerate(spin_combinations):
                Veffbar = np.load(f"arrays/FS with  eta={eta}/Veffbar_matrices/{sta.get(s1)}{sta.get(s2)}{sta.get(s3)}{sta.get(s4)}.npy")[th0i,:]
                l, = ax.plot(th_array_, np.imag(Veffbar), color=colors[i], \
                        label=r"${{{}{}{}{}}}$".format(sta.get(s1),sta.get(s2),sta.get(s3),sta.get(s4)), dashes=ds[i])
                ax.set_ylabel(r"$N \mathfrak{Im}( \bar{V}^{\sigma_1\sigma_2\sigma_3\sigma_4}_{\mathbf{k}\mathbf{k}^\prime} ) $", fontsize=14)
                lines.append(l)
        else:
            raise Exception
        if do_print:
            print(f"|Vbar({sta.get(s1)}{sta.get(s2)}{sta.get(s3)}{sta.get(s4)})| = {np.round(Veffbar[0], 10)}")
        plt.legend(handles=lines, loc='center right', bbox_to_anchor=(1.25,0.5), ncol=1)
    else:   ## not see_all_lines
        if do_print:
            print(u"|Vbar(\u03C3\u2081, \u03C3\u2082, \u03C3\u2083, \u03C3\u2084)| for q=0")
        if part_vals == "real_vals":
            spin_combinations = [(1,-1,1,-1), (-1,1,-1,1),   (1,1,-1,-1), (-1,-1,1,1),       (1,1,1,-1),  (1,1,-1,1),  (1,-1,1,1),  (-1,1,1,1),  (-1,-1,-1,1), (-1,-1,1,-1),  (-1,1,-1,-1),  (1,-1,-1,-1),     (1,1,1,1), (-1,-1,-1,-1),   (1,-1,-1,1), (-1,1,1,-1)]
            colors =            ["crimson", "crimson",       "royalblue", "royalblue",       "limegreen", "limegreen", "limegreen", "limegreen", "limegreen", "limegreen", "limegreen", "limegreen",            "gold", "gold",            "purple", "purple"]
            zo =                [2,2,                        2,2,                            2,2,2,2,2,2,2,2,                                                                                                   3,3,                       2,2]
            for i, (s1,s2,s3,s4) in enumerate(spin_combinations):
                Veffbar = np.load(f"arrays/FS with  eta={eta}/Veffbar_matrices/{sta.get(s1)}{sta.get(s2)}{sta.get(s3)}{sta.get(s4)}.npy")[th0i,:]
                l, = ax.plot(th_array_, np.real(Veffbar), color=colors[i], lw=1.2, zorder=zo[i], \
                        label=r"${{{}{}{}{}}}$".format(sta.get(s1),sta.get(s2),sta.get(s3),sta.get(s4)))
                ax.set_ylabel(r"$N \mathfrak{Re}( \bar{V}^{\sigma_1\sigma_2\sigma_3\sigma_4}_{\mathbf{k}\mathbf{k}^\prime} ) $", fontsize=14)
                lines.append(l)
                if do_print:
                    print(f"|Vbar({sta.get(s1)}{sta.get(s2)}{sta.get(s3)}{sta.get(s4)})| = {np.round(Veffbar[0], 10)}")
        elif part_vals == "imag_vals":
            spin_combinations = [(1,1,-1,1), (-1,-1,-1,1),              (1,-1,1,1), (1,-1,-1,-1),      (1,-1,1,-1), (-1,1,-1,1), (-1,-1,-1,-1), (1,1,1,1), (1,1,-1,-1), (-1,-1,1,1), (1,-1,-1,1), (-1,1,1,-1),       (-1,1,1,1), (-1,1,-1,-1),        (1,1,1,-1), (-1,-1,1,-1)]
            colors =            ["forestgreen","forestgreen",           "sienna",    "sienna",         "violet",   "violet",    "violet",      "violet",   "violet",   "violet",    "violet",    "violet",          "olive",    "olive",     "mediumturquoise","mediumturquoise"]
            for i, (s1,s2,s3,s4) in enumerate(spin_combinations):
                Veffbar = np.load(f"arrays/FS with  eta={eta}/Veffbar_matrices/{sta.get(s1)}{sta.get(s2)}{sta.get(s3)}{sta.get(s4)}.npy")[th0i,:]
                l, = ax.plot(th_array_, np.imag(Veffbar), color=colors[i], lw=1.2, \
                        label=r"${{{}{}{}{}}}$".format(sta.get(s1),sta.get(s2),sta.get(s3),sta.get(s4)))
                ax.set_ylabel(r"$N \mathfrak{Im}( \bar{V}^{\sigma_1\sigma_2\sigma_3\sigma_4}_{\mathbf{k}\mathbf{k}^\prime} ) $", fontsize=14)
                lines.append(l)
                if do_print:
                    print(f"|Vbar({sta.get(s1)}{sta.get(s2)}{sta.get(s3)}{sta.get(s4)})| = {np.round(Veffbar[0], 10)}")
        else:
            raise Exception
    ### ^
        if part_vals=="real_vals":
            legend1 = plt.legend(handles=[lines[0], lines[1]],              loc="upper right",  bbox_to_anchor=(1.15,1.025), handletextpad=0.0, handlelength=0, framealpha=1, edgecolor=lines[1].get_color())#[lines[0].get_color(), lines[1].get_color()])
            legend2 = plt.legend(handles=[lines[i_] for i_ in range(2,4)],  loc="center right", bbox_to_anchor=(1.15,0.1925), handletextpad=0.0, handlelength=0, framealpha=1, edgecolor=lines[3].get_color())#, labelcolor=[lines[i_].get_color() for i_ in range(2,6)])
            legend3 = plt.legend(handles=[lines[i_] for i_ in range(4,12)],  loc="lower right",  bbox_to_anchor=(1.15,0.19+0.053), handletextpad=0.0, handlelength=0, framealpha=1, edgecolor=lines[11].get_color())#, labelcolor=[lines[i_].get_color() for i_ in range(6,8)])
            legend4 = plt.legend(handles=[lines[i_] for i_ in range(12,14)], loc="lower right",  bbox_to_anchor=(1.15,0.725), handletextpad=0.0, handlelength=0, framealpha=1, edgecolor=lines[13].get_color())#, labelcolor=[lines[i_].get_color() for i_ in range(8,16)])
            legend5 = plt.legend(handles=[lines[i_] for i_ in range(14,16)], loc="lower right",  bbox_to_anchor=(1.15,-0.025), handletextpad=0.0, handlelength=0, framealpha=1, edgecolor=lines[15].get_color())#, labelcolor=[lines[i_].get_color() for i_ in range(8,16)])
            ax.add_artist(legend1)
            ax.add_artist(legend2)
            ax.add_artist(legend3)
            ax.add_artist(legend4)
            ax.add_artist(legend5)
        elif part_vals=="imag_vals":
            legend1 = plt.legend(handles=[lines[0], lines[1]],              loc="upper right",  bbox_to_anchor=(1.15,1.025), handletextpad=0.0, handlelength=0, framealpha=1, edgecolor=lines[1].get_color())#[lines[0].get_color(), lines[1].get_color()])
            legend2 = plt.legend(handles=[lines[2], lines[3]],              loc="center right", bbox_to_anchor=(1.15,0.1925), handletextpad=0.0, handlelength=0, framealpha=1, edgecolor=lines[3].get_color())#, labelcolor=[lines[i_].get_color() for i_ in range(2,6)])
            legend3 = plt.legend(handles=[lines[i_] for i_ in range(4,12)], loc="lower right",  bbox_to_anchor=(1.15,0.244),         handletextpad=0.0, handlelength=0, framealpha=1, edgecolor=lines[11].get_color())#, labelcolor=[lines[i_].get_color() for i_ in range(6,8)])
            legend4 = plt.legend(handles=[lines[12], lines[13]],            loc="lower right",  bbox_to_anchor=(1.15,0.725), handletextpad=0.0, handlelength=0, framealpha=1, edgecolor=lines[13].get_color())#, labelcolor=[lines[i_].get_color() for i_ in range(8,16)])
            legend5 = plt.legend(handles=[lines[14], lines[15]],            loc="lower right",  bbox_to_anchor=(1.15,-0.025), handletextpad=0.0, handlelength=0, framealpha=1, edgecolor=lines[15].get_color())#, labelcolor=[lines[i_].get_color() for i_ in range(8,16)])
            ax.add_artist(legend1)
            ax.add_artist(legend2)
            ax.add_artist(legend3)
            ax.add_artist(legend4)
            ax.add_artist(legend5)
    if save:
        pts_save = {"abs_vals" : "abs", "real_vals" : "real", "imag_vals" : "imag"} ### part_vals to string dictionary
        type_save ={1 : "allLines", 0 : "colorMatched"}
        plt.savefig(f'plots svg/{pts_save.get(part_vals)}_{type_save.get(see_all_lines)}_Veffbar-plot-FS_eta{eta}__n{period}.svg', dpi=1000, format='svg', bbox_inches='tight')
        plt.savefig(f'plots jpg/{pts_save.get(part_vals)}_{type_save.get(see_all_lines)}_Veffbar-plot-FS_eta{eta}__n{period}.jpg', dpi=1000, format='jpg', bbox_inches='tight')

    annot = ax.annotate("", xy=(0, 0), xytext=(-20, 20), textcoords="offset points",
                        bbox=dict(boxstyle="round", fc="w"),
                        arrowprops=dict(arrowstyle="->"))
    annot.set_visible(False)
    annot_coords = ax.annotate("", xy=(0, 0), xytext=(-20, -30), textcoords="offset points",
                        bbox=dict(boxstyle="round", fc="w"),
                        arrowprops=dict(arrowstyle="->"))
    annot_coords.set_visible(False)

    h = lambda evnt : hover(evnt, fig,ax,annot,lines)
    oc_t = lambda evnt : onclick_theta(evnt, annot_coords,lines)
    fig.canvas.mpl_connect("motion_notify_event", h)
    fig.canvas.mpl_connect('button_press_event', oc_t)
    plt.show()

# %%
def plot_coupling_functions_FS(th_array_, th0i, eta, save=False, varykp=True, period=n):
    """ Plotting from loading Veffbar """
    varykp_to_str = {1 : r"for some $k$", 0 : r"for some $k'$"}
    varykp_to_theta_str = {1 : r"$\theta^\prime$", 0 : r"$\theta$"}

    fig, axs = plt.subplots(2,2, sharex=True, sharey=False, figsize=(8,7))
    axs = axs.flatten()
    if not save:
        plt.suptitle(fr"Pair potential on the Fermi sphere for $\mu/t={eta}$: {varykp_to_str.get(varykp)}", y=0.98)


    theta_tick_values = [th_array_[j] for j in range(0, len(th_array_), len(th_array_)//4)]
    theta_tick_labels = get_radian_ticks(theta_tick_values)
    plt.text(-11, 0.009+0.0265*bool(float(eta) == -5.43), r"$N\ \mathcal{V}[r,c]$  [eV]", rotation='vertical', horizontalalignment='left', fontsize=12)
    for idx_, axi in enumerate(axs):
        axi.set_xticks(theta_tick_values,   theta_tick_labels)
        if idx_ >=2:
            axi.set_xlabel(varykp_to_theta_str.get(varykp))
        axi.xaxis.set_minor_locator(MultipleLocator(np.pi/4))
        axi.set_xlim(th_array_[0],th_array_[-1])
        axi.tick_params(axis='both')
        axi.grid(c='gainsboro', which='both')
    
    colorsRe = ['blue', 'darkgreen', 'red', 'purple']
    colorsIm = ['deepskyblue', 'limegreen', 'darkorange', 'magenta']
    ds = [(5,0), (5,7), (5,15), (5,0)]

    Nu = np.load(f"arrays/FS with  eta={eta}/coupling matrix.npy")
    if varykp: Nu = Nu[:,:,th0i,:]
    else: Nu = Nu[:,:,:,th0i]
    all_lines = []
    annotations = []
    annotations_coords = []
    for r in range(4):
        dont_plot_cs = []
        lines=[]
        title_ = fr"$r={r+1}$"
        for c in range(4):
            if ((title_ == fr"$r={r+1}$") and (np.around(Nu[r,c,:],12) == 0).all()):
                title_ += ": " + r"$\mathcal{V}_{k'k}$" + r"$[{{{}}}, {{{}}}]$".format(r+1,c+1) + r"$=0$"
                dont_plot_cs += [c]
            elif (np.around(Nu[r,c,:],12) == 0).all():
                title_ += ", " + r"$\mathcal{V}_{k'k}$" + r"$[{{{}}}, {{{}}}]$".format(r+1,c+1) + r"$=0$"
                dont_plot_cs += [c]

            if c not in dont_plot_cs:
                l1, = axs[r].plot(th_array_, np.real(Nu[r,c,:]), color=colorsRe[c], lw=1, dashes=ds[c], \
                    label=r"$\mathfrak{Re}(\mathcal{V}_{k'k}$" + r"$[{}, {}])$".format(r+1,c+1))
                l2, = axs[r].plot(th_array_, np.imag(Nu[r,c,:]), color=colorsIm[c], lw=1, dashes=ds[c], \
                    label=r"$\mathfrak{Im}(\mathcal{V}_{k'k}$" + r"$[{}, {}])$".format(r+1,c+1))
                lines.append(l1)
                lines.append(l2)
        axs[r].set_title(title_, fontsize=10)
        annot = axs[r].annotate("", xy=(0, 0), xytext=(-20, 20), textcoords="offset points",
                    bbox=dict(boxstyle="round", fc="w"),
                    arrowprops=dict(arrowstyle="->"))
        annot.set_visible(False)
        annot_coords = axs[r].annotate("", xy=(0, 0), xytext=(-20, -30), textcoords="offset points",
                            bbox=dict(boxstyle="round", fc="w"),
                            arrowprops=dict(arrowstyle="->"))
        annot_coords.set_visible(False)
        all_lines.append(lines)
        annotations.append(annot)
        annotations_coords.append(annot_coords)
    axs[-1].legend(labels =[r"$\mathfrak{Re}(\mathcal{V}_{k'k}[r, 1])$",\
                            r"$\mathfrak{Im}(\mathcal{V}_{k'k}[r, 1])$",\
                            r"$\mathfrak{Re}(\mathcal{V}_{k'k}[r, 2])$",\
                            r"$\mathfrak{Im}(\mathcal{V}_{k'k}[r, 2])$",\
                            r"$\mathfrak{Re}(\mathcal{V}_{k'k}[r, 3])$",\
                            r"$\mathfrak{Im}(\mathcal{V}_{k'k}[r, 3])$",\
                            r"$\mathfrak{Re}(\mathcal{V}_{k'k}[r, 4])$",\
                            r"$\mathfrak{Im}(\mathcal{V}_{k'k}[r, 4])$" ], \
                   handles=[all_lines[0][0], all_lines[0][1], all_lines[0][2],\
                            all_lines[0][3], all_lines[0][4], all_lines[0][5],\
                            all_lines[-1][-2], all_lines[-1][-1]
                    ]
            ,loc='upper left', bbox_to_anchor=(-1.42,2.75), ncol=4)#, fontsize=9)

    plt.subplots_adjust(wspace=0.4, hspace=0.3, top=0.8)
    
    if save:
        varykp_to_str2 = {1:"varykp", 0:"varyk"}
        plt.savefig(f'plots svg/Coupling-functions-plot-FS_eta{eta}_{varykp_to_str2.get(varykp)}__n{period}.svg', dpi=1000, format='svg', bbox_inches='tight')
        plt.savefig(f'plots jpg/Coupling-functions-plot-FS_eta{eta}_{varykp_to_str2.get(varykp)}__n{period}.jpg', dpi=1000, format='jpg', bbox_inches='tight')

    h2 = lambda evnt : hover_multiple(evnt, fig,axs,annotations,all_lines)
    oc_t2 = lambda evnt : onclick_theta_multiple(evnt, annotations_coords,all_lines)
    fig.canvas.mpl_connect("motion_notify_event", h2)
    fig.canvas.mpl_connect('button_press_event', oc_t2)
    plt.show()



# %%

def plot_gaps(eta_, Delta_, th_array_, save_fig=False, period=n, nrows=n_chains):
    Nth = len(th_array_)

    D_max = np.amax([np.amax(np.abs(Delta_.imag)), np.amax(np.abs(Delta_.real))])

    fig, axs = plt.subplots(2,2, figsize=(8,7))
    axs = axs.flatten()

    if not save_fig:
        plt.suptitle(r"$\mu/t=$" + str(eta_))

    axs[0].set_title(r"$\Delta^{O(s)}_{\uparrow\downarrow}/\Delta_{\mathrm{max}}$")
    axs[0].plot(th_array_, Delta_[0:4*Nth:4].real/D_max, label="Re", c='blue',       lw=1)
    axs[0].plot(th_array_, Delta_[0:4*Nth:4].imag/D_max, label="Im", c='darkorange', lw=1)#, ls='--', dashes=(5,10))  ## 0
    axs[0].text(np.pi, 0.8, r"$s$-wave", horizontalalignment='center')

    axs[1].set_title(r"$\Delta_{\uparrow\uparrow}/\Delta_{\mathrm{max}}$")
    axs[1].plot(th_array_, Delta_[1:4*Nth:4].real/D_max, label=r"Re", c='blue',       lw=1) ## 0
    axs[1].plot(th_array_, Delta_[1:4*Nth:4].imag/D_max, label=r"Im", c='darkorange', lw=1)#, ls='--', dashes=(5,10))
    axs[1].text(np.pi, 0.08+0.15*bool(eta_==-5.43), r"$i p_x$-wave", horizontalalignment='center')

    axs[2].set_title(r"$\Delta_{\downarrow\downarrow}/\Delta_{\mathrm{max}}$")
    axs[2].plot(th_array_, Delta_[2:4*Nth:4].real/D_max, label=r"Re", c='blue',       lw=1) ## 0
    axs[2].plot(th_array_, Delta_[2:4*Nth:4].imag/D_max, label=r"Im", c='darkorange', lw=1)#, ls='--', dashes=(5,10))
    axs[2].text(np.pi, 0.08+0.15*bool(eta_==-5.43), r"$i p_x$-wave", horizontalalignment='center')

    axs[3].set_title(r"$\Delta^{E(s)}_{\uparrow\downarrow}/\Delta_{\mathrm{max}}$")
    axs[3].plot(th_array_, Delta_[3:4*Nth:4].real/D_max, label=r"Re", c='blue',       lw=1) ## 0
    axs[3].plot(th_array_, Delta_[3:4*Nth:4].imag/D_max, label=r"Im", c='darkorange', lw=1, ls='--', dashes=(5,10)) ## 0

    for ax in axs:
        ax.xaxis.set_minor_locator(MultipleLocator(np.pi/2))
        ax.grid(color='gainsboro', axis='both', which='both')
        ax.set_xticks([0,np.pi,2*np.pi])
        ax.set_xticklabels([r"$0$", r"$\pi$", r"$2\pi$"])
        ax.set_xlabel(r"$\theta$")
        ax.set_xlim(0,2*np.pi)
    plt.tight_layout()
    axs[0].legend(loc='upper right', ncol=2, bbox_to_anchor=(1.38,1.3))

    if save_fig:
        plt.savefig(f'plots jpg/gap-functions-plot-FS_eta{eta_}__n{period}.jpg', dpi=1000, format='jpg', bbox_inches='tight')
        plt.savefig(f'plots svg/gap-functions-plot-FS_eta{eta_}__n{period}.svg', dpi=1000, format='svg', bbox_inches='tight')

    plt.show()


# %%
def get_text_multiple(l):
    return l.get_label()

def update_annot_multiple(l,ind, text_arr, annot):
    x,y = l.get_data()
    annot.xy = (x[ind["ind"][0]], y[ind["ind"][0]])
    annot.set_text("\n".join(text_arr))
    annot.get_bbox_patch().set_alpha(0.4)

def hover_multiple(event, fig,axs,annotations,all_lines):
    for annot_idx, annot in enumerate(annotations):
        vis = annot.get_visible()
        if event.inaxes == axs[annot_idx]:
            text=[]
            for line in all_lines[annot_idx]:
                cont, ind = line.contains(event)
                if cont:
                    text.insert(0,get_text_multiple(line))
                    update_annot_multiple(line,ind,text, annot)
                    annot.set_visible(True)
                    fig.canvas.draw_idle()
                else:
                    if vis:
                        annot.set_visible(False)
                        fig.canvas.draw_idle()


def update_annot_coords_theta_multiple(l,ind, annot_coords):
    x,y = l.get_data()
    xi, yi = (x[ind["ind"][0]], y[ind["ind"][0]])
    annot_coords.xy = (xi, yi)
    xi_frac = str(Fraction(round(xi,10)/np.pi).limit_denominator())
    if "/" in xi_frac:
        num,denom = xi_frac.split('/')
        if num == '1':
            num = ""
        frac="/"
        xi_text = r"\frac{{{}}}{{{}}}".format(num+"\pi", denom)
    else:
        num = xi_frac
        if xi_frac == "1":
            num = ""
        frac=denom=""
        xi_text = f"{num}\pi"
        if xi_frac == "0":
            xi_text = f"{num}"
    annot_coords.set_text(r"$({}, {})$".format(xi_text, round(yi,10)))
    annot_coords.get_bbox_patch().set_alpha(0.4)    
    if num == "0":
        print(f'x = {num},   y = {round(yi,10)}                          ', end='\r')
    else:
        print(f'x = {num}\u03C0{frac}{denom} (={round(xi,10)}),   y = {round(yi,10)}                          ', end='\r')


def onclick_theta_multiple(event, annotations_coords,all_lines):
    for annot_c_idx, annot_coords in enumerate(annotations_coords):
        for line in all_lines[annot_c_idx]:
            cont, ind = line.contains(event)
            if cont:
                update_annot_coords_theta_multiple(line, ind, annot_coords)
                annot_coords.set_visible(True)


# %%
""" Plot mBZ and eBZ (and electron disperion) """

rot_mat = np.array([[0,-1], [1,0]])

def plot_WS(cell, helping_lines=True, rectangular=False, center=[0,0], cell_color='r', zo=2, lw=1, plot_center=False, ax=plt):
    global rot_mat

    for l_idx, l in enumerate(cell):
        if helping_lines:
            plt.arrow(0,0,  l[0], l[1], color='gray',zorder=0)
        w = l@rot_mat
        w = w/np.linalg.norm(w)
        if not rectangular:
            w *= np.linalg.norm(l)*np.tan(np.pi/6)
        else:
            w *= cell[l_idx%2+1]
        ax.plot([l[0]/2+center[0], l[0]/2+center[0]+w[0]/2], [l[1]/2+center[1], l[1]/2+center[1]+w[1]/2], color=cell_color, zorder=zo, lw=lw)
        ax.plot([l[0]/2+center[0], l[0]/2+center[0]-w[0]/2], [l[1]/2+center[1], l[1]/2+center[1]-w[1]/2], color=cell_color, zorder=zo, lw=lw)
        if plot_center:
            ax.scatter(center[0], center[1], alpha=0.1, c=cell_color, zorder=1)


def set_plot_settings(a,b, ax):
    tick_vals = np.arange(-3*np.pi/2, 3*np.pi/2+np.pi/2, np.pi/2)
    tick_labels=[r"$-\frac{3\pi}{2}$", r"$-\pi$", r"$-\frac{\pi}{2}$", r"$0$", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$"]
    ax.set_xlabel(r"$k_x,\ q_x$")
    ax.set_xticks(ticks=tick_vals, labels=tick_labels)
    ax.set_ylabel(r"$k_y,\ q_y$")
    ax.set_yticks(ticks=tick_vals, labels=tick_labels)
    ax.set_aspect('equal')
    ax.set_xlim(-2*a[0],2*a[0])
    ax.set_ylim(-2*b[1],2*b[1])


def get_mcell_ecell(period=n):
    if period != n:
        a1_magnon = period*a1_e
        a2_magnon = np.array([0, 2*a2_e[1]])
        a_mat_magnon = np.array([a1_magnon,a2_magnon])
        b_mat_magnon = 2*np.pi*np.linalg.inv(a_mat_magnon).T
        [b1_magnon,b2_magnon] = b_mat_magnon
    else:
        b1_magnon = globals()["b1_magnon"]
        b2_magnon = globals()["b2_magnon"]
    return np.vstack([[0,0], b2_magnon, b1_magnon, -b2_magnon, -b1_magnon]),\
           np.vstack([[0,0], b1_electron, -b1_electron, b2_electron, -b2_electron, b1_electron+b2_electron, -b1_electron-b2_electron])


def mBZ_filling_eBZ(ax, period=n):
    reciprocal_cell_magnon, reciprocal_cell_electron = get_mcell_ecell(period)
    for i in range(-5,6):
        for j in range(-2,3):
            if [i,j] in [[c,0] for c in range(-3,4)] + [[cc,1] for cc in range(-1,2)]:
                c_m = 'g'
                zo_m = 1
            else:
                c_m = 'r'
                zo_m = 0
            plot_WS(reciprocal_cell_magnon, helping_lines=False, rectangular=True, center=b1_magnon*i+b2_magnon*j, cell_color=c_m, zo=zo_m, plot_center=True)
            plot_WS(reciprocal_cell_electron, helping_lines=False, rectangular=False, cell_color='k', lw=2, center=b1_electron*i+b2_electron*j)
    set_plot_settings(2*b1_electron/5,2*b2_electron/5, ax)

# %%

def plot_mBZs_filling_eBZ1(period=n, save=False):
    plt.figure(figsize=(6,6))
    mBZ_filling_eBZ(ax=plt.gca())
    if save:
        plt.savefig(f"plots jpg/mBZ filling eBZ -- n{n}.jpg", dpi=1000, format='jpg', bbox_inches='tight')
        plt.savefig(f"plots svg/mBZ filling eBZ -- n{n}.svg", dpi=1000, format='svg', bbox_inches='tight')
    plt.show()

# %%
""" Plot electron dispersion """

def create_electron_dispersion(etas=[-5.9, -5.43], period=n, save_fig=False, recalculate=False):
    try:
        k_arrays = [np.load(f"arrays/FS with  eta={etas[0]}/kp_array (+k').npy"),\
                    np.load(f"arrays/FS with  eta={etas[1]}/kp_array (+k').npy")]
        if recalculate: raise Exception
    except:
        th_array_tmp = np.load("arrays/th_array.npy")
        k_arrays = [np.vectorize(get_kx_ky, excluded=["eta_", "a"])(th_array_tmp, eta_=etas[0], a=1),\
                    np.vectorize(get_kx_ky, excluded=["eta_", "a"])(th_array_tmp, eta_=etas[1], a=1)]

    xlim = 2*b1_electron[0]/3+0.014
    ylim = 2*b2_electron[1]/3*0.758
    kx = np.linspace(-xlim, xlim, 100)
    ky = np.linspace(-ylim, ylim, 100)
    KX, KY = np.meshgrid(kx, ky)
    EDRs = []
    for eta_ in etas:
        EDRs += [eDR_np(KX, KY, eta=eta_, a=1)]
    zs = np.concatenate(EDRs, axis=0)
    min_, max_ = zs.min(), zs.max()

    plt.figure()
    cm = plt.cm.get_cmap('viridis')
    reciprocal_cell_magnon, reciprocal_cell_electron = get_mcell_ecell(period)
    plot_WS(reciprocal_cell_electron, helping_lines=False, lw=3, cell_color='k')
    plot_WS(reciprocal_cell_magnon, rectangular=True, helping_lines=False, lw=2, cell_color='r')


    sc0 = plt.scatter(KX, KY, c=EDRs[0], cmap=cm)
    plt.clim(min_, max_)
    plt.scatter(KX, KY, c=EDRs[1], cmap=cm)
    plt.clim(min_, max_)

    plt.plot(k_arrays[0][:,0],  k_arrays[0][:,1], label=fr"$\eta={etas[0]}$", color='white', ls='-')
    plt.plot(k_arrays[1][:,0],  k_arrays[1][:,1], label=fr"$\eta={etas[1]}$", color='white', ls='--')

    set_plot_settings(b1_electron/3+0.014, b2_electron/3*0.758, ax=plt.gca())

    plt.text(-np.pi/2-0.4, np.pi, "eBZ", c='k', fontsize=20, fontname='Cambria')
    plt.text(-b1_magnon[0]/2, np.pi*5/8, "mBZ", c='r', fontsize=20, fontname='Cambria')
    plt.text(np.pi/3, -0.1, "FS", c='white', fontsize=20, fontname='Cambria')


    cax0 = plt.axes([1, 0.1, 0.075, 0.8])
    cax0.text(0.13,9.1, r"$\tilde{\epsilon}_k/t$", fontsize=15)
    cax1 = cax0.twinx()

    cb = plt.colorbar(sc0, cax=cax0, label=fr"$\mu/t = {etas[0]}$")
    cb.ax.yaxis.set_ticks_position('left')
    cb.ax.yaxis.set_label_position('left')

    cax0.set_ylim(EDRs[0].min(), EDRs[0].max())
    cax1.set_ylim(EDRs[1].min(), EDRs[1].max())
    cax1.set_ylabel(fr'$\mu/t = {etas[1]}$')

    if save_fig:
        plt.savefig(f"plots svg/electron energy with eBZ, mBZ and FSs -- n{period}.svg", dpi=1000, format='svg', bbox_inches='tight')
        plt.savefig(f"plots jpg/electron energy with eBZ, mBZ and FSs -- n{period}.jpg", dpi=1000, format='jpg', bbox_inches='tight')
    plt.show()


def create_Umklapp_electron_dispersion(th0i, etas=[-5.9, -5.43], period=n, save_fig=False, recalculate=False):
    try:
        k_arrays = [np.load(f"arrays/FS with  eta={etas[0]}/kp_array (+k').npy"),\
                    np.load(f"arrays/FS with  eta={etas[1]}/kp_array (+k').npy")]
        if recalculate: raise Exception
    except:
        th_array_tmp = np.load("arrays/th_array.npy")
        k_arrays = [np.vectorize(get_kx_ky, excluded=["eta_", "a"])(th_array_tmp, eta_=etas[0], a=1),\
                    np.vectorize(get_kx_ky, excluded=["eta_", "a"])(th_array_tmp, eta_=etas[1], a=1)]

    xlim = 2*b1_electron[0]/3+0.014
    ylim = 2*b2_electron[1]/3*0.758
    kx = np.linspace(-xlim, xlim, 100)
    ky = np.linspace(-ylim, ylim, 100)
    KX, KY = np.meshgrid(kx, ky)
    EDRs = []
    for eta_ in etas:
        EDRs += [eDR_np(KX, KY, eta=eta_, a=1)]
    zs = np.concatenate(EDRs, axis=0)
    min_, max_ = zs.min(), zs.max()

    linestyles = ['-', '-']
    scatter_pts = [-48, 15]
    scatter_colors = ['pink', 'orange']
    scatter_text = [r"$\mathbf{k}+\mathbf{q}$", r"$\mathbf{k}+\mathbf{q}+\mathbf{Q_q}$"]
    scatter_pad = [[-0.8, -0.46], [-0.3, 0.3]]


    fig, axes = plt.subplots(1,2, figsize=(10,3.8), sharey=True)
    axes = axes.flatten()
    ax0, ax1 = axes

    cm = plt.cm.get_cmap('viridis')
    reciprocal_cell_magnon, reciprocal_cell_electron = get_mcell_ecell(period)
    for idx, axi in enumerate(axes):
        plot_WS(reciprocal_cell_electron, helping_lines=False, lw=2, cell_color='k', ax=axi)
        plot_WS(reciprocal_cell_magnon, rectangular=True, helping_lines=False, lw=1.6, cell_color='r', center=[k_arrays[idx][th0i][0], k_arrays[idx][th0i][1]], ax=axi)
        sci = axi.scatter(KX, KY, c=EDRs[idx], cmap=cm)
        axi.plot(k_arrays[idx][:,0],  k_arrays[idx][:,1], label=fr"$\eta={etas[idx]}$", color='white', ls=linestyles[idx])
        axi.scatter(k_arrays[idx][th0i][0], k_arrays[idx][th0i][1], c='white')
        axi.text(k_arrays[idx][th0i][0] - 0.44, k_arrays[idx][th0i][1]-0.00, r"$\mathbf{k}$", c='white', fontsize=12, fontname='Cambria')

        set_plot_settings(b1_electron/3+0.014, b2_electron/3*0.758, ax=axi)
        axi.text(-np.pi/2-0.4,                             np.pi-0.1, "eBZ", c='k',     fontsize=18, fontname='Cambria')
        axi.text(k_arrays[idx][th0i][0]-b1_magnon[0]*1/2,  np.pi*5/8, "mBZ", c='r',     fontsize=18, fontname='Cambria')
        axi.text(k_arrays[idx][-1][0]+0.3/(idx+1),         -0.15,     "FS",  c='white', fontsize=18, fontname='Cambria')
        fig.colorbar(sci, ax=axi, shrink=0.8, aspect=10)
        axi.text(4.7,3.5, r"$\tilde{\epsilon}_k/t$", fontsize=13)
        axi.set_title(fr"$\mu/t = {etas[idx]}$", fontsize=15)
        axi.set_xlabel(r"$k_x$")
        axi.set_ylabel(r"$k_y$")
    ###
    for idx, scp in enumerate(scatter_pts):
        ax1.scatter(k_arrays[1][scp][0], k_arrays[1][scp][1], c=scatter_colors[idx], zorder=3)
        ax1.text(k_arrays[1][scp][0] + scatter_pad[idx][0], k_arrays[1][scp][1] + scatter_pad[idx][1], scatter_text[idx], c=scatter_colors[idx], fontsize=12, fontname='Cambria')
    ### 
    plt.tight_layout()
    if save_fig:
        plt.savefig(f"plots jpg/electron energy with scattering - eBZ, mBZ and FSs -- n{period}.jpg", dpi=1000, format='jpg', bbox_inches='tight')
        plt.savefig(f"plots svg/electron energy with scattering - eBZ, mBZ and FSs -- n{period}.svg", dpi=1000, format='svg', bbox_inches='tight')
    plt.show()
