# -*- coding: utf-8 -*-
"""
Created on Thu Dec 30 15:35:20 2021

@author: Arjun Chakrawal
arjun.chakrawal@natgeo.su.se
chakrawalarjun9105@gmail.com
https://twitter.com/ArjunChakrawal
"""
# %%
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from scipy import optimize
import warnings

warnings.filterwarnings("ignore")

# %%

plt.style.use("ggplot")

# %%
if not os.path.exists("fig"):
    os.makedir("fig/png")
    os.makedir("fig/svg")
    


# %% [markdown]
# ### function to estimate efficiency

# %%
def dGox_func(gamma):
    return 60.3 - 28.5 * (4 - gamma)


def dCG_O2(gamma):
    return dGox_func(gamma) + (gamma / gamma_O2) * dGred_O2


def efficiency_CNED_Inf(gamma, T, eA, CNB, NSource, CNED):

    if gamma > 8:
        print("Error. wrong values of DR of S of P")
        return

    # Ts = 298
    # pre-assign std G of formation from CHNOSZ
    dfGH2O = -237.2  # kJ/mol
    dfGO2aq = 16.5  # kJ/mol
    dfGNH4 = -79.5  # kJ/mol
    # dfGHCO3aq = -586.9 #kJ/mol
    # dfGglu = -919.8 #kJ/mol
    # dfGbio = -67 #kJ/mol
    # dfGeth = -181.8 #kJ/mol
    dfGNO3 = -111  # kJ/mol
    # dfGNO2 = -32.2 #kJ/mol
    # dfGNO = 102 #kJ/mol
    # dfGN2H4 = 159.17 #kJ/mol
    # dfGNH2OH = -43.6 #kJ/mol
    dfGN2 = 18.2  # kJ/mol
    # dfGH2 = 17.7
    gamma_B = 4.2  # e- mol/Cmol

    # define epectron acceptor and some other DR
    if eA == "O2":
        gamma_eA = 4
        dGred_eA = 2 * dfGH2O - dfGO2aq
    elif eA == "NO3":
        gamma_eA = 8
        dGred_eA = dfGNH4 + 3 * dfGH2O - dfGNO3
    elif eA == "Denitrification":
        gamma_eA = 5
        dGred_eA = 0.5 * dfGN2 + 3 * dfGH2O - dfGNO3
    elif eA == "Fe(III)":  # ferrihydrite
        # dGred_eA values are taken from La Rowe 2015
        gamma_eA = 1
        dGred_eA = -100 * gamma_eA
    elif eA == "FeOOH":  # goethite
        gamma_eA = 1
        dGred_eA = -75.58 * gamma_eA
    elif eA == "Mn4":  # goethite
        gamma_eA = 2
        dGred_eA = -120.03 * gamma_eA
    elif eA == "SO4":  # goethite
        gamma_eA = 8
        dGred_eA = -24.04 * gamma_eA

    # define NSource
    if NSource == "NO3":
        gamma_NS = 8
        dGred_NS = dfGNH4 + 3 * dfGH2O - dfGNO3
    elif NSource == "NH4":
        # 0.5N2 + 4e + 5H ->  NH4 + 0.5H2
        gamma_NS = 0
        dGred_NS = 0
        # dGred_NS values are taken from La Rowe 2015

    # growth yield calculations
    if gamma < 4.67:
        dGrX = -(666.7 / gamma + 243.1)  # kJ/Cmol biomass
    else:
        dGrX = -(157 * gamma - 339)  # kJ/Cmol biomass

    # Anbolic reaction
    dCGX = dCG_O2(gamma_B)
    dGana = (gamma_B / gamma) * dCG_O2(gamma) - dCGX
    dGana1 = dGana

    dG_ox = dGox_func(gamma)
    if gamma_NS == 0:  # NSource = NH4
        dGcat = dG_ox + gamma * dGred_eA / gamma_eA
        Y = dGcat / (dGrX - dGana1 + gamma_B / gamma * dGcat)
        xN_cat = 0
        xEA_cat = (gamma) / gamma_eA
        dGrS = dGcat
    else:

        def xN(y1):
            return (y1 / CNB - 1 / CNED) * (1 - y1 * gamma_B / gamma) ** -1

        def dGcat(y2):
            return (
                dG_ox
                + dGred_NS * xN(y2)
                + dGred_eA * (gamma - gamma_NS * xN(y2)) / gamma_eA
            )

        def fun(y3):
            return y3 - dGcat(y3) / (dGrX - dGana1 + gamma_B / gamma * dGcat(y3))

        sol = optimize.root_scalar(fun, bracket=[0, 3], method="brentq")
        Y = sol.root
        xN_cat = xN(Y)
        xEA_cat = (gamma - gamma_NS * xN_cat) / gamma_eA
        TER = CNB / Y
        dGrS = dGcat(Y)
        if CNED < TER:  # true means N rich ED
            xN_cat = 0
            xEA_cat = gamma / gamma_eA
            dGcatNoNS = dG_ox + dGred_eA * gamma / gamma_eA
            Y = dGcatNoNS / (dGrX - dGana1 + gamma_B / gamma * dGcatNoNS)
            if dGcatNoNS > 0:
                print("Error. dGcat POSITIVE")
                return
            dGrS = dGcatNoNS
        else:
            if dGcat(Y) > 0:
                print("Error. dGcat POSITIVE")
                return
    v_EA = (1 - Y * gamma_B / gamma) * xEA_cat
    v_N = Y / CNB - 1 / CNED
    YCO2 = 1 - (Y)

    if xEA_cat < 0:
        print("eTransferEA <0")
        return

    return [Y, YCO2, dGrS, xN_cat, xEA_cat, v_EA, v_N]


# %%
T = 273 + 25
dfGH2O = -237.2  # kJ/mol
dfGO2aq = 16.5  # kJ/mol
# DeltaG of complete oxidation of organic matter
gamma_O2 = 4
dGred_O2 = 2 * dfGH2O - dfGO2aq
eA = ["O2", "NO3", "Fe(III)", "FeOOH", "SO4"]  # Electron acceptors
gammaB = 4.2  # Degree of reduction of biomass
xycolor = np.array([1, 1, 1]) * 0.2
lstyle = ["-", "--", "-.", "."]
labelfont = 16
# backcolor = [0.9412, 0.9412, 0.9412]
backcolor = np.array([1, 1, 1]) * 0.975
GridCol = np.array([1, 1, 1]) * 0.9
# backcolor=[1 1 1]
axisfont = 12
LC = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]

xycol = np.array([1, 1, 1]) * 0.4
gamma_B = 4.2
CNB = 5  # CN ratio of biomass
INORM = (
    np.arange(0.001, 0.1 + 0.0005, 0.0005) / 0.5
)  # normalized inorganic N supply rate
delta_psi = 120 * 0.001  # volt # Membrane potential
R = 8.314  # J/ mol K , Gas constant
FaradayC = 96.48534  # Coulomb per mol e or kJ/ volt, faraday constant

# %% [markdown]
# # Figure 2 matplotllib

# %%
gamma = np.arange(0.5, 8 + 0.05, 0.05)
dGred = -np.array([5, 24.04, 40, 75.6, 85, 122.7])
# dGred = -np.array([122])
dGrX = np.zeros((len(gamma)))
dCGX = dCG_O2(gamma_B)
dGB_ana = (gamma_B / gamma) * dCG_O2(gamma) - dCGX
# viridis = plt.cm.get_cmap('viridis', len(dGred))
# LC = viridis.colors

for i in range(0, len(gamma)):
    if gamma[i] < 4.67:
        dGrX[i] = -(666.7 / gamma[i] + 243.1)  # kJ/Cmol biomass
    else:
        dGrX[i] = -(157 * gamma[i] - 339)  # kJ/Cmol biomass
fig = plt.figure(figsize=(5.5, 4.5), facecolor="w")
axs = fig.subplots(nrows=1, ncols=1)

for i in range(0, len(dGred)):
    dGcat = (60.3 - 28.5 * (4 - gamma)) + gamma * dGred[i]
    Y = dGcat / (dGrX - dGB_ana + (4.2 / gamma) * dGcat)
    Ycat = 1 - (Y * 4.2 / gamma)
    # nTh = (Y * dGB_ana) / (-Ycat * dGcat)
    # nTh = dGB_ana / (dGrX - dGB_ana)
    nTh = dCGX / (dGcat / Y)
    axs.plot(gamma, Y, label=" " + str(-dGred[i]), linewidth=3.0, color=LC[i])
# axs.set_ylim([0, 0.65])
axs.set_ylim(bottom=0)
# axs.set_xlim([0, 3])


axs.tick_params(axis="x", labelsize=axisfont)
axs.tick_params(axis="y", labelsize=axisfont)
axs.set_xlabel(r"$\gamma_{ED}$", fontsize=labelfont)

axs.legend(
    loc="upper left",
    bbox_to_anchor=(-0.125, 1.3),
    frameon=False,
    fontsize=labelfont - 4,
    title=r"$ \frac{-\Delta_{red}G_{EA}}{\gamma_{EA}}$ kJ $\mathrm{(e^- mol)^{-1}}$",
    title_fontsize=labelfont,
)

# axs.text(2.3, 0.6175, r"($\mathrm{{SO_4}^{2-}}$)", fontsize=labelfont - 4)
# axs.text(2.2, 0.5, r"(goethite)", fontsize=labelfont - 4)
# axs.text(2.3, 0.385, r"($\mathrm{O_2}$)", fontsize=labelfont - 4)


axs.set_ylabel(r"$G_{norm} = e$", fontsize=labelfont)

fig.tight_layout()
fig.savefig("fig/png/Figure2.png", dpi=300)
fig.savefig("fig/svg/Figure2.svg", dpi=300)
plt.show()


# %% [markdown]
# ## Figure3 Aerobic condition Nitrate and ammonium as N source

# %%

CNED = np.inf

gamma_S = [3, 4, 5]
# LC = linspecer(length(gamma_S))
NS = ["NH4", "NO3"]

fig = plt.figure(figsize=(6, 4), facecolor="w")
ax = plt.axes()
plt.style.use("ggplot")
for m in range(0, len(NS)):
    for j in range(0, len(gamma_S)):
        emax_S, YCO2, dGrS, xN, xEA, v_EA, v_N = efficiency_CNED_Inf(
            gamma_S[j], T, eA[0], CNB, NS[m], CNED
        )
        Gnorm = np.minimum(emax_S * np.ones((len(INORM)),), (emax_S / v_N) * INORM)
        plt.plot(
            INORM,
            Gnorm,
            lstyle[m],
            linewidth=3,
            color=LC[j],
            label=r"{$\gamma_{ED}}$ = " + str(gamma_S[j]),
        )
# ax.set(xlabel = r'${I_{norm}}$', ylabel=r'$G_{norm}}$')

plt.ylabel(r"${G_{norm}}$", fontsize=labelfont)
plt.xlabel(r"${I_{norm}}$", fontsize=labelfont)

lstr = []
for i in gamma_S:
    lstr.append(r"$\gamma_{ED} =" + str(i) + "$")
leg1 = ax.legend(lstr, loc="best", frameon=False, fontsize=labelfont - 4)
ax.text(0.7, 1.05, "C limited", transform=ax.transAxes, fontsize=labelfont)
ax.text(0.1, 0.35, "N limited", transform=ax.transAxes, rotation=45, fontsize=labelfont)
ax.xaxis.set_major_formatter(lambda x, pos: "%.2f" % x)

(p1,) = ax.plot(np.nan, np.nan, linewidth=3, linestyle=lstyle[0], color="k")
(p2,) = ax.plot(np.nan, np.nan, linewidth=3, linestyle=lstyle[1], color="k")

ax.legend(
    [p1, p2],
    [r"$\mathrm{NH_4^+}$", r"$\mathrm{NO_3^-}$"],
    loc="lower right",
    frameon=False,
    fontsize=labelfont - 4,
    title="Inorg. N source",
    title_fontsize=labelfont - 2,
)
ax.add_artist(leg1)

fig.tight_layout()
fig.savefig("fig/png/Figure3.png", dpi=300)
fig.savefig("fig/svg/Figure3.svg", dpi=300)
plt.show()


# %% [markdown]
# # Figure 4

# %%

fig = plt.figure(figsize=(14, 6), facecolor="w")
axs = fig.subplots(nrows=1, ncols=2)

INORM = [0.01, 0.1, 0.2, 0.3, 0.5, 1.5]
# INORM = [0.01]
gamma_S = np.arange(0.05, 8, 0.1)

NS = "NO3"
eA = ["NO3", "Denitrification"]
vN = np.zeros((len(gamma_S), len(eA)))
e = np.zeros((len(gamma_S), len(eA)))

for i in range(0, len(INORM)):
    Gnorm = np.zeros((len(gamma_S), len(eA)))
    for m in range(0, len(eA)):
        for j in range(0, len(gamma_S)):
            emax_S, YCO2, dGrS, xN, xEA, v_EA, v_N = efficiency_CNED_Inf(
                gamma_S[j], T, eA[m], CNB, NS, CNED
            )
            vN_met = (xN + xEA) * (1 - emax_S * gammaB / gamma_S[j])

            Gnorm[j, m] = np.minimum(emax_S, (emax_S / (vN_met) * INORM[i]))
            e[j, m] = emax_S
            vN[j, m] = vN_met
    # fig = plt.figure(figsize=(14, 6), facecolor='w')
    # axs = fig.subplots(nrows=1, ncols=2)
    axs[0].plot(gamma_S, Gnorm[:, 0], lstyle[0], linewidth=3, color=LC[i])
    axs[0].plot(gamma_S, Gnorm[:, 1], lstyle[1], linewidth=3, color=LC[i])

    GN = (e[:, 0] / vN[:, 0]) * INORM[i]
    GC = e[:, 0]
    id = GC < GN
    idGN = sum(id)
    axs[1].plot(
        gamma_S[0:idGN], GN[0:idGN], lstyle[0], linewidth=3, color=LC[i], alpha=0.3
    )
    axs[1].plot(gamma_S[idGN:], GN[idGN:], lstyle[0], linewidth=3, color=LC[i])

    GN = (e[:, 1] / vN[:, 1]) * INORM[i]
    GC = e[:, 1]
    id = GC < GN
    idGN = sum(id)
    axs[1].plot(
        gamma_S[0:idGN], GN[0:idGN], lstyle[1], linewidth=3, color=LC[i], alpha=0.3
    )
    axs[1].plot(gamma_S[idGN:], GN[idGN:], lstyle[1], linewidth=3, color=LC[i])
axs[1].plot(gamma_S, e[:, 0], lstyle[0], linewidth=3, color="k")
axs[1].plot(gamma_S, e[:, 1], lstyle[1], linewidth=3, color="k")


for i in axs:
    i.tick_params(axis="x", labelsize=axisfont + 4)
    i.tick_params(axis="y", labelsize=axisfont + 4)
lstr = []
lhlist = []
for i in range(0, len(INORM)):
    (p1,) = axs[0].plot(np.nan, np.nan, linestyle="-", linewidth=3, color=LC[i])
    lstr.append(str(INORM[i]))
    lhlist.append(p1)
leg1 = axs[0].legend(
    lhlist,
    lstr,
    loc="lower left",
    frameon=False,
    fontsize=labelfont - 2,
    title=r"$I_{norm}$",
    title_fontsize=labelfont,
    bbox_to_anchor=(0, 0.2),
    ncol=1,
)

(p1,) = axs[0].plot(np.nan, np.nan, linestyle=lstyle[0], color="k", linewidth=3)
(p2,) = axs[0].plot(np.nan, np.nan, linestyle=lstyle[1], color="k", linewidth=3)

leg2 = axs[0].legend(
    [p1, p2],
    [r"$\mathrm{NO_3^- → NH_4^+}$", r"$\mathrm{NO_3^- → N_2}$"],
    loc="upper left",
    frameon=False,
    fontsize=labelfont - 2,
)

axs[0].add_artist(leg1)

# axs[1].text('Black lines = $G_{C, norm}$ and Colored lines = $G_{N, norm}$', fontsize=labelfont)

axs[0].set_ylabel(r"${G_{norm}}$", fontsize=labelfont + 6)
axs[0].set_xlabel(r"${\gamma_{ED}}$", fontsize=labelfont + 6)
axs[1].set_xlabel(r"${\gamma_{ED}}$", fontsize=labelfont + 6)

axs[0].text(0.0, 1.05, "(A)", transform=axs[0].transAxes, fontsize=labelfont)
axs[1].text(0.0, 1.05, "(B)", transform=axs[1].transAxes, fontsize=labelfont)
fig.tight_layout()
fig.savefig("fig/png/Figure4.png", dpi=300)
fig.savefig("fig/svg/Figure4.svg", dpi=300)
plt.show()


# %% [markdown]
# #   Figure 4 zoomed out

# %%
INORM = [0.01]
gamma_S = np.arange(0.05, 8, 0.1)

NS = "NO3"
eA = ["NO3", "Denitrification"]
Gnorm = np.zeros((len(gamma_S), len(eA)))
vN = np.zeros((len(gamma_S), len(eA)))
e = np.zeros((len(gamma_S), len(eA)))

i = 0

for m in range(0, len(eA)):
    for j in range(0, len(gamma_S)):
        emax_S, YCO2, dGrS, xN, xEA, v_EA, v_N = efficiency_CNED_Inf(
            gamma_S[j], T, eA[m], CNB, NS, CNED
        )
        vN_met = (xN + xEA) * (1 - emax_S * gammaB / gamma_S[j])

        Gnorm[j, m] = np.minimum(emax_S, (emax_S / (vN_met) * INORM[i]))
        e[j, m] = emax_S
        vN[j, m] = vN_met
fig = plt.figure(figsize=(14, 6), facecolor="w")
axs = fig.subplots(nrows=1, ncols=2)
axs[0].plot(gamma_S, Gnorm[:, 0], lstyle[0], linewidth=3, color=LC[i])
axs[0].plot(gamma_S, Gnorm[:, 1], lstyle[1], linewidth=3, color=LC[i])

axs[1].plot(gamma_S, e[:, 0], lstyle[0], linewidth=3, color="k", alpha=0.3)
axs[1].plot(gamma_S, e[:, 1], lstyle[1], linewidth=3, color="k", alpha=0.3)

axs[0].set_ylabel(r"${G_{norm}}$", fontsize=labelfont + 6)
axs[0].set_xlabel(r"${\gamma_{ED}}$", fontsize=labelfont + 6)
axs[1].set_xlabel(r"${\gamma_{ED}}$", fontsize=labelfont + 6)
axs[1].set_ylabel(r"$G_{C, norm}$", fontsize=labelfont + 6)

ax2 = axs[1].twinx()  # instantiate a second axes that shares the same x-axis
ax2.plot(gamma_S, (e[:, 0] / vN[:, 0]) * INORM[i], lstyle[0], linewidth=3, color=LC[i])
ax2.plot(gamma_S, (e[:, 1] / vN[:, 1]) * INORM[i], lstyle[1], linewidth=3, color=LC[i])
# we already handled the x-label with ax1
ax2.set_ylabel(r"$G_{N, norm}$", color=LC[i], fontsize=labelfont + 6)
ax2.tick_params(axis="y", labelcolor=LC[i], labelsize=axisfont + 4)


for i in axs:
    i.tick_params(axis="x", labelsize=axisfont + 4)
    i.tick_params(axis="y", labelsize=axisfont + 4)
lstr = []
for i in INORM:
    lstr.append(str(i))
leg1 = axs[0].legend(
    lstr,
    loc="lower center",
    frameon=False,
    fontsize=labelfont - 2,
    title=r"$I_{norm}$",
    title_fontsize=labelfont,
)


(p1,) = axs[0].plot(np.nan, np.nan, linestyle=lstyle[0], color="k", linewidth=3)
(p2,) = axs[0].plot(np.nan, np.nan, linestyle=lstyle[1], color="k", linewidth=3)

leg2 = axs[0].legend(
    [p1, p2],
    [r"$\mathrm{NO_3^- → NH_4^+}$", r"$\mathrm{NO_3^- → N_2}$"],
    loc="upper right",
    frameon=False,
    fontsize=labelfont - 2,
)
axs[0].add_artist(leg1)

axs[0].text(0.0, 1.05, "(A)", transform=axs[0].transAxes, fontsize=labelfont)
axs[1].text(0.0, 1.05, "(B)", transform=axs[1].transAxes, fontsize=labelfont)
fig.tight_layout()
fig.savefig("fig/png/FigureA1.png", dpi=300)
fig.savefig("fig/svg/FigureA1.svg", dpi=300)
plt.show()


# %% [markdown]
# # Figure 5

# %%

INORM = [0.01, 0.001]
eA = "O2"
CNS = np.arange(1, 20.1, 0.1)
CNS = np.append(CNS, np.arange(21, 1001, 1))

gamma_S = np.arange(1, 7, 1)
NS = "NH4"
fig = plt.figure(figsize=(7, 5), facecolor="w")
ax = plt.axes()
for j in range(0, len(INORM)):
    Gnorm = np.zeros((len(CNS), 1))
    for m in range(0, len(gamma_S)):
        for i in range(0, len(CNS)):
            [emax_S, YCO2, dGrS, xN, xEA, v_EA, v_N] = efficiency_CNED_Inf(
                gamma_S[m], T, eA, CNB, NS, CNS[i]
            )
            if v_N < 0:  # NH4 mineralization occurs
                Gnorm[i] = emax_S
            else:
                Gnorm[i] = np.minimum(emax_S, (emax_S / v_N) * INORM[j])
        ax.plot(CNS, Gnorm, linestyle=lstyle[j], linewidth=3, color=LC[m])
plt.ylabel(r"${G_{norm}}$", fontsize=labelfont)
plt.xlabel(r"${CN_{ED}}$", fontsize=labelfont)
plt.ylim([0.005, 1])
plt.yscale("log")
plt.xscale("log")
lhlist = []
lstr = []
for j in gamma_S:
    lstr.append(str(j))
leg1 = ax.legend(lstr, loc="lower left", frameon=False, title=r"$\gamma_{ED}}$")

lstr = []
for i in range(0, len(INORM)):
    (p1,) = ax.plot(np.nan, np.nan, linestyle=lstyle[i], linewidth=3, color="k")
    lstr.append(r"$I_{norm} =" + str(INORM[i]) + "$")
    lhlist.append(p1)
ax.legend(lhlist, lstr, loc="upper right", frameon=False)
ax.add_artist(leg1)
ax.grid(True, which="both")
fig.savefig("fig/png/Figure5.png", dpi=300)
fig.savefig("fig/svg/Figure5.svg", dpi=300)
plt.show()

# %% [markdown]
# # Figure 6

# %%
eA = ["O2", "Fe(III)", "FeOOH", "SO4"]
# eA = ['O2']

String_eA = [r"$\mathrm{O_2}$", "ferrihydrite", "goethite", r"$\mathrm{SO_4^{2-}}$"]
color_map = plt.cm.get_cmap("Spectral")
color_map = color_map.reversed()
gamma_S = np.linspace(1, 8, 71)

INORM = 0.01

fig = plt.figure(figsize=(11, 7), facecolor="w")
axs = fig.subplots(nrows=2, ncols=2, sharey="row", sharex="col")
axs = axs.reshape((2 * 2, 1))

NS = "NH4"
# CNS=[10000]
for k in range(0, len(eA) - 1):
    Gnorm = np.zeros((len(CNS), len(gamma_S)))
    e = np.zeros((len(CNS), len(gamma_S)))
    for m in range(0, len(gamma_S)):
        for i in range(0, len(CNS)):
            [emax_S, YCO2, dGrS, xN, xEA, v_EA, v_N] = efficiency_CNED_Inf(
                gamma_S[m], T, eA[k], CNB, NS, CNS[i]
            )
            e[i, m] = emax_S
            if emax_S < 0:
                print("negative CUE")
            else:
                if v_N < 0:  # NH4 mineralization occurs
                    Gnorm[i, m] = emax_S
                else:
                    Gnorm[i, m] = np.minimum(emax_S, (emax_S / v_N) * INORM)
    [gY, CNX] = np.meshgrid(gamma_S, CNS)
    cl = np.arange(0.05, 0.6 + 0.05, 0.05)
    cs = axs[k][0].contourf(gY, CNX, Gnorm, cl, cmap=color_map, vmin=0.0, vmax=0.6)
    axs[k][0].set_yscale("log")
    # fig.colorbar(cs,ax=axs[k][0])
    fig.colorbar(
        cm.ScalarMappable(norm=cs.norm, cmap=cs.cmap),
        ticks=np.arange(0, 0.6 + 0.1, 0.1),
        ax=axs[k][0],
    )
    axs[k][0].set_title(String_eA[k], fontsize=labelfont)
k = len(eA) - 1
Gnorm = np.zeros((len(CNS), len(gamma_S)))
for m in range(0, len(gamma_S)):
    for i in range(0, len(CNS)):
        [emax_S, YCO2, dGrS, xN, xEA, v_EA, v_N] = efficiency_CNED_Inf(
            gamma_S[m], T, eA[k], CNB, NS, CNS[i]
        )
        if v_N < 0:  # NH4 mineralization occurs
            Gnorm[i, m] = emax_S
        else:
            Gnorm[i, m] = np.minimum(emax_S, (emax_S / v_N) * INORM)
[gY, CNX] = np.meshgrid(gamma_S, CNS)
cl = np.arange(0, 0.1 + 0.01, 0.01)
cs = axs[k][0].contourf(gY, CNX, Gnorm, cl, cmap=color_map, vmin=0.0, vmax=0.1)
axs[k][0].set_yscale("log")
# fig.colorbar(cs,ax=axs[k][0])
fig.colorbar(
    cm.ScalarMappable(norm=cs.norm, cmap=cs.cmap),
    ticks=np.arange(0, 0.1 + 0.02, 0.02),
    ax=axs[k][0],
)
axs[k][0].set_title(String_eA[k], fontsize=labelfont)


fcaption = ["(A)", "(B)", "(C)", "(D)"]
for i in range(len(axs)):
    axs[i][0].text(
        0.0, 1.05, fcaption[i], transform=axs[i][0].transAxes, fontsize=labelfont
    )
axs[2][0].set_xlabel(r"$\gamma_{ED}$", fontsize=labelfont)
axs[3][0].set_xlabel(r"$\gamma_{ED}$", fontsize=labelfont)
axs[0][0].set_ylabel(r"$CN_{ED}$", fontsize=labelfont)
axs[2][0].set_ylabel(r"$CN_{ED}$", fontsize=labelfont)

axs[0][0].text(
    0.4, 0.8, "N limited", transform=axs[0][0].transAxes, fontsize=labelfont, color="w"
)
axs[0][0].text(
    0.075,
    0.15,
    "Energy\nlimited",
    transform=axs[0][0].transAxes,
    fontsize=labelfont,
    color="k",
)
axs[0][0].text(
    0.6, 0.15, "C limited", transform=axs[0][0].transAxes, fontsize=labelfont, color="w"
)

axs[1][0].text(
    0.4, 0.8, "N limited", transform=axs[1][0].transAxes, fontsize=labelfont, color="w"
)
axs[1][0].text(
    0.05,
    0.15,
    "Energy\nlimited",
    transform=axs[1][0].transAxes,
    fontsize=labelfont,
    color="k",
)
axs[1][0].text(
    0.5, 0.15, "C limited", transform=axs[1][0].transAxes, fontsize=labelfont, color="w"
)

axs[2][0].text(
    0.4, 0.8, "N limited", transform=axs[2][0].transAxes, fontsize=labelfont, color="w"
)
axs[2][0].text(
    0.05,
    0.15,
    "Energy\nlimited",
    transform=axs[2][0].transAxes,
    fontsize=labelfont,
    color="k",
)
axs[2][0].text(
    0.6, 0.15, "C limited", transform=axs[2][0].transAxes, fontsize=labelfont, color="k"
)

axs[3][0].text(
    0.075,
    0.9,
    "N limited",
    transform=axs[3][0].transAxes,
    fontsize=labelfont,
    color="k",
)
axs[3][0].text(
    0.7,
    0.5,
    "Energy\nlimited",
    transform=axs[3][0].transAxes,
    fontsize=labelfont,
    color="k",
)
axs[3][0].text(
    0.15,
    0.15,
    "C limited",
    transform=axs[3][0].transAxes,
    fontsize=labelfont,
    color="k",
)
plt.subplots_adjust(left=0.1, bottom=0.1, right=0.6, top=0.8)

fig.tight_layout()
fig.savefig("fig/png/Figure6.png", dpi=300)
fig.savefig("fig/svg/Figure6.svg", dpi=300)

plt.show()


# %% [markdown]
# #  Appendix
# ## Figure A2
#

# %%


gamma = np.arange(0.5, 8.1, 0.1)  # Degree of reduction of ED
dGrX = np.zeros((len(gamma)))
# dGB_ana = -2.7
# Anbolic reaction
dCGX = dCG_O2(gamma_B)
dGB_ana = gamma_B / gamma * dCG_O2(gamma) - dCGX

for i in range(len(gamma)):
    if gamma[i] < 4.67:
        dGrX[i] = -(666.7 / gamma[i] + 243.1)  # kJ/Cmol biomass
    else:
        dGrX[i] = -(157 * gamma[i] - 339)  # kJ/Cmol biomass
dGred = np.array([0, 5, 24.04, 40, 75.6, 85, 122.7])
# dGred = np.array([0,  5,24])
fig = plt.figure(figsize=(12, 4.5), facecolor="w")
axs = fig.subplots(nrows=1, ncols=3)

axs[0].plot(
    gamma,
    60.3 - 28.5 * (4 - gamma) + -dGred[0] * gamma,
    linewidth=3,
    color="k",
    label=str(dGred[0]),
)


for i in range(1, len(dGred)):
    dGcat = (60.3 - 28.5 * (4 - gamma)) + gamma * -dGred[i]
    axs[0].plot(
        gamma, dGcat, linewidth=2, color=LC[i - 1], label=str(dGred[i]),
    )
    Y = dGcat / (dGrX - dGB_ana + (gammaB / gamma) * dGcat)
    # above Y is same as Y = dGcat / (dGrX + dCGX)
    denom = dGrX - dGB_ana + (gammaB / gamma) * dGcat
    axs[1].plot(gamma, dGrX * Y, linewidth=2, color=LC[i - 1])
    axs[2].plot(gamma, denom, "--", linewidth=2, color=LC[i - 1])
    axs[2].plot(gamma, dGcat, "-", linewidth=2, color=LC[i - 1])

axs[1].plot(gamma, dGrX, "--", linewidth=3, color="k")
# axs[1].set_ylim([-1000, 200])
axs[0].set_ylim([-800, 200])
# axs[2].set_ylim(bottom=0)
# axs[2].set_ylim([0, -2000])

axs[0].plot([1.88, 1.88], axs[0].get_ylim(), "--", linewidth=1, color="k")
axs[0].plot(axs[0].get_xlim(), [0, 0], "--", linewidth=1, color="k")

axs[0].set_xlabel(r"$\gamma_{ED}$", fontsize=labelfont)
axs[1].set_xlabel(r"$\gamma_{ED}$", fontsize=labelfont)
axs[2].set_xlabel(r"$\gamma_{ED}$", fontsize=labelfont)
axs[0].set_ylabel(
    r"$ \Delta_{cat}G_{ED}$  kJ $\mathrm{(C mol \ ED)^{-1}}$", fontsize=labelfont
)

axs[1].set_ylabel(r"$ \Delta_{r}G$", fontsize=labelfont)


axs[0].text(0.0, 1.05, "(A)", transform=axs[0].transAxes, fontsize=labelfont)
axs[1].text(0.0, 1.05, "(B)", transform=axs[1].transAxes, fontsize=labelfont)
axs[2].text(0.0, 1.05, "(C)", transform=axs[2].transAxes, fontsize=labelfont)

axs[1].text(
    0.1,
    1.05,
    r" Dashed line = $ \Delta_{r}G_{B}$",
    transform=axs[1].transAxes,
    fontsize=labelfont - 2,
)
axs[1].text(
    0.1,
    0.95,
    r" Solid lines = $ \Delta_{r}G_{ED}$",
    transform=axs[1].transAxes,
    fontsize=labelfont - 2,
)

fig.tight_layout(pad=5, w_pad=0.5, h_pad=1)

lhlist = []
lstr = []
p1 = []
(p1,) = axs[0].plot(np.nan, np.nan, linestyle=lstyle[0], color="k", linewidth=3)
lstr.append(str(dGred[0]))
lhlist.append(p1)
for i in range(1, len(dGred)):
    (p1,) = axs[0].plot(
        np.nan, np.nan, linestyle=lstyle[0], color=LC[i - 1], linewidth=3
    )
    lstr.append(str(dGred[i]))
    lhlist.append(p1)

axs[0].legend(
    lhlist,
    lstr,
    loc="lower left",
    frameon=False,
    fontsize=labelfont - 2,
    ncol=4,
    bbox_to_anchor=(0.75, 1.15),
)
axs[0].text(
    -0.1,
    1.35,
    r"$ \frac{-\Delta_{red}G_{EA}}{\gamma_{EA}}$ kJ $\mathrm{(e^- mol)^{-1}}$",
    transform=axs[0].transAxes,
    fontsize=labelfont,
    color="k",
)

lhlist = []
p1 = []
for i in range(0, 2):
    (p1,) = axs[2].plot(np.nan, np.nan, linestyle=lstyle[i], color="k", linewidth=3)
    lhlist.append(p1)

axs[2].legend(
    lhlist,
    [r"$ \Delta_{cat}G_{ED}$", "denominator Eq. (34)"],
    loc="lower left",
    frameon=False,
    fontsize=labelfont - 2,
)

fig.savefig("fig/svg/FigureA2.svg", dpi=300)
fig.savefig("fig/png/FigureA2.png", dpi=300)
plt.show()


# %% [markdown]
# # Figure A3

# %%

eA = ["NO3", "Denitrification"]


gamma_S = np.linspace(0.5, 8, 100)

# INORM = np.array([0.01,0.2,1.5])
INORM = np.array([0.01, 0.1, 0.2, 0.5, 1.5])
# INORM = np.array([0.005]) / 0.5

NS = "NO3"
GG = []
for j in range(0, len(INORM)):
    G = []
    for k in range(0, len(eA)):
        Gnorm = np.zeros((len(CNS), len(gamma_S)))
        for m in range(0, len(gamma_S)):
            for i in range(0, len(CNS)):
                [emax_S, YCO2, dGrS, xN, xEA, v_EA, v_N] = efficiency_CNED_Inf(
                    gamma_S[m], T, eA[k], CNB, NS, CNS[i]
                )
                vN_met = (xN + xEA) * (1 - emax_S * gammaB / gamma_S[m])
                Gnorm[i, m] = np.minimum(emax_S, (emax_S / vN_met) * INORM[j])
        G.append(Gnorm)
    GG.append(G)
color_map = plt.cm.get_cmap("cividis")
color_map = color_map.reversed()
j = 0
fig = plt.figure(figsize=(12, 7), facecolor="w")
axs = fig.subplots(nrows=3, ncols=len(INORM), sharey="row", sharex="col")
G = GG[j]
Gnorm_diff = G[0] - G[1]  # Gnorm_diff = DNRA- Denitrification
id = Gnorm_diff > 0
Gnorm_diff_norm = Gnorm_diff
Gnorm_diff_norm[:] = 0
Gnorm_diff_norm[id] = 1
[gY, CNX] = np.meshgrid(gamma_S, CNS)


cs = axs[0, j].contourf(gY, CNX, G[0], cmap=color_map)
axs[0, j].set_yscale("log")
# v1= np.linspace(np.amin(G[0]), np.amax(G[0]),5)*1000
v1 = np.linspace(5, 15, 5)
norm = mpl.colors.Normalize(vmin=v1.min(), vmax=v1.max())
cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cs.cmap), ticks=v1, ax=axs[0, j])

# cbar= fig.colorbar(cs,ax=axs[0,j])
# v1=cbar.get_ticks()
cbar.ax.set_yticklabels(["{:1.0f}".format(i) for i in v1])
cbar.ax.set_title(r"         $\mathrm{x10^{-3}}$", fontsize=10)


cs = axs[1, j].contourf(gY, CNX, G[1], cmap=color_map)
axs[1, j].set_yscale("log")
# v1= np.linspace(np.amin(G[1]), np.amax(G[1]),5)*1000
norm = mpl.colors.Normalize(vmin=v1.min(), vmax=v1.max())
cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cs.cmap), ticks=v1, ax=axs[1, j])
cbar.ax.set_yticklabels(["{:1.0f}".format(i) for i in v1])
cbar.ax.set_title(r"          $\mathrm{x10^{-3}}$", fontsize=10)


v1 = np.array([0, 1])
cs = axs[2, j].contourf(
    gY, CNX, Gnorm_diff_norm, cmap=color_map, vmin=v1.min(), vmax=v1.max()
)
axs[2, j].set_yscale("log")
# norm = mpl.colors.Normalize(vmin=v1.min(), vmax=v1.max())
cbar = fig.colorbar(
    cm.ScalarMappable(norm=cs.norm, cmap=cs.cmap), ticks=v1, ax=axs[2, j]
)
cbar.ax.set_yticklabels(["{:1.1f}".format(i) for i in v1])

maxv = [0.015, 0.15, 0.3, 0.6, 0.6]
vstep = [0.001, 0.05, 0.1, 0.1, 0.1]
for j in range(1, len(INORM)):
    G = GG[j]
    Gnorm_diff = G[0] - G[1]  # Gnorm_diff = DNRA- Denitrification
    id = Gnorm_diff > 0
    Gnorm_diff_norm = Gnorm_diff
    Gnorm_diff_norm[:] = 0
    Gnorm_diff_norm[id] = 1
    [gY, CNX] = np.meshgrid(gamma_S, CNS)

    v1 = np.arange(0, maxv[j] + vstep[j], vstep[j])
    cs = axs[0, j].contourf(gY, CNX, G[0], cmap=color_map, vmin=v1.min(), vmax=v1.max())
    axs[0, j].set_yscale("log")
    # norm = mpl.colors.Normalize(vmin=v1.min(), vmax=v1.max())
    cbar = fig.colorbar(
        cm.ScalarMappable(norm=cs.norm, cmap=cs.cmap), ticks=v1, ax=axs[0, j]
    )
    cbar.ax.set_yticklabels(["{:1.1f}".format(i) for i in v1])

    cs = axs[1, j].contourf(gY, CNX, G[1], cmap=color_map, vmin=v1.min(), vmax=v1.max())
    axs[1, j].set_yscale("log")
    # norm = mpl.colors.Normalize(vmin=v1.min(), vmax=v1.max())
    cbar = fig.colorbar(
        cm.ScalarMappable(norm=cs.norm, cmap=cs.cmap), ticks=v1, ax=axs[1, j]
    )
    cbar.ax.set_yticklabels(["{:1.1f}".format(i) for i in v1])

    v1 = np.array([0.0, 1.0])
    cs = axs[2, j].contourf(
        gY, CNX, Gnorm_diff_norm, cmap=color_map, vmin=v1.min(), vmax=v1.max()
    )
    axs[2, j].set_yscale("log")
    # cbar= fig.colorbar(cs,ax=axs[2,j])
    # norm = mpl.colors.Normalize(vmin=v1.min(), vmax=v1.max())
    cbar = fig.colorbar(
        cm.ScalarMappable(norm=cs.norm, cmap=cs.cmap), ticks=v1, ax=axs[2, j]
    )
    # cbar.ax.set_yticklabels(["{:1.1f}".format(i) for i in v1])
for row in [0, 1, 2]:
    axs[row, 0].set_ylabel(r"$CN_{ED}$", fontsize=labelfont)
txt = ["(A)", "(B)", "(C)", "(D)", "(E)", "(F)"]
for col in range(0, len(INORM)):
    axs[2, col].set_xlabel(r"$\gamma_{ED}$", fontsize=labelfont - 2)
    axs[0, col].set_title(
        txt[col] + "\n" + r"$I_{norm} =" + str(INORM[col]) + "$", fontsize=labelfont - 2
    )
fig.tight_layout()
fig.savefig("fig/png/FigureA3.png", dpi=300)
fig.savefig("fig/svg/FigureA3.svg", dpi=300)
plt.show()


# %% [markdown]
# # Figure A4

# %%

INORM = 0.2
eA = ["NO3", "Denitrification"]
eAstr = ["DNRA", "Denitrification"]
CNS = np.arange(1, 20.1, 0.1)
CNS = np.append(CNS, np.arange(21, 1001, 1))

gamma_S = np.arange(1, 8 + 1, 1)
#
# gamma_S = np.array([2]);
NS = "NO3"


fig = plt.figure(figsize=(20, 10), facecolor="w")
axs = fig.subplots(nrows=1, ncols=5)


for m in range(0, len(gamma_S)):
    Gnorm = np.zeros((len(CNS), 2))
    e = np.zeros((len(CNS), 2))
    vN = np.zeros((len(CNS), 2))
    xNgrowth = np.zeros((len(CNS), 2))
    NEA = np.zeros((len(CNS), 2))
    for i in range(0, len(CNS)):
        [emax_S, YCO2, dGrS, xN, xEA, v_EA, v_N] = efficiency_CNED_Inf(
            gamma_S[m], T, eA[0], CNB, NS, CNS[i]
        )
        vN_met = (xN + xEA) * (1 - emax_S * gammaB / gamma_S[m])
        Gnorm[i, 0] = np.minimum(emax_S, (emax_S / vN_met) * INORM)
        e[i, 0] = emax_S
        vN[i, 0] = vN_met
        vN[i, 0] = vN_met
        xNgrowth[i, 0] = xN * (1 - emax_S * gammaB / gamma_S[m])
        NEA[i, 0] = xEA * (1 - emax_S * gammaB / gamma_S[m])

        [emax_S, YCO2, dGrS, xN, xEA, v_EA, v_N] = efficiency_CNED_Inf(
            gamma_S[m], T, eA[1], CNB, NS, CNS[i]
        )
        vN_met = (xN + xEA) * (1 - emax_S * gammaB / gamma_S[m])
        Gnorm[i, 1] = np.minimum(emax_S, (emax_S / vN_met) * INORM)
        e[i, 1] = emax_S
        vN[i, 1] = vN_met
        vN[i, 1] = vN_met
        xNgrowth[i, 1] = xN * (1 - emax_S * gammaB / gamma_S[m])
        NEA[i, 1] = xEA * (1 - emax_S * gammaB / gamma_S[m])
    # fig = plt.figure(figsize=(20,10), facecolor='w')
    # axs = fig.subplots(nrows=1, ncols=5)
    axs[0].plot(CNS, Gnorm[:, 0], linestyle=lstyle[0], linewidth=3, color=LC[m])
    axs[0].plot(CNS, Gnorm[:, 1], linestyle=lstyle[1], linewidth=3, color=LC[m])

    axs[1].plot(CNS, e[:, 0], linestyle=lstyle[0], linewidth=3, color=LC[m])
    axs[1].plot(CNS, e[:, 1], linestyle=lstyle[1], linewidth=3, color=LC[m])

    axs[2].plot(CNS, vN[:, 0], linestyle=lstyle[0], linewidth=3, color=LC[m])
    axs[2].plot(CNS, vN[:, 1], linestyle=lstyle[1], linewidth=3, color=LC[m])

    axs[3].plot(CNS, xNgrowth[:, 0], linestyle=lstyle[0], linewidth=3, color=LC[m])
    axs[3].plot(CNS, xNgrowth[:, 1], linestyle=lstyle[1], linewidth=3, color=LC[m])

    axs[4].plot(CNS, NEA[:, 0], linestyle=lstyle[0], linewidth=3, color=LC[m])
    axs[4].plot(CNS, NEA[:, 1], linestyle=lstyle[1], linewidth=3, color=LC[m])
for a in axs:
    a.set_xscale("log")
    a.set_xlabel(r"$CN_{ED}$", fontsize=labelfont)
    a.tick_params(axis="x", labelsize=axisfont)
    a.tick_params(axis="y", labelsize=axisfont)
axs[0].set_title(r"$G_{norm}$", fontsize=labelfont)
axs[1].set_title(r"$e$", fontsize=labelfont)
axs[2].set_title(r"$y_N$", fontsize=labelfont)
axs[3].set_title(r"$\nu_N$", fontsize=labelfont)
axs[4].set_title(r"$\nu_{EA}$", fontsize=labelfont)

(p1,) = axs[0].plot(np.nan, np.nan, linestyle=lstyle[0], color="k", linewidth=3)
(p2,) = axs[0].plot(np.nan, np.nan, linestyle=lstyle[1], color="k", linewidth=3)

leg1 = axs[0].legend(
    [p1, p2],
    [r"$\mathrm{NO_3^- → NH_4^+}$", r"$\mathrm{NO_3^- → N_2}$"],
    loc="upper left",
    frameon=False,
    fontsize=labelfont,
    ncol=2,
    bbox_to_anchor=(0.15, 1.15),
)
lhlist = []
lstr = []
for i in range(0, len(gamma_S)):
    (p1,) = axs[0].plot(np.nan, np.nan, linestyle=lstyle[0], color=LC[i], linewidth=5)
    lstr.append(r"$\gamma_{ED}=$" + str(gamma_S[i]))
    lhlist.append(p1)
axs[0].legend(
    lhlist,
    lstr,
    loc="lower left",
    frameon=False,
    fontsize=labelfont,
    ncol=4,
    bbox_to_anchor=(2, 1.025),
)
axs[0].add_artist(leg1)
fig.tight_layout()
fig.savefig("fig/png/FigureA4.png", dpi=300)
fig.savefig("fig/svg/FigureA4.svg", dpi=300)
plt.show()


# %% [markdown]
# #  Figure A5

# %%

INORM = [0.01, 0.001]
eA = "O2"
CNS = np.arange(1, 20.1, 0.1)
CNS = np.append(CNS, np.arange(21, 1001, 1))

gamma_S = np.arange(1, 7, 1)
NS = "NH4"
fig = plt.figure(figsize=(6, 5), facecolor="w")
ax = plt.axes()
lw = [2, 4]
for j in range(0, len(INORM)):
    Gnorm = np.zeros((len(CNS), 1))
    for m in range(0, len(gamma_S)):
        for i in range(0, len(CNS)):
            [emax_S, YCO2, dGrS, xN, xEA, v_EA, v_N] = efficiency_CNED_Inf(
                gamma_S[m], T, eA, CNB, NS, CNS[i]
            )
            if v_N < 0:  # NH4 mineralization occurs
                Gnorm[i] = emax_S
            else:
                Gnorm[i] = np.minimum(emax_S, CNB * (1 / CNS[i] + INORM[j]))
        ax.plot(CNS, Gnorm, linestyle=lstyle[j], linewidth=lw[j], color=LC[m])
plt.ylabel(r"${G_{norm}}$", fontsize=labelfont)
plt.xlabel(r"${CN_{ED}}$", fontsize=labelfont)
plt.ylim([0.005, 1])
plt.yscale("log")
plt.xscale("log")
lhlist = []
lstr = []
for j in gamma_S:
    lstr.append(str(j))
leg1 = ax.legend(lstr, loc="lower left", frameon=False, title=r"$\gamma_{ED}}$")
for i in leg1.get_lines():
    i.set_lw(3)
lstr = []
for i in range(0, len(INORM)):
    (p1,) = ax.plot(np.nan, np.nan, linestyle=lstyle[i], linewidth=3, color="k")
    lstr.append(r"$I_{norm} =" + str(INORM[i]) + "$")
    lhlist.append(p1)
ax.legend(lhlist, lstr, loc="upper right", frameon=False)
ax.add_artist(leg1)
ax.grid(True, which="both")
fig.savefig("fig/png/FigureA5.png", dpi=300)
fig.savefig("fig/svg/FigureA5.svg", dpi=300)
plt.show()


# %% [markdown]
# # Figure A6

# %%

eA = ["O2", "Fe(III)", "FeOOH", "SO4"]
# eA = ['O2']

String_eA = [r"$\mathrm{O_2}$", "ferrihydrite", "goethite", r"$\mathrm{SO_4^{2-}}$"]
color_map = plt.cm.get_cmap("Spectral")
color_map = color_map.reversed()
gamma_S = np.linspace(1, 8, 71)

INORM = 0.01

fig = plt.figure(figsize=(10, 7), facecolor="w")
axs = fig.subplots(nrows=2, ncols=2, sharey="row", sharex="col")
axs = axs.reshape((2 * 2, 1))

NS = "NH4"

for k in range(0, len(eA) - 1):
    Gnorm = np.zeros((len(CNS), len(gamma_S)))
    for m in range(0, len(gamma_S)):
        for i in range(0, len(CNS)):
            [emax_S, YCO2, dGrS, xN, xEA, v_EA, v_N] = efficiency_CNED_Inf(
                gamma_S[m], T, eA[k], CNB, NS, CNS[i]
            )
            if v_N < 0:  # NH4 mineralization occurs
                Gnorm[i, m] = emax_S
            else:
                Gnorm[i, m] = np.minimum(emax_S, CNB * (1 / CNS[i] + INORM))
    [gY, CNX] = np.meshgrid(gamma_S, CNS)
    cl = np.arange(0.05, 0.6 + 0.05, 0.05)
    cs = axs[k][0].contourf(gY, CNX, Gnorm, cl, cmap=color_map, vmin=0.0, vmax=0.6)
    # fig.colorbar(cs,ax=axs[k][0])
    axs[k][0].set_yscale("log")
    # fig.colorbar(cs,ax=axs[k][0])
    fig.colorbar(
        cm.ScalarMappable(norm=cs.norm, cmap=cs.cmap),
        ticks=np.arange(0, 0.6 + 0.1, 0.1),
        ax=axs[k][0],
    )
    axs[k][0].set_title(String_eA[k], fontsize=labelfont)
k = len(eA) - 1
Gnorm = np.zeros((len(CNS), len(gamma_S)))
for m in range(0, len(gamma_S)):
    for i in range(0, len(CNS)):
        [emax_S, YCO2, dGrS, xN, xEA, v_EA, v_N] = efficiency_CNED_Inf(
            gamma_S[m], T, eA[k], CNB, NS, CNS[i]
        )
        if v_N < 0:  # NH4 mineralization occurs
            Gnorm[i, m] = emax_S
        else:
            Gnorm[i, m] = np.minimum(emax_S, CNB * (1 / CNS[i] + INORM))
[gY, CNX] = np.meshgrid(gamma_S, CNS)
cl = np.arange(0, 0.1 + 0.01, 0.01)
cs = axs[k][0].contourf(gY, CNX, Gnorm, cl, cmap=color_map, vmin=0.0, vmax=0.1)
axs[k][0].set_yscale("log")
fig.colorbar(
    cm.ScalarMappable(norm=cs.norm, cmap=cs.cmap),
    ticks=np.arange(0, 0.1 + 0.02, 0.02),
    ax=axs[k][0],
)
axs[k][0].set_title(String_eA[k], fontsize=labelfont)

fig.tight_layout(pad=1)

fcaption = ["(A)", "(B)", "(C)", "(D)"]
for i in range(len(axs)):
    axs[i][0].text(
        0.0, 1.05, fcaption[i], transform=axs[i][0].transAxes, fontsize=labelfont
    )
axs[2][0].set_xlabel(r"$\gamma_{ED}$", fontsize=labelfont)
axs[3][0].set_xlabel(r"$\gamma_{ED}$", fontsize=labelfont)
axs[0][0].set_ylabel(r"$CN_{ED}$", fontsize=labelfont)
axs[2][0].set_ylabel(r"$CN_{ED}$", fontsize=labelfont)

axs[0][0].text(
    0.4, 0.8, "N limited", transform=axs[0][0].transAxes, fontsize=labelfont, color="w"
)
axs[0][0].text(
    0.075,
    0.15,
    "Energy\nlimited",
    transform=axs[0][0].transAxes,
    fontsize=labelfont,
    color="k",
)
axs[0][0].text(
    0.6, 0.15, "C limited", transform=axs[0][0].transAxes, fontsize=labelfont, color="w"
)

axs[1][0].text(
    0.4, 0.8, "N limited", transform=axs[1][0].transAxes, fontsize=labelfont, color="w"
)
axs[1][0].text(
    0.05,
    0.15,
    "Energy\nlimited",
    transform=axs[1][0].transAxes,
    fontsize=labelfont,
    color="k",
)
axs[1][0].text(
    0.5, 0.15, "C limited", transform=axs[1][0].transAxes, fontsize=labelfont, color="w"
)

axs[2][0].text(
    0.4, 0.8, "N limited", transform=axs[2][0].transAxes, fontsize=labelfont, color="w"
)
axs[2][0].text(
    0.05,
    0.15,
    "Energy\nlimited",
    transform=axs[2][0].transAxes,
    fontsize=labelfont,
    color="k",
)
axs[2][0].text(
    0.5, 0.15, "C limited", transform=axs[2][0].transAxes, fontsize=labelfont, color="k"
)

axs[3][0].text(
    0.15,
    0.9,
    "N limited",
    transform=axs[3][0].transAxes,
    fontsize=labelfont,
    color="k",
)
axs[3][0].text(
    0.7,
    0.5,
    "Energy\nlimited",
    transform=axs[3][0].transAxes,
    fontsize=labelfont,
    color="k",
)
axs[3][0].text(
    0.15,
    0.15,
    "C limited",
    transform=axs[3][0].transAxes,
    fontsize=labelfont,
    color="k",
)

fig.tight_layout()
fig.savefig("fig/png/FigureA6.png", dpi=300)
fig.savefig("fig/svg/FigureA6.svg", dpi=300)
plt.show()
