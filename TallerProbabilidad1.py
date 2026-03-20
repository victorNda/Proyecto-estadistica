import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom, norm
from scipy import stats
import os

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
def out(name):
    return os.path.join(OUTPUT_DIR, name)

# ══════════════════════════════════════════════════════
# DATOS
# ══════════════════════════════════════════════════════

# Moneda
m_x   = [0, 1]
m_pmf = [0.5, 0.5]
m_cdf = [0.5, 1.0]

# Dado
d_x   = list(range(1, 7))
d_pmf = [1/6]*6
d_cdf = [i/6 for i in range(1, 7)]

# Dos dados
dd_x   = list(range(2, 13))
casos  = [1,2,3,4,5,6,5,4,3,2,1]
dd_pmf = [c/36 for c in casos]
dd_cdf = list(np.cumsum(dd_pmf))

# Conjunta discreta: X=moneda, Y=dado 4 caras (independientes)
joint = np.outer([0.5, 0.5], [0.25]*4)
joint_cdf = np.zeros((2,4))
for i in range(2):
    for j in range(4):
        joint_cdf[i,j] = joint[:i+1,:j+1].sum()

# Conjunta continua: Normal bivariada
rho = 0.5
rv  = stats.multivariate_normal([0,0], [[1,rho],[rho,1]])
g   = np.linspace(-3, 3, 150)
Xg, Yg = np.meshgrid(g, g)
PDF2 = rv.pdf(np.dstack((Xg, Yg)))
CDF2 = np.cumsum(np.cumsum(PDF2, axis=0)*(g[1]-g[0]), axis=1)*(g[1]-g[0])

# Binomial: X ~ Bin(10, 0.4)
n, p = 10, 0.4
k = np.arange(0, n+1)
b_pmf = binom.pmf(k, n, p)
b_cdf = binom.cdf(k, n, p)

# Normal: X ~ N(50, 8)
mu, sg = 50, 8
xn = np.linspace(mu-4*sg, mu+4*sg, 400)
n_pdf = norm.pdf(xn, mu, sg)
n_cdf = norm.cdf(xn, mu, sg)

# ══════════════════════════════════════════════════════
# FIGURA 1 – Univariadas (3 filas x 2 columnas)
# ══════════════════════════════════════════════════════
fig, ax = plt.subplots(3, 2, figsize=(10, 10))
fig.suptitle("Distribuciones Univariadas", fontsize=13, fontweight="bold")
fig.subplots_adjust(hspace=0.5, wspace=0.35)

# Moneda PMF
ax[0,0].bar(["Sello","Cara"], m_pmf, color="steelblue", edgecolor="black")
ax[0,0].set_title("Moneda – PMF"); ax[0,0].set_ylim(0, 0.8)
ax[0,0].set_ylabel("P(X=x)")

# Moneda CDF
ax[0,1].step([0,1], m_cdf, where="post", color="darkorange", lw=2)
ax[0,1].scatter([0,1], m_cdf, color="darkorange", s=50, zorder=5)
ax[0,1].set_title("Moneda – CDF"); ax[0,1].set_ylim(0, 1.15)
ax[0,1].set_xticks([0,1]); ax[0,1].set_xticklabels(["Sello","Cara"])
ax[0,1].set_ylabel("F(x)")

# Dado PMF
ax[1,0].bar(d_x, d_pmf, color="seagreen", edgecolor="black")
ax[1,0].set_title("1 Dado – PMF"); ax[1,0].set_ylim(0, 0.28)
ax[1,0].set_xticks(d_x); ax[1,0].set_ylabel("P(X=x)")

# Dado CDF
ax[1,1].step(d_x, d_cdf, where="post", color="darkorange", lw=2)
ax[1,1].scatter(d_x, d_cdf, color="darkorange", s=50, zorder=5)
ax[1,1].set_title("1 Dado – CDF"); ax[1,1].set_ylim(0, 1.15)
ax[1,1].set_xticks(d_x); ax[1,1].set_ylabel("F(x)")

# Dos dados PMF
ax[2,0].bar(dd_x, dd_pmf, color="mediumpurple", edgecolor="black")
ax[2,0].set_title("2 Dados (suma) – PMF"); ax[2,0].set_ylabel("P(X=x)")
ax[2,0].set_xticks(dd_x)

# Dos dados CDF
ax[2,1].step(dd_x, dd_cdf, where="post", color="darkorange", lw=2)
ax[2,1].scatter(dd_x, dd_cdf, color="darkorange", s=40, zorder=5)
ax[2,1].set_title("2 Dados (suma) – CDF"); ax[2,1].set_ylim(0, 1.15)
ax[2,1].set_xticks(dd_x); ax[2,1].set_ylabel("F(x)")

for a in ax.flat:
    a.grid(axis="y", linestyle="--", alpha=0.4)

fig.savefig(out("fig1_univariadas.png"), dpi=130, bbox_inches="tight")

# ══════════════════════════════════════════════════════
# FIGURA 2 – Conjunta Discreta
# ══════════════════════════════════════════════════════
fig2, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
fig2.suptitle("Conjunta Discreta – Moneda × Dado 4 caras", fontweight="bold")

im1 = a1.imshow(joint, cmap="Blues", vmin=0, vmax=0.2)
a1.set_title("PMF  P(X=x, Y=y)")
a1.set_xticks([0,1,2,3]); a1.set_xticklabels(["Y=1","Y=2","Y=3","Y=4"])
a1.set_yticks([0,1]);     a1.set_yticklabels(["X=0","X=1"])
for i in range(2):
    for j in range(4):
        a1.text(j, i, f"{joint[i,j]:.3f}", ha="center", va="center", fontsize=11)
plt.colorbar(im1, ax=a1)

im2 = a2.imshow(joint_cdf, cmap="Oranges")
a2.set_title("CDF  F(x,y)")
a2.set_xticks([0,1,2,3]); a2.set_xticklabels(["Y≤1","Y≤2","Y≤3","Y≤4"])
a2.set_yticks([0,1]);     a2.set_yticklabels(["X≤0","X≤1"])
for i in range(2):
    for j in range(4):
        a2.text(j, i, f"{joint_cdf[i,j]:.3f}", ha="center", va="center", fontsize=11)
plt.colorbar(im2, ax=a2)

fig2.tight_layout()
fig2.savefig(out("fig2_conjunta_discreta.png"), dpi=130, bbox_inches="tight")

# ══════════════════════════════════════════════════════
# FIGURA 3 – Conjunta Continua
# ══════════════════════════════════════════════════════
fig3, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
fig3.suptitle(f"Conjunta Continua – Normal Bivariada (rho={rho})", fontweight="bold")

cp1 = a1.contourf(Xg, Yg, PDF2, levels=15, cmap="viridis")
plt.colorbar(cp1, ax=a1)
a1.set_title("PDF  f(x,y)"); a1.set_xlabel("x"); a1.set_ylabel("y")

cp2 = a2.contourf(Xg, Yg, CDF2, levels=15, cmap="plasma")
plt.colorbar(cp2, ax=a2)
a2.set_title("CDF  F(x,y)"); a2.set_xlabel("x"); a2.set_ylabel("y")

fig3.tight_layout()
fig3.savefig(out("fig3_conjunta_continua.png"), dpi=130, bbox_inches="tight")

# ══════════════════════════════════════════════════════
# FIGURA 4 – Binomial  X ~ Bin(10, 0.4)
# ══════════════════════════════════════════════════════
fig4, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
fig4.suptitle("Binomial  X ~ Bin(n=10, p=0.4)", fontweight="bold")

a1.bar(k, b_pmf, color="steelblue", edgecolor="black")
a1.axvline(n*p, color="red", linestyle="--", lw=1.5, label=f"media={n*p}")
a1.set_title("PMF"); a1.set_xlabel("k"); a1.set_ylabel("P(X=k)")
a1.set_xticks(k); a1.legend(); a1.grid(axis="y", linestyle="--", alpha=0.4)

a2.step(k, b_cdf, where="post", color="darkorange", lw=2)
a2.scatter(k, b_cdf, color="darkorange", s=40, zorder=5)
a2.set_title("CDF"); a2.set_xlabel("k"); a2.set_ylabel("F(k)")
a2.set_xticks(k); a2.set_ylim(0, 1.05)
a2.grid(axis="y", linestyle="--", alpha=0.4)

fig4.tight_layout()
fig4.savefig(out("fig4_binomial.png"), dpi=130, bbox_inches="tight")

# Resultados en consola
print(f"\n=== Binomial X ~ Bin(10, 0.4) ===")
print(f"  P(X=4)      = {binom.pmf(4,n,p):.4f}")
print(f"  P(X<=5)     = {binom.cdf(5,n,p):.4f}")
print(f"  P(X>=6)     = {1-binom.cdf(5,n,p):.4f}")

# ══════════════════════════════════════════════════════
# FIGURA 5 – Normal  X ~ N(50, 8)
# ══════════════════════════════════════════════════════
fig5, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
fig5.suptitle("Normal  X ~ N(50, 8)", fontweight="bold")

a1.plot(xn, n_pdf, color="steelblue", lw=2)
a1.fill_between(xn, n_pdf, where=(xn>=42)&(xn<=58), alpha=0.4, color="steelblue",
                label=f"P(42<X<58)={norm.cdf(58,mu,sg)-norm.cdf(42,mu,sg):.3f}")
a1.fill_between(xn, n_pdf, where=(xn>=66), alpha=0.5, color="tomato",
                label=f"P(X>66)={1-norm.cdf(66,mu,sg):.4f}")
a1.axvline(mu, color="black", linestyle="--", lw=1.5, label=f"media={mu}")
a1.set_title("PDF"); a1.set_xlabel("x"); a1.set_ylabel("f(x)")
a1.legend(fontsize=8); a1.grid(alpha=0.3)

a2.plot(xn, n_cdf, color="darkorange", lw=2)
for perc, col, label in [(0.5,"gray","P50"), (0.9,"green","P90"), (0.95,"red","P95")]:
    val = norm.ppf(perc, mu, sg)
    a2.axhline(perc, color=col, linestyle=":", lw=1)
    a2.axvline(val,  color=col, linestyle=":", lw=1, label=f"{label}={val:.1f}")
a2.set_title("CDF"); a2.set_xlabel("x"); a2.set_ylabel("F(x)")
a2.set_ylim(0, 1.05); a2.legend(fontsize=8); a2.grid(alpha=0.3)

fig5.tight_layout()
fig5.savefig(out("fig5_normal.png"), dpi=130, bbox_inches="tight")

print(f"\n=== Normal X ~ N(50, 8) ===")
print(f"  P(X<58)         = {norm.cdf(58,mu,sg):.4f}")
print(f"  P(X>66)         = {1-norm.cdf(66,mu,sg):.4f}")
print(f"  P(42<X<58)      = {norm.cdf(58,mu,sg)-norm.cdf(42,mu,sg):.4f}")
print(f"  Percentil 90    = {norm.ppf(0.9,mu,sg):.2f}")

print("\nFiguras guardadas.")
plt.show()