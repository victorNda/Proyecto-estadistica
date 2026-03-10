import numpy as np
import matplotlib.pyplot as plt

# ──────────────────────────────────────────
# DATOS
# ──────────────────────────────────────────

# 1. Moneda: X = 0 (sello), 1 (cara)
moneda_x   = [0, 1]
moneda_pmf = [0.5, 0.5]
moneda_cdf = [0.5, 1.0]

# 2. Un dado: X = 1..6
dado1_x   = list(range(1, 7))
dado1_pmf = [1/6] * 6
dado1_cdf = [i/6 for i in range(1, 7)]

# 3. Dos dados: X = suma (2..12)
dado2_x   = list(range(2, 13))
casos     = [1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1]
dado2_pmf = [c/36 for c in casos]
dado2_cdf = list(np.cumsum(dado2_pmf))

# ──────────────────────────────────────────
# TABLAS EN CONSOLA
# ──────────────────────────────────────────

print("\n=== 1 MONEDA ===")
print(f"{'X':<6} {'P(X=x)':<10} {'F(x)'}")
for x, p, f in zip(moneda_x, moneda_pmf, moneda_cdf):
    print(f"{x:<6} {p:<10.4f} {f:.4f}")

print("\n=== 1 DADO ===")
print(f"{'X':<6} {'P(X=x)':<10} {'F(x)'}")
for x, p, f in zip(dado1_x, dado1_pmf, dado1_cdf):
    print(f"{x:<6} {p:<10.4f} {f:.4f}")

print("\n=== 2 DADOS (suma) ===")
print(f"{'X':<6} {'Casos':<8} {'P(X=x)':<10} {'F(x)'}")
for x, c, p, f in zip(dado2_x, casos, dado2_pmf, dado2_cdf):
    print(f"{x:<6} {c:<8} {p:<10.4f} {f:.4f}")

# ──────────────────────────────────────────
# GRAFICAS
# ──────────────────────────────────────────

fig, axes = plt.subplots(3, 2, figsize=(12, 12))
fig.suptitle("Funciones de Probabilidad  y Acumulativa ", fontsize=14, fontweight="bold")
plt.subplots_adjust(hspace=0.5, wspace=0.35)

def graficar_pmf(ax, x, pmf, etiquetas, color, titulo):
    ax.bar(etiquetas, pmf, color=color, edgecolor="black", width=0.5)
    ax.set_title(titulo)
    ax.set_xlabel("x")
    ax.set_ylabel("P(X = x)")
    ax.set_ylim(0, max(pmf) * 1.4)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    for i, v in enumerate(pmf):
        ax.text(i, v + 0.005, f"{v:.4f}", ha="center", fontsize=8)

def graficar_cdf(ax, x, cdf, etiquetas, color, titulo):
    ax.step(range(len(etiquetas)), cdf, where="post", color=color, linewidth=2)
    ax.scatter(range(len(etiquetas)), cdf, color=color, s=50, zorder=5)
    ax.set_title(titulo)
    ax.set_xlabel("x")
    ax.set_ylabel("F(x) = P(X <= x)")
    ax.set_xticks(range(len(etiquetas)))
    ax.set_xticklabels(etiquetas)
    ax.set_ylim(0, 1.1)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    for i, v in enumerate(cdf):
        ax.text(i, v + 0.03, f"{v:.3f}", ha="center", fontsize=8)

# Fila 1 - Moneda
lbl_m = ["0 (Sello)", "1 (Cara)"]
graficar_pmf(axes[0, 0], moneda_x, moneda_pmf, lbl_m, "steelblue",    "Moneda - Probabilidad")
graficar_cdf(axes[0, 1], moneda_x, moneda_cdf, lbl_m, "darkorange",   "Moneda - Acumulativa")

# Fila 2 - 1 Dado
lbl_d1 = [str(x) for x in dado1_x]
graficar_pmf(axes[1, 0], dado1_x, dado1_pmf, lbl_d1, "seagreen",     "1 Dado - Probabilidad")
graficar_cdf(axes[1, 1], dado1_x, dado1_cdf, lbl_d1, "darkorange",   "1 Dado - Acumulativa")

# Fila 3 - 2 Dados
lbl_d2 = [str(x) for x in dado2_x]
graficar_pmf(axes[2, 0], dado2_x, dado2_pmf, lbl_d2, "mediumpurple", "2 Dados - Probabilidad")
graficar_cdf(axes[2, 1], dado2_x, dado2_cdf, lbl_d2, "darkorange",   "2 Dados - Acumulativa")

plt.savefig("probabilidad.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nGrafica guardada como probabilidad.png")