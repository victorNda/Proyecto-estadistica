import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


#  CREAR DATASET SIMULADO

np.random.seed()

data = {
    "Fecha": pd.date_range(start="2024-01-01", periods=120, freq="D"),
    "Region": np.random.choice(["Norte", "Sur", "Este", "Oeste"], 120),
    "Producto": np.random.choice(["Laptop", "Tablet", "Celular", "Monitor"], 120),
    "Vendedor": np.random.choice(["Ana", "Luis", "Carlos", "Marta"], 120),
    "Precio": np.random.randint(200, 2000, 120),
    "Cantidad": np.random.randint(1, 10, 120)
}

df = pd.DataFrame(data)

# Introducir valores faltantes
df.loc[np.random.choice(df.index, 5), "Precio"] = np.nan


#  LIMPIEZA DE DATOS


print("Valores nulos antes de limpiar:")
print(df.isnull().sum())

# Reemplazar precios faltantes con el promedio
df["Precio"] = df["Precio"].fillna(df["Precio"].mean())

print("\nValores nulos después de limpiar:")
print(df.isnull().sum())


# CREAR NUEVAS COLUMNAS


df["Ingreso"] = df["Precio"] * df["Cantidad"]
df["Mes"] = df["Fecha"].dt.month


# MÉTRICAS GENERALES


print("\nIngreso total:", df["Ingreso"].sum())
print("Ingreso promedio:", df["Ingreso"].mean())


#  AGRUPACIONES


ingreso_region = df.groupby("Region")["Ingreso"].sum()
ingreso_mes = df.groupby("Mes")["Ingreso"].sum()
ranking_vendedores = df.groupby("Vendedor")["Ingreso"].sum().sort_values(ascending=False)


#  GRÁFICAS CON PANDAS


# Barras - Ingreso por Región
ingreso_region.plot(
    kind="bar",
    title="Ingreso Total por Región",
    xlabel="Región",
    ylabel="Ingreso Total",
    rot=45
)
plt.show()


# Línea - Ingreso por Mes
ingreso_mes.plot(
    kind="line",
    title="Ingreso Total por Mes",
    xlabel="Mes",
    ylabel="Ingreso Total"
)
plt.show()


# Barras - Ranking de Vendedores
ranking_vendedores.plot(
    kind="bar",
    title="Ranking de Vendedores por Ingreso",
    xlabel="Vendedor",
    ylabel="Ingreso Total",
    rot=45
)
plt.show()


# Histograma - Distribución de Ingresos
df["Ingreso"].plot(
    kind="hist",
    bins=15,
    title="Distribución de Ingresos",
    xlabel="Ingreso"
)
plt.show()


# Relación Precio vs Ingreso
df.plot(
    kind="scatter",
    x="Precio",
    y="Ingreso",
    title="Relación entre Precio e Ingreso"
)
plt.show()

# Mapa de calor

plt.figure(figsize=(8,5))
sns.heatmap(df[["Precio", "Cantidad", "Ingreso"]].corr(),annot=True)
plt.show()



# CORRELACIÓN


print("\nMatriz de correlación:")
print(df[["Precio", "Cantidad", "Ingreso"]].corr())


# EXPORTAR RESULTADO


df.to_csv("ventas_procesadas.csv", index=False)

print("\nArchivo 'ventas_procesadas.csv' generado correctamente.")