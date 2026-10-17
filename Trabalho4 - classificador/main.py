"""
Trabalho: classificação do dataset Iris (UCI)
Etapas: estatísticas descritivas, relações sépala/pétala, distribuições,
PCA e K-NN (split 80/20).

Dependências: pip install pandas numpy matplotlib scikit-learn
Estrutura: main.py e a pasta Data/ (com iris.data) no mesmo diretório.
Uso: python main.py   (gráficos são salvos na pasta ./saida)
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
OUT = "saida"
os.makedirs(OUT, exist_ok=True)

FEATURES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Data", "iris.data")
COLORS = {"Iris-setosa": "tab:blue", "Iris-versicolor": "tab:orange",
          "Iris-virginica": "tab:green"}


def titulo(txt):
    print("\n" + "=" * 70 + f"\n{txt}\n" + "=" * 70)


# Carregamento dos dados
def carregar_dados():
    df = pd.read_csv(DATA_PATH, header=None, names=FEATURES + ["class"]).dropna()
    # Correção dos erros citados na página da UCI (amostras 35 e 38)
    df.loc[34, FEATURES] = [4.9, 3.1, 1.5, 0.2]
    df.loc[37, FEATURES] = [4.9, 3.6, 1.4, 0.1]
    return df


df = carregar_dados()
print(df.head())
print(f"\nShape: {df.shape} | Valores faltantes: {int(df.isna().sum().sum())}")

# Média, desvio, moda (geral e por categoria) + frequência das classes
titulo("1. ESTATÍSTICAS DESCRITIVAS")


def resumo(dados):
    return pd.DataFrame({
        "média": dados[FEATURES].mean(),
        "desvio": dados[FEATURES].std(),
        "moda": [", ".join(map(str, dados[c].mode().tolist())) for c in FEATURES],
    }).round(3)


print("\n--- Geral ---")
print(resumo(df))
for cls, g in df.groupby("class"):
    print(f"\n--- {cls} ---")
    print(resumo(g))

print("\n--- Frequência das categorias ---")
freq = df["class"].value_counts()
print(pd.DataFrame({"frequência": freq,
                    "proporção (%)": (freq / len(df) * 100).round(1)}))

# Relação comprimento x largura (sépala e pétala) por categoria

def relacao(x, y, nome, arquivo):
    titulo(f"RELAÇÃO {nome.upper()}: comprimento x largura")
    fig, ax = plt.subplots(figsize=(7, 5))
    linhas = []
    for cls, g in df.groupby("class"):
        r = g[x].corr(g[y])
        a, b = np.polyfit(g[x], g[y], 1)
        linhas.append({"classe": cls, "correlação (Pearson)": round(r, 3),
                       "R²": round(r ** 2, 3), "inclinação": round(a, 3),
                       "intercepto": round(b, 3)})
        ax.scatter(g[x], g[y], label=cls, color=COLORS[cls], alpha=0.7)
        xs = np.linspace(g[x].min(), g[x].max(), 50)
        ax.plot(xs, a * xs + b, color=COLORS[cls])
    r_geral = df[x].corr(df[y])
    linhas.append({"classe": "GERAL", "correlação (Pearson)": round(r_geral, 3),
                   "R²": round(r_geral ** 2, 3), "inclinação": np.nan,
                   "intercepto": np.nan})
    print(pd.DataFrame(linhas).to_string(index=False))
    ax.set_xlabel(f"{nome} - comprimento (cm)")
    ax.set_ylabel(f"{nome} - largura (cm)")
    ax.set_title(f"{nome}: comprimento x largura")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{OUT}/{arquivo}", dpi=150)
    plt.close(fig)


relacao("sepal_length", "sepal_width", "Sépala", "2_relacao_sepala.png")
relacao("petal_length", "petal_width", "Pétala", "3_relacao_petala.png")

# Distribuição das categorias nas 4 dimensões
titulo("4. DISTRIBUIÇÃO DAS CATEGORIAS NAS 4 DIMENSÕES")

fig, axes = plt.subplots(2, 4, figsize=(16, 7))
for i, c in enumerate(FEATURES):
    for cls, g in df.groupby("class"):
        axes[0, i].hist(g[c], bins=12, alpha=0.6, label=cls, color=COLORS[cls])
    axes[0, i].set_title(c)
    df.boxplot(column=c, by="class", ax=axes[1, i], grid=False)
    axes[1, i].set_title("")
    axes[1, i].set_xlabel("")
    axes[1, i].tick_params(axis="x", rotation=20)
axes[0, 0].legend(fontsize=8)
fig.suptitle("Histogramas (topo) e boxplots (base) por classe")
fig.tight_layout()
fig.savefig(f"{OUT}/4_distribuicoes.png", dpi=150)
plt.close(fig)

# Matriz de dispersão (todos os pares de variáveis)
fig, axes = plt.subplots(4, 4, figsize=(12, 12))
for i, yc in enumerate(FEATURES):
    for j, xc in enumerate(FEATURES):
        ax = axes[i, j]
        if i == j:
            for cls, g in df.groupby("class"):
                ax.hist(g[xc], bins=12, alpha=0.6, color=COLORS[cls])
        else:
            for cls, g in df.groupby("class"):
                ax.scatter(g[xc], g[yc], s=10, color=COLORS[cls], alpha=0.7)
        if i == 3:
            ax.set_xlabel(xc)
        if j == 0:
            ax.set_ylabel(yc)
fig.tight_layout()
fig.savefig(f"{OUT}/4_matriz_dispersao.png", dpi=130)
plt.close(fig)

# Separabilidade: razão entre variância entre-classes e intra-classes (F de Fisher)
media_geral = df[FEATURES].mean()
sep = {}
for c in FEATURES:
    entre = sum(len(g) * (g[c].mean() - media_geral[c]) ** 2
                for _, g in df.groupby("class")) / 2
    dentro = sum(((g[c] - g[c].mean()) ** 2).sum()
                 for _, g in df.groupby("class")) / (len(df) - 3)
    sep[c] = entre / dentro
print("\nPoder de separação por variável (estatística F; maior = melhor):")
print(pd.Series(sep).sort_values(ascending=False).round(1).to_string())

# PCA
titulo("5. PCA (Principal Component Analysis)")

X = df[FEATURES].values
y = df["class"].to_numpy(dtype=object)
X_pad = StandardScaler().fit_transform(X)  # PCA exige variáveis na mesma escala

pca = PCA(n_components=4).fit(X_pad)
var = pca.explained_variance_ratio_
print("Variância explicada por componente (%):", (var * 100).round(2))
print("Variância acumulada (%):               ", (np.cumsum(var) * 100).round(2))
print("\nCargas (loadings) dos componentes:")
print(pd.DataFrame(pca.components_.T, index=FEATURES,
                   columns=[f"PC{i+1}" for i in range(4)]).round(3))

Z = pca.transform(X_pad)
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for cls in COLORS:
    m = y == cls
    axes[0].scatter(Z[m, 0], Z[m, 1], label=cls, color=COLORS[cls], alpha=0.7)
axes[0].set_xlabel(f"PC1 ({var[0]*100:.1f}%)")
axes[0].set_ylabel(f"PC2 ({var[1]*100:.1f}%)")
axes[0].set_title("Projeção PCA (2 componentes)")
axes[0].legend()
axes[1].bar(range(1, 5), var * 100, label="individual")
axes[1].plot(range(1, 5), np.cumsum(var) * 100, "ro-", label="acumulada")
axes[1].set_xlabel("Componente")
axes[1].set_ylabel("Variância explicada (%)")
axes[1].set_xticks(range(1, 5))
axes[1].legend()
fig.tight_layout()
fig.savefig(f"{OUT}/5_pca.png", dpi=150)
plt.close(fig)

# Split 80/20 e K-NN
titulo("6. K-NN (80% treino / 20% teste)")

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED)
print(f"Treino: {len(X_tr)} amostras | Teste: {len(X_te)} amostras (split estratificado)")

# Escolha de k por validação cruzada (5-fold) APENAS no treino
ks = range(1, 26)
cv = [cross_val_score(make_pipeline(StandardScaler(), KNeighborsClassifier(k)),
                      X_tr, y_tr, cv=5).mean() for k in ks]
k_best = list(ks)[int(np.argmax(cv))]
print(f"Melhor k (CV no treino): {k_best} (acurácia CV = {max(cv):.4f})")

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(list(ks), cv, "o-")
ax.axvline(k_best, color="r", ls="--", label=f"k = {k_best}")
ax.set_xlabel("k")
ax.set_ylabel("Acurácia (validação cruzada)")
ax.set_title("Escolha de k")
ax.legend()
fig.tight_layout()
fig.savefig(f"{OUT}/6_escolha_k.png", dpi=150)
plt.close(fig)


def avaliar(modelo, Xtr, Xte, nome, arquivo):
    modelo.fit(Xtr, y_tr)
    pred = modelo.predict(Xte)
    print(f"\n--- {nome} ---")
    print(f"Acurácia no teste: {accuracy_score(y_te, pred):.4f}")
    print(classification_report(y_te, pred, digits=3))
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ConfusionMatrixDisplay(confusion_matrix(y_te, pred, labels=list(COLORS)),
                           display_labels=[c.replace("Iris-", "") for c in COLORS]
                           ).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(nome)
    fig.tight_layout()
    fig.savefig(f"{OUT}/{arquivo}", dpi=150)
    plt.close(fig)


# (a) 4 variáveis originais, com padronização
avaliar(make_pipeline(StandardScaler(), KNeighborsClassifier(k_best)),
        X_tr, X_te, f"K-NN (k={k_best}) - 4 variáveis", "6_cm_knn_4var.png")

# (b) sem padronização (todas estão em cm, então deve ser parecido)
avaliar(KNeighborsClassifier(k_best), X_tr, X_te,
        f"K-NN (k={k_best}) - 4 variáveis sem padronização", "6_cm_knn_sem_pad.png")

# (c) PCA com 2 componentes (ajustado só no treino, para não vazar dados do teste)
avaliar(make_pipeline(StandardScaler(), PCA(n_components=2),
                      KNeighborsClassifier(k_best)),
        X_tr, X_te, f"K-NN (k={k_best}) - PCA 2 componentes", "6_cm_knn_pca2.png")

# (d) só as variáveis da pétala (as mais discriminativas)
idx = [FEATURES.index("petal_length"), FEATURES.index("petal_width")]
avaliar(make_pipeline(StandardScaler(), KNeighborsClassifier(k_best)),
        X_tr[:, idx], X_te[:, idx], f"K-NN (k={k_best}) - só pétala",
        "6_cm_knn_petala.png")

# Robustez: variação da acurácia em várias divisões 80/20
accs = []
for s in range(100):
    a, b, c, d = train_test_split(X, y, test_size=0.2, stratify=y, random_state=s)
    m = make_pipeline(StandardScaler(), KNeighborsClassifier(k_best)).fit(a, c)
    accs.append(accuracy_score(d, m.predict(b)))
print(f"\nRobustez (100 splits 80/20 distintos, k={k_best}): "
      f"acurácia média = {np.mean(accs):.4f} ± {np.std(accs):.4f} "
      f"(mín {np.min(accs):.3f}, máx {np.max(accs):.3f})")

print(f"\nGráficos salvos em ./{OUT}/")