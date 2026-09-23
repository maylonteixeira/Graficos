import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Leitura do arquivo CSV e configurações iniciais
df = pd.read_csv('ecommerce_estatistica.csv')
sns.set_theme(style="whitegrid")


# 1. Gráfico de Histograma

plt.figure(figsize=(8, 6))
sns.histplot(df['Preço'], bins=20, kde=False, color='skyblue')
plt.title('Histograma de Preços dos Produtos')
plt.xlabel('Preço (R$)')
plt.ylabel('Frequência')
plt.tight_layout()
plt.show()


# 2. Gráfico de Dispersão

plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='N_Avaliações', y='Preço', color='coral')
plt.title('Dispersão: Preço vs. Número de Avaliações')
plt.xlabel('Número de Avaliações')
plt.ylabel('Preço (R$)')
plt.tight_layout()
plt.show()


# 3. Mapa de Calor (Correlação)

plt.figure(figsize=(8, 6))
num_cols = ['Nota', 'N_Avaliações', 'Desconto', 'Preço']
corr = df[num_cols].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Mapa de Calor de Correlação')
plt.tight_layout()
plt.show()


# 4. Gráfico de Barra

plt.figure(figsize=(10, 6))
preco_genero = df.groupby('Gênero')['Preço'].mean().reset_index()

sns.barplot(
    data=preco_genero,
    x='Gênero',
    y='Preço',
    hue='Gênero',      # Atribui o x ao hue
    palette='viridis',
    legend=False       # Desativa a legenda repetida
)

plt.title('Preço Médio por Gênero')
plt.xlabel('Gênero')
plt.ylabel('Preço Médio (R$)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# 5. Gráfico de Pizza

plt.figure(figsize=(10, 6))
temporada_counts = df['Temporada'].value_counts()

wedges, texts = plt.pie(
    temporada_counts,
    labels=None,    # Remove rótulos diretos das fatias
    autopct=None,   # Remove porcentagens
    startangle=90,
    colors=sns.color_palette('pastel')
)

# Adiciona a legenda lateral contendo o nome e a contagem de cada categoria
plt.legend(
    wedges,
    [f'{label} ({count})' for label, count in zip(temporada_counts.index, temporada_counts.values)],
    title="Temporadas",
    loc="center left",
    bbox_to_anchor=(1, 0, 0.5, 1)
)

plt.title('Distribuição por Temporada', fontsize=14, pad=20)
plt.tight_layout()
plt.show()


# 6. Gráfico de Densidade (KDE)

plt.figure(figsize=(8, 6))
sns.kdeplot(df['Nota'], fill=True, color='purple')
plt.title('Gráfico de Densidade (KDE) da Nota')
plt.xlabel('Nota')
plt.ylabel('Densidade')
plt.tight_layout()
plt.show()


# 7. Gráfico de Regressão

plt.figure(figsize=(8, 6))
sns.regplot(data=df, x='Desconto', y='Preço', color='green', scatter_kws={'alpha':0.5})
plt.title('Gráfico de Regressão: Preço vs. Desconto')
plt.xlabel('Desconto (%)')
plt.ylabel('Preço (R$)')
plt.tight_layout()
plt.show()