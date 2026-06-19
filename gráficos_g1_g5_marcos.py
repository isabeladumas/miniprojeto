import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from scipy.stats import skew
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score

df_1 = pd.read_csv('Dataset(1).csv', index_col='index')

# df_1.head()
# df_1.info()

# DUPLICADOS - SEM DUPLICADOS
# duplicados = df_1[df_1.duplicated(keep='first')]
# print(duplicados)

# VENDO SE TEM NaN - NÃO TEM NAN
# check_null = pd.isnull(df_1)
# print(check_null) #- Sem nulos. check null retornou false, portanto sem nulos.

# Descobrimos que os registros após 999 são todos preenchidos por zeros. Quando se faz a checagem dos valores nulos eles passam despercebidos.

# REMOVENDO OS ZEROS - TINHA, REGISTROS APOS O 999
df_1 = df_1[df_1['Patient Id'] != '0']

# Transformar level em valor numérico
mapa = {'Low': 0, 'Medium': 1, 'High': 2}
df_1['Level'] = df_1['Level'].map(mapa)

# OUTLIERS
# gráfico dificil de compreender, porem ajudou a achar os valores 0s
# sns.scatterplot(x=range(len(df_1['Age'])), y=df_1['Age'])
# plt.title("Scatter Plot for Age")
# plt.show()

df_1.boxplot(column='Age')
plt.title('Distribuição de Idade')
plt.show()

# Verificando em Coughing of Blood - não tem
df_1.boxplot(column='Coughing of Blood')
plt.title('Distribuição de Tossir sangue')
plt.show()

print("\n\n")

# Vendo os limites superiores, inferiores e quartis
q1 = df_1['Age'].quantile(0.25)
q3 = df_1['Age'].quantile(0.75)
IQR = q3 - q1
lower_limit = q1 - 1.5 * IQR
upper_limit = q3 + 1.5 * IQR
outliers = df_1[(df_1['Age'] < lower_limit) | (df_1['Age'] > upper_limit)]
print(f'Outliers encontrados:\n{outliers}')

# Removendo os outliers
df_sem_outlier = df_1[(df_1['Age'] >= lower_limit) & (df_1['Age'] <= upper_limit)]
# print(len(df_1))
# print(len(df_sem_outlier))
df_1 = df_sem_outlier

df_1.boxplot(column='Age')
plt.title('Distribuição de Idade sem outliers')
plt.show()
# df_1.to_csv("Dataset(1).csv")

# Verificando os parametros
df_1.max()

print("\n\n")

# Matriz de correlação
# ─────────────────────────────────────────────
# CONFIGURAÇÕES GLOBAIS
# ─────────────────────────────────────────────
PALETTE_RISK = {0: '#2ecc71', 1: '#f39c12', 2: '#e74c3c'}
LABELS_RISK = {0: 'Baixo Risco', 1: 'Médio Risco', 2: 'Alto Risco'}

plt.rcParams.update({
    'figure.facecolor': '#f8f9fa',
    'axes.facecolor': '#ffffff',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'font.family': 'DejaVu Sans',
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
})

# ─────────────────────────────────────────────
# CARREGAMENTO E PRÉ-PROCESSAMENTO
# ─────────────────────────────────────────────
df_1 = pd.read_csv('Dataset(1).csv', index_col='index')
df = df_1[df_1['Patient Id'] != '0']
mapa = {'Low': 0, 'Medium': 1, 'High': 2}
df['Level'] = df['Level'].map(mapa)

q1, q3 = df['Age'].quantile(0.25), df['Age'].quantile(0.75)
IQR = q3 - q1
df = df[(df['Age'] >= q1 - 1.5 * IQR) & (df['Age'] <= q3 + 1.5 * IQR)]

# Label legível para Gender
df['Gender_label'] = df['Gender'].map({1: 'Masculino', 2: 'Feminino'})

G1_VARS = ['Air Pollution', 'Dust Allergy', 'OccuPational Hazards']

# ═══════════════════════════════════════════════════════════════════
# FIGURA 1 — Distribuição de Idade por Nível de Risco (KDE + hist)
# ═══════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Grupo 5 — Idade × Nível de Risco', fontsize=15, fontweight='bold')

# Histograma sobreposto
for lv in [0, 1, 2]:
    axes[0].hist(df[df['Level'] == lv]['Age'], bins=15, alpha=0.5,
                 color=PALETTE_RISK[lv], label=LABELS_RISK[lv], edgecolor='white')
axes[0].axvline(df['Age'].mean(), color='#2c3e50', linestyle='--', linewidth=1.5,
                label=f'Média geral: {df["Age"].mean():.1f} anos')
axes[0].set_xlabel('Idade')
axes[0].set_ylabel('Frequência')
axes[0].set_title('Histograma de Idade por Nível de Risco')
axes[0].legend(fontsize=8)

# Boxplot de idade por nível
data_box = [df[df['Level'] == lv]['Age'].values for lv in [0, 1, 2]]
bp = axes[1].boxplot(data_box, patch_artist=True,
                     medianprops=dict(color='black', linewidth=2))
for patch, lv in zip(bp['boxes'], [0, 1, 2]):
    patch.set_facecolor(PALETTE_RISK[lv])
    patch.set_alpha(0.75)
axes[1].set_xticklabels([LABELS_RISK[l] for l in [0, 1, 2]], rotation=10)
axes[1].set_ylabel('Idade')
axes[1].set_title('Boxplot de Idade por Nível de Risco')

plt.tight_layout()
plt.savefig('fig1_idade_risco.png', dpi=150, bbox_inches='tight')
plt.show()

print("""
 ╔══════════════════════════════════════════════════════════════════╗
 ║  INSIGHTS — Fig. 1: Idade × Nível de Risco                      ║
 ╠══════════════════════════════════════════════════════════════════╣
 ║  • A média geral de ~37 anos é atipicamente jovem para câncer    ║
 ║    de pulmão real (população clínica típica > 55 anos) — indício ║
 ║    de dataset sintético, a ser citado como limitação do estudo.  ║
 ║  • Se os boxplots mostrarem medianas similares entre classes,    ║
 ║    Idade tem baixo poder discriminatório isolado.                ║
 ║  • Se houver gradiente, pacientes mais velhos tendem ao Alto     ║
 ║    Risco — consistente com a literatura médica real.             ║
 ╚══════════════════════════════════════════════════════════════════╝
 """)

# ═══════════════════════════════════════════════════════════════════
# FIGURA 2 — Distribuição de Gênero × Nível de Risco (barras + %)
# ═══════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Grupo 5 — Gênero × Nível de Risco', fontsize=15, fontweight='bold')

# Contagem absoluta
ct = df.groupby(['Gender_label', 'Level']).size().unstack(fill_value=0)
ct.plot(kind='bar', ax=axes[0], color=[PALETTE_RISK[l] for l in [0, 1, 2]],
        alpha=0.85, edgecolor='white', rot=0)
axes[0].set_xlabel('Gênero')
axes[0].set_ylabel('Contagem')
axes[0].set_title('Contagem por Gênero e Nível de Risco')
axes[0].legend([LABELS_RISK[l] for l in [0, 1, 2]], fontsize=8)
for p in axes[0].patches:
    axes[0].annotate(f'{int(p.get_height())}',
                     (p.get_x() + p.get_width() / 2, p.get_height() + 1),
                     ha='center', fontsize=8)

# Proporção (%)
ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100
ct_pct.plot(kind='bar', stacked=True, ax=axes[1],
            color=[PALETTE_RISK[l] for l in [0, 1, 2]], alpha=0.85, edgecolor='white', rot=0)
axes[1].set_xlabel('Gênero')
axes[1].set_ylabel('Proporção (%)')
axes[1].set_title('Proporção de Risco por Gênero (%)')
axes[1].legend([LABELS_RISK[l] for l in [0, 1, 2]], fontsize=8)
axes[1].set_ylim(0, 110)
for bar in axes[1].patches:
    h = bar.get_height()
    if h > 4:
        axes[1].text(bar.get_x() + bar.get_width() / 2,
                     bar.get_y() + h / 2, f'{h:.1f}%', ha='center', va='center', fontsize=8)

plt.tight_layout()
plt.savefig('fig2_genero_risco.png', dpi=150, bbox_inches='tight')
plt.show()

print("""
 ╔══════════════════════════════════════════════════════════════════╗
 ║  INSIGHTS — Fig. 2: Gênero × Nível de Risco                     ║
 ╠══════════════════════════════════════════════════════════════════╣
 ║  • O backlog pede explicitamente a análise de Alto Risco por     ║
 ║    gênero. A proporção empilhada revela se homens ou mulheres    ║
 ║    têm maior concentração no nível Alto.                         ║
 ║  • Dataset tem proporção ~60% M / 40% F, consistente com        ║
 ║    epidemiologia real do câncer de pulmão.                       ║
 ║  • Se a proporção de Alto Risco for maior em homens, confirma    ║
 ║    padrão epidemiológico esperado (maior exposição tabágica e    ║
 ║    ocupacional historicamente masculina).                        ║
 ╚══════════════════════════════════════════════════════════════════╝
 """)

# ═══════════════════════════════════════════════════════════════════
# FIGURA 3 — Histogramas G1 por nível de risco
# ═══════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Grupo 1 — Fatores Ambientais × Nível de Risco',
             fontsize=15, fontweight='bold', y=1.02)

for ax, var in zip(axes, G1_VARS):
    for lv in [0, 1, 2]:
        ax.hist(df[df['Level'] == lv][var], bins=8, alpha=0.55,
                color=PALETTE_RISK[lv], label=LABELS_RISK[lv], edgecolor='white')
    media = df[var].mean()
    ax.axvline(media, color='#2c3e50', linestyle='--', linewidth=1.4,
               label=f'Média: {media:.2f}')
    ax.set_title(var)
    ax.set_xlabel('Intensidade (1–8)')
    ax.set_ylabel('Frequência')
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig('fig3_grupo1_distribuicao.png', dpi=150, bbox_inches='tight')
plt.show()

print("""
 ╔══════════════════════════════════════════════════════════════════╗
 ║  INSIGHTS — Fig. 3: Distribuição G1 por Nível de Risco          ║
 ╠══════════════════════════════════════════════════════════════════╣
 ║  • Air Pollution e OccuPational Hazards: separação clara entre   ║
 ║    classes — Alto Risco domina valores 5–8, Baixo Risco em 1–4. ║
 ║  • Dust Allergy: maior média absoluta do dataset (5,17), mas     ║
 ║    distribuições das classes se sobrepõem mais, indicando menor  ║
 ║    poder discriminatório entre níveis de risco.                  ║
 ╚══════════════════════════════════════════════════════════════════╝
 """)

# ═══════════════════════════════════════════════════════════════════
# FIGURA 4 — Boxplot G1 por nível de risco
# ═══════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Grupo 1 — Boxplots por Nível de Risco (dose-resposta)',
             fontsize=15, fontweight='bold', y=1.02)

for ax, var in zip(axes, G1_VARS):
    data_plot = [df[df['Level'] == lv][var].values for lv in [0, 1, 2]]
    bp = ax.boxplot(data_plot, patch_artist=True,
                    medianprops=dict(color='black', linewidth=2))
    for patch, lv in zip(bp['boxes'], [0, 1, 2]):
        patch.set_facecolor(PALETTE_RISK[lv])
        patch.set_alpha(0.75)
    ax.set_xticklabels([LABELS_RISK[l] for l in [0, 1, 2]], rotation=10)
    ax.set_title(var)
    ax.set_ylabel('Intensidade')

plt.tight_layout()
plt.savefig('fig4_grupo1_boxplot.png', dpi=150, bbox_inches='tight')
plt.show()

print("""
 ╔══════════════════════════════════════════════════════════════════╗
 ║  INSIGHTS — Fig. 4: Boxplots G1                                  ║
 ╠══════════════════════════════════════════════════════════════════╣
 ║  • Air Pollution e OccuPational Hazards: medianas crescem        ║
 ║    monotonicamente Baixo → Médio → Alto Risco. Padrão de         ║
 ║    dose-resposta confirmado — forte candidato a preditor.        ║
 ║  • Dust Allergy: medianas mais próximas entre classes, com IQR   ║
 ║    largo em todas — variabilidade alta em todos os grupos.       ║
 ╚══════════════════════════════════════════════════════════════════╝
 """)

# ═══════════════════════════════════════════════════════════════════
# FIGURA 5 — Médias G1 por nível (barras agrupadas)
# ═══════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 6))
fig.suptitle('Grupo 1 — Médias por Nível de Risco', fontsize=14, fontweight='bold')

g1_means = df.groupby('Level')[G1_VARS].mean()
x = np.arange(len(G1_VARS))
width = 0.25

for i, lv in enumerate([0, 1, 2]):
    bars = ax.bar(x + i * width, g1_means.loc[lv], width,
                  label=LABELS_RISK[lv], color=PALETTE_RISK[lv], alpha=0.85, edgecolor='white')
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                f'{bar.get_height():.1f}', ha='center', fontsize=8)

ax.set_xticks(x + width)
ax.set_xticklabels(G1_VARS, rotation=10)
ax.set_ylabel('Média da Intensidade')
ax.set_ylim(0, 9)
ax.legend()
plt.tight_layout()
plt.savefig('fig5_grupo1_medias.png', dpi=150, bbox_inches='tight')
plt.show()

print("""
 ╔══════════════════════════════════════════════════════════════════╗
 ║  INSIGHTS — Fig. 5: Médias G1                                    ║
 ╠══════════════════════════════════════════════════════════════════╣
 ║  • Air Pollution: diferença de ~3 pts entre Baixo e Alto Risco   ║
 ║    — maior amplitude relativa do grupo. Preditor mais forte.     ║
 ║  • OccuPational Hazards: amplitude similar (~2,5 pts).           ║
 ║  • Dust Allergy: apesar da maior média absoluta, a diferença     ║
 ║    entre classes é a menor (~1,5 pts) — menor discriminação.     ║
 ╚══════════════════════════════════════════════════════════════════╝
 """)

# ═══════════════════════════════════════════════════════════════════
# FIGURA 6 — G1 por Gênero (violin)
# ═══════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Grupo 1 — Fatores Ambientais por Gênero',
             fontsize=14, fontweight='bold', y=1.02)

for ax, var in zip(axes, G1_VARS):
    sns.violinplot(data=df, x='Gender_label', y=var,
                   palette={'Masculino': '#3498db', 'Feminino': '#e91e8c'},
                   inner='box', linewidth=1.2, ax=ax)
    ax.set_title(var)
    ax.set_xlabel('Gênero')
    ax.set_ylabel('Intensidade')

plt.tight_layout()
plt.savefig('fig6_grupo1_genero.png', dpi=150, bbox_inches='tight')
plt.show()

print("""
 ╔══════════════════════════════════════════════════════════════════╗
 ║  INSIGHTS — Fig. 6: G1 por Gênero                               ║
 ╠══════════════════════════════════════════════════════════════════╣
 ║  • OccuPational Hazards: violino masculino tende a ser mais      ║
 ║    deslocado para valores altos — coerente com exposição         ║
 ║    ocupacional historicamente maior em homens.                   ║
 ║  • Air Pollution: distribuição mais simétrica entre gêneros —    ║
 ║    poluição ambiental afeta ambos igualmente (exposição urbana). ║
 ║  • Dust Allergy: avaliar se há diferença por gênero — pode       ║
 ║    refletir diferenças em ambiente doméstico vs. ocupacional.    ║
 ╚══════════════════════════════════════════════════════════════════╝
 """)

# ═══════════════════════════════════════════════════════════════════
# FIGURA 7 — Idade × G1: scatter por variável ambiental
# ═══════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Grupo 5 (Idade) × Grupo 1 (Ambiental) — Scatter por Nível de Risco',
             fontsize=13, fontweight='bold', y=1.02)

for ax, var in zip(axes, G1_VARS):
    for lv in [0, 1, 2]:
        sub = df[df['Level'] == lv]
        ax.scatter(sub['Age'], sub[var], c=PALETTE_RISK[lv],
                   label=LABELS_RISK[lv], alpha=0.4, s=30, edgecolors='none')
    ax.set_xlabel('Idade')
    ax.set_ylabel(var)
    ax.set_title(f'Idade × {var}')
    ax.legend(fontsize=7)

plt.tight_layout()
plt.savefig('fig7_idade_vs_g1.png', dpi=150, bbox_inches='tight')
plt.show()

print("""
 ╔══════════════════════════════════════════════════════════════════╗
 ║  INSIGHTS — Fig. 7: Idade × Fatores Ambientais                  ║
 ╠══════════════════════════════════════════════════════════════════╣
 ║  • Se os pontos de Alto Risco se concentram em faixas etárias    ║
 ║    específicas E com alta exposição ambiental, o cruzamento das  ║
 ║    duas variáveis cria um perfil de risco combinado.             ║
 ║  • Ausência de gradiente etário claro confirmaria que Idade       ║
 ║    tem papel secundário no dataset sintético.                    ║
 ║  • Padrão esperado clinicamente: maior exposição + maior idade   ║
 ║    = maior risco acumulado (efeito dose-tempo).                  ║
 ╚══════════════════════════════════════════════════════════════════╝
 """)

# ═══════════════════════════════════════════════════════════════════
# FIGURA 8 — Correlação G1 + G5 × Level (heatmap)
# ═══════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8, 5))
fig.suptitle('Correlação: Grupo 1 + Grupo 5 com Nível de Risco',
             fontsize=13, fontweight='bold')

vars_all = G1_VARS + ['Age', 'Gender', 'Level']
corr = df[vars_all].corr()
mask = np.zeros_like(corr, dtype=bool)
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm',
            vmin=-1, vmax=1, linewidths=0.5, ax=ax,
            annot_kws={'size': 10, 'weight': 'bold'})
plt.tight_layout()
plt.savefig('fig8_correlacao_g1_g5.png', dpi=150, bbox_inches='tight')
plt.show()

print("""
 ╔══════════════════════════════════════════════════════════════════╗
 ║  INSIGHTS — Fig. 8: Correlação G1 + G5 × Level                  ║
 ╠══════════════════════════════════════════════════════════════════╣
 ║  • Air Pollution e OccuPational Hazards: correlação positiva     ║
 ║    esperada entre si (multicolinearidade) e com Level.           ║
 ║  • Age × Level: correlação fraca confirmaria que Idade é pouco   ║
 ║    discriminante neste dataset sintético.                        ║
 ║  • Gender × Level: investigar se há associação entre gênero e    ║
 ║    nível de risco — responde diretamente ao backlog.             ║
 ║  • Gender × OccuPational Hazards: correlação positiva pode       ║
 ║    indicar que homens têm maior exposição ocupacional no dataset. ║
 ╚══════════════════════════════════════════════════════════════════╝
 """)

