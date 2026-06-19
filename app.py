import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import joblib

from scipy.cluster.hierarchy import average
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import LabelEncoder
from scipy.stats import skew
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score,f1_score,roc_curve, roc_auc_score

st.set_page_config(page_title="Data Lung Cancer Risk", layout="wide")

@st.cache_data
def carregar_e_limpar_dados():
    df = pd.read_csv('Dataset(1).csv', index_col='index') # Ajustado o caminho para a sua pasta 'cancer'
    df = df[df['Patient Id'] != '0']
    mapa = {'Low': 0, 'Medium': 1, 'High': 2}
    df['Level'] = df['Level'].map(mapa)

    q1 = df['Age'].quantile(0.25)
    q3 = df['Age'].quantile(0.75)
    IQR = q3 - q1
    lower_limit = q1 - 1.5 * IQR
    upper_limit = q3 + 1.5 * IQR
    df = df[(df['Age'] >= lower_limit) & (df['Age'] <= upper_limit)]
    return df

df_1 = carregar_e_limpar_dados()

@st.cache_resource
def treinar_modelo(dados):
    X = dados.drop(columns=["Level", "Patient Id"])
    y = dados["Level"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    return model, scaler, X_test, y_test

model, scaler, X_test, y_test = treinar_modelo(df_1)

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)

X_colunas = df_1.drop(columns=["Level", "Patient Id"]).columns

#DIVISÃO DAS GUIAS E TITULO PRINCIPAL
st.title("Painel Preditivo: Análise de Risco de Câncer de Pulmão")
titulos_guias = ['Análise exploratória de dados', 'Simulador de diagnóstico', 'Desempenho das inteligências']
guia1, guia2, guia3 = st.tabs(titulos_guias)

#GUIA1 - ANALISE
with guia1:
    st.header('Análise exploratória de dados')

    #criação das colunas
    c_m1, c_m2, c_m3 = st.columns(3)
    c_m1.metric("Total de Pacientes Base", len(df_1))
    c_m2.metric("Idade Média", f"{df_1['Age'].mean():.1f} Anos")
    c_m3.metric("Fatores Avaliados", len(X_colunas))
    st.markdown("---")

    #Boxplots
    st.header("Verificação de outliers com boxplot")
    st.markdown("A análise prévia identificou a presença de dados discrepantes (**outliers**) exclusivamente na variável **Idade**. Por precaução, a coluna **'Coughing of Blood'** também foi inspecionada, mas nenhum outlier foi encontrado nela.")

    colCOM, col_do_graficoSEM, colcou = st.columns([1, 1, 1])
    with colCOM:
        df_bruto = pd.read_csv('Dataset(1).csv', index_col='index')
        df_bruto = df_bruto[df_bruto['Patient Id'] != '0']
        fig, ax = plt.subplots(figsize=(4, 3))
        df_bruto.boxplot(column='Age', ax=ax)
        ax.set_title('Distribuição de Idade com outliers', fontsize=10)
        st.pyplot(fig, use_container_width=False)

    with col_do_graficoSEM:
        fig, ax = plt.subplots(figsize=(4, 3))
        df_1.boxplot(column='Age', ax=ax)
        ax.set_title('Distribuição de Idade sem outliers', fontsize=10)
        st.pyplot(fig, use_container_width=False)

    with colcou:
        fig, ax = plt.subplots(figsize=(4, 3))
        df_1.boxplot(column='Coughing of Blood', ax=ax)
        ax.set_title('Distribuição de Coughing of blood', fontsize=10)
        st.pyplot(fig, use_container_width=False)
    st.markdown("---")

    #Imagens Histograma - vou sugerir retirada
    #Separado por grupos e expandir pelo tamanho dos histogramas
    st.subheader("Histogramas por grupos")

    with st.expander("🌍 Grupo 1 e 3— Fatores Ambientais & Historico Clinico"):
        st.markdown("#### Fatores Ambientais")
        st.image("fig3_grupo1_distribuicao.png", use_container_width=True)
        st.markdown("---")

        st.markdown("#### Historico Clinico")
        colg1, colg3 = st.columns([1, 1])
        with colg1:
            st.image("histogramag.png", use_container_width=True)
        with colg3:
            st.image("histogramag1.png", use_container_width=True)

    with st.expander("🚬 Grupo 2 — Comportamento & Estilo de Vida"):
        st.markdown("Análise de hábitos como tabagismo, uso de álcool, dieta e obesidade.")
        st.image("histograma_grupo_2.jpg", use_container_width=True)

    with st.expander("🩺 Grupo 4 — Sintomas Clínicos (Painel Longo)"):
        st.markdown("Distribuição da intensidade dos 11 sintomas relatados pelos pacientes.")
        st.image("histograma_grupo_4.jpg", use_container_width=True)
    st.markdown("---")

    #Mapa de correlação
    st.subheader("Mapas de correlação por grupo")

    with st.expander("🌍 Grupo 1, 5 e 3— Fatores Ambientais & Historico Clinico"):
        st.markdown("#### Fatores Ambientais")
        st.image("fig8_correlacao_g1_g5.png", use_container_width=True)
        st.markdown("---")

        st.markdown("#### Historico Clinico")
        st.image("correlaçãogrupo1.png", use_container_width=True)
        st.markdown("---")

    with st.expander("🚬🩺 Grupo 2 e 4 — Comportamento & Estilo de Vida e Sintomas Clínicos "):
        st.image("correlação2e3.png", use_container_width=True)
    st.markdown("---")  

    #Perguntas
    st.header("Respondendo as perguntas do backlog")
    st.write("❓ **Pergunta:** Dos casos com alta probabilidade de câncer, quantos são homens?")
    with st.expander("Clique aqui para ver a resposta detalhada"):
        st.markdown("""
        Nesses casos, **252 homens** apresentam uma alta probabilidade de câncer.
        """)

    st.write("❓ **Pergunta:** Quantos pacientes fumam e bebem alta quantidade de álcool? (nível maior que 6)")
    with st.expander("Clique aqui para ver os dados estatísticos"):
        st.markdown("""
        Identificamos **285 pacientes** que fumam e, simultaneamente, consomem uma alta quantidade de álcool.
        """)

    st.write(
        "❓ **Pergunta:** Qual o perfil dos pacientes com alta chance de incidência baseado em genética, ambiente e hábitos?")
    with st.expander("Clique aqui para ver a análise de perfil detalhada"):
        st.markdown("""
        A análise dos dados revelou os seguintes fatores de impacto:

        * **Hábitos:** O tabagismo, o uso abusivo de álcool e a obesidade estão fortemente correlacionados com a alta incidência.
        * **Ambiente:** Fatores ambientais, como a exposição a fumo passivo, também demonstraram um papel crítico no aumento do risco.
        * **Sintomas:** A severidade de sintomas físicos (como *Chest pain* e *Coughing of blood*) possui relação direta com o nível de desenvolvimento do câncer de pulmão.
        """)
        st.markdown("Para responder essas questões foram utilizadas essas fórmulas: ")
        st.image("img.png", use_container_width=True)

#GUIA COM OS INPUTS
with guia2:
    st.header('Simulador de diagnóstico')

    with st.form(key = 'forms'):
        st.markdown("Perguntas sobre váriaveis demográficas")
        idade = st.number_input("Qual a sua idade (apenas o valor numérico): ", min_value=0)
        genero = {"Homem": 1, "Mulher": 2}[st.selectbox("Qual é o seu genêro:", ["Homem", "Mulher"] )]

        st.markdown("Perguntas sobre fatores ambientais")
        airpollution = st.number_input("Qual é o seu nivel de exposição a poluição no ar? sendo 8 muito e 1 pouco:", min_value=1, max_value = 8)
        dustallergy = st.number_input("Qual é o seu nivel de sensibilidade a poeira? sendo 8 muito e 1 pouco:", min_value=1, max_value=8)
        occupationalhazards = st.number_input("Qual é o seu nivel de exposição a riscos ocupacionais? Inclui inalação de amianto, fumaça industrial, solventes químicos ou poeiras minerais, sendo 8 muito e 1 pouco:", min_value=1, max_value=8)

        st.markdown("Perguntas sobre comportamento & estilo de vida")
        smoking = st.number_input("Qual é a sua intensidade e frequência de tabagismo? sendo 8 muito e 1 pouco:", min_value=1, max_value = 8)
        passivesmoker = st.number_input("Qual é seu nível de exposição ao tabagismo de maneira passiva? sendo 8 muito e 1 pouco:", min_value=1, max_value=8)
        alcohol = st.number_input("Qual é a intensidade em que consome álcool? sendo 8 muito e 1 pouco:", min_value=1, max_value=8)
        balanceddiet = st.number_input("Como é a sua qualidade da alimentação? sendo 7 muito e 1 pouco:", min_value=1, max_value=7)
        obesity = st.number_input("Qual é seu nivel de obesidade? sendo 7 muito e 1 pouco:", min_value=1, max_value=7)

        st.markdown("Perguntas sobre Histórico Clínico")
        geneticrisk = st.number_input("Qual é o nível de predisposição genética e histórico familiar de câncer? Sendo 7 muito e 1 pouco:", min_value=1, max_value=7)
        chronic_lung_disease = st.number_input("Qual é a severidade de doenças pulmonares crônicas preexistentes? Sendo 7 muito e 1 pouco:", min_value=1, max_value=7)

        st.markdown("Perguntas sobre Sintomas Clínicos")
        chest_pain = st.number_input("Qual é a intensidade da dor torácica sentida? Sendo 9 muito e 1 pouco:", min_value=1, max_value=9)
        coughing_of_blood = st.number_input("Qual é a intensidade ou frequência de episódios de tosse com sangue? Sendo 9 muito e 1 pouco:", min_value=1, max_value=9)
        fatigue = st.number_input("Qual é o nível de fadiga persistente sentido? Sendo 9 muito e 1 pouco:", min_value=1, max_value=9)
        weight_loss = st.number_input("Qual foi a intensidade da perda de peso não intencional recente? Sendo 8 muito e 1 pouco:", min_value=1, max_value=8)
        shortness_of_breath = st.number_input("Qual é a frequência ou intensidade da falta de ar? Sendo 9 muito e 1 pouco:", min_value=1, max_value=9)
        wheezing = st.number_input("Qual é a intensidade do chiado na respiração? Sendo 8 muito e 1 pouco:", min_value=1, max_value=8)
        swallowing_difficulty = st.number_input("Qual é o nível de dificuldade para engolir alimentos ou líquidos? Sendo 8 muito e 1 pouco:", min_value=1, max_value=8)
        clubbing_of_finger_nails = st.number_input("Qual é o grau de alargamento das pontas dos dedos e unhas? Sendo 9 muito e 1 pouco:", min_value=1,max_value=9)
        frequent_cold = st.number_input("Qual é a frequência com que contrai resfriados ou infecções respiratórias? Sendo 7 muito e 1 pouco:", min_value=1, max_value=7)
        dry_cough = st.number_input("Qual é a intensidade ou persistência de tosse seca? Sendo 7 muito e 1 pouco:", min_value=1, max_value=7)
        snoring = st.number_input("Qual é a intensidade ou frequência do seu ronco durante o sono? Sendo 7 muito e 1 pouco:", min_value=1, max_value=7)

        enviado = st.form_submit_button("Enviar")

        if enviado:
            dados_paciente = {
                'Age': idade,
                'Gender': genero,
                'Air Pollution': airpollution,
                'Alcohol use': alcohol,
                'Dust Allergy': dustallergy,
                'OccuPational Hazards': occupationalhazards,
                'Genetic Risk': geneticrisk,
                'chronic Lung Disease': chronic_lung_disease,
                'Balanced Diet': balanceddiet,
                'Obesity': obesity,
                'Smoking': smoking,
                'Passive Smoker': passivesmoker,
                'Chest Pain': chest_pain,
                'Coughing of Blood': coughing_of_blood,
                'Fatigue': fatigue,
                'Weight Loss': weight_loss,
                'Shortness of Breath': shortness_of_breath,
                'Wheezing': wheezing,
                'Swallowing Difficulty': swallowing_difficulty,
                'Clubbing of Finger Nails': clubbing_of_finger_nails,
                'Frequent Cold': frequent_cold,
                'Dry Cough': dry_cough,
                'Snoring': snoring
            }
            model = joblib.load("model.pkl")
            scaler = joblib.load("scaler.pkl")

            df_entrada = pd.DataFrame([dados_paciente])
            st.dataframe(df_entrada)

            df_modelo = df_entrada.copy()
            df_modelo_scaled = scaler.transform(df_modelo)
            mapa_risco = {0: "Low", 1: "Medium", 2: "High"}

            probs = model.predict_proba(df_modelo_scaled)[0]

            st.success("Dados processados e analisados com sucesso!")

            col_resultado, col_dados = st.columns([1, 1])

            with col_resultado:
                st.subheader("Análise de Risco")

                for classe, p in zip(model.classes_, probs):
                    st.metric(label=f"Risco {mapa_risco[classe]}", value=f"{p * 100:.1f}%")

                classe_predita = model.classes_[probs.argmax()]
                porcentagem_predita = probs.max() * 100
                risco_nome = mapa_risco[classe_predita]

                if risco_nome == "Low":
                    st.success(f"Risco Classificado: BAIXO ({porcentagem_predita:.1f}%)")
                elif risco_nome == "Medium":
                    st.warning(f"Risco Classificado: MODERADO ({porcentagem_predita:.1f}%)")
                else:
                    st.error(f"Risco Classificado: ALTO ({porcentagem_predita:.1f}%)")

            with col_dados:
                st.subheader("Dados Enviados")
                df_exibicao = df_entrada.T.rename(columns={0: "Valor"})
                st.dataframe(df_exibicao)

with guia3:
    st.header('Desempenho das inteligências')
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image("confusão1.png", use_container_width=True)
    with col2:
        with col2:
            st.write("**Análise do Modelo Random Forest**")
            st.write("""
            Este gráfico mostra o desempenho do nosso modelo de Random Forest. 
            Como podemos observar pela matriz de confusão ao lado:
            """)
            st.markdown("- **Classe Low:** 61 amostras classificadas corretamente.")
            st.markdown("- **Classe Medium:** 59 amostras classificadas corretamente.")
            st.markdown("- **Classe High:** 78 amostras classificadas corretamente.")
            st.markdown("Não houve nenhuma ocorrência de confusão entre as classes (erros de falso positivo ou falso negativo zerados). Isso indica que as variáveis selecionadas possuem altíssimo poder preditivo para determinar o nível de risco do paciente.")
            st.metric(label="Acurácia Geral", value="100%")
    st.markdown("---")
    co1, co2 = st.columns([2, 1])
    with co1:
        st.write("**Análise do Modelo de regressão logística**")
        st.write("""
                   Este gráfico mostra o desempenho do nosso modelo de Regressão Logistica. 
                   Como podemos observar pela matriz de confusão ao lado:
                   """)
        st.markdown("- **Classe Low:** 61 amostras classificadas corretamente.")
        st.markdown("- **Classe Medium:** 59 amostras classificadas corretamente.")
        st.markdown("- **Classe High:** 78 amostras classificadas corretamente.")
        st.markdown("Não houve nenhuma ocorrência de confusão entre as classes (erros de falso positivo ou falso negativo zerados). Isso indica que as variáveis selecionadas possuem altíssimo poder preditivo para determinar o nível de risco do paciente.")
        st.metric(label="Acurácia Geral", value="100%")
    with co2:
        st.image("confusão2.png", use_container_width=True)
