import streamlit as st

# =============================
# 🔧 Funções - Questão 1
# =============================
def q1_etapa1():
    st.subheader("🏠 Q1 - 1️⃣ Análise Descritiva dos Dados")
    st.info("Inclua estatísticas descritivas, histogramas, boxplots, correlações...")
    import streamlit as st
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt

    st.subheader("🏠 Q1 - 1️⃣ Análise Descritiva dos Dados")

    # Upload do dataset
    uploaded_file = st.file_uploader("kc_house_data")
    
    if uploaded_file:
        df = carregar_dados(uploaded_file)

        st.success("✅ Arquivo carregado com sucesso!")

        # Preview do dataset
        st.markdown("### 🔍 Preview dos Dados")
        st.dataframe(df.head())

        # Estatísticas descritivas
        st.markdown("### 📊 Estatísticas Descritivas")
        st.dataframe(df.describe())

        # Mediana das variáveis
        st.markdown("### 🧮 Mediana das Variáveis Numéricas")
        st.dataframe(df.median(numeric_only=True))

        # Correlação com o preço
        if "price" in df.columns:
            st.markdown("### 🔗 Correlação com o Preço (`price`)")
            correlacoes = df.corr(numeric_only=True)["price"].sort_values(ascending=False)
            st.write(correlacoes)

            # Mapa de calor
            st.markdown("### 🔥 Mapa de Correlação")
            plt.figure(figsize=(10, 8))
            sns.heatmap(df.corr(numeric_only=True), annot=True, fmt=".2f", cmap="coolwarm")
            st.pyplot(plt.gcf())
            plt.clf()

        # Histogramas
        st.markdown("### 📈 Distribuição de Variáveis")
        cols_to_plot = st.multiselect("Selecione variáveis numéricas:", df.select_dtypes(include='number').columns.tolist(), default=["price", "sqft_living"])
        for col in cols_to_plot:
            fig, ax = plt.subplots()
            sns.histplot(df[col], kde=True, ax=ax)
            ax.set_title(f'Distribuição da variável: {col}')
            st.pyplot(fig)
    else:
        st.warning("⚠️ Por favor, envie um arquivo CSV para iniciar a análise.")
    
    
    

def q1_etapa2():
    st.subheader("🏠 Q1 - 2️⃣ Modelo de Regressão Linear")
    st.info("Crie o modelo de regressão linear com métricas de desempenho.")

def q1_etapa3():
    st.subheader("🏠 Q1 - 3️⃣ Interpretação dos Resultados")
    st.info("Explique os coeficientes e os pressupostos.")

def q1_etapa4():
    st.subheader("🏠 Q1 - 4️⃣ Ajustes no Modelo")
    st.info("Ajustes como transformação log, normalização ou remoção de outliers.")

def q1_etapa5():
    st.subheader("🏠 Q1 - 5️⃣ Tomada de Decisão")
    st.info("Descreva aplicações práticas do modelo no contexto de negócio.")

# =============================
# 🔧 Funções - Questão 2
# =============================
def q2_etapa1():
    st.subheader("🏨 Q2 - a) Análise Descritiva dos Dados")
    st.info("Explore variáveis, tipos e comportamento dos cancelamentos.")

def q2_etapa2():
    st.subheader("🏨 Q2 - b) Modelo de Regressão Logística")
    st.info("Treine o modelo e mostre acurácia, precisão, recall e F1.")

def q2_etapa3():
    st.subheader("🏨 Q2 - c) Análise das Features")
    st.info("Use coeficientes ou feature importance.")

def q2_etapa4():
    st.subheader("🏨 Q2 - d) Justificativa do Método")
    st.info("Compare com regressão linear e defenda a escolha da logística.")

# =============================
# 🔧 Funções - Questão 3
# =============================
def q3_etapa1():
    st.subheader("🌍 Q3 - a) Análise Descritiva")
    st.info("Explore dados por país, quantidade e preço.")

def q3_etapa2():
    st.subheader("🌍 Q3 - b) ANOVA entre Países")
    st.info("Apresente F, p-valor e interprete o teste.")

def q3_etapa3():
    st.subheader("🌍 Q3 - c) Ajustes no Modelo")
    st.info("Verifique normalidade, homocedasticidade etc.")

def q3_etapa4():
    st.subheader("🌍 Q3 - d) Interpretação e Decisão")
    st.info("Decisões com base nas diferenças de médias.")

# =============================
# 🔧 Funções - Questão 4
# =============================
def q4_etapa1():
    st.subheader("🛒 Q4 - a) Discussão do Problema")
    st.info("Importância de prever reclamações no varejo.")

def q4_etapa2():
    st.subheader("🛒 Q4 - b) Análise Descritiva")
    st.info("Examine variáveis ligadas à variável 'Complain'.")

def q4_etapa3():
    st.subheader("🛒 Q4 - c) Seleção de Modelos")
    st.info("Compare Logistic, Árvores, Random Forest, XGBoost.")

def q4_etapa4():
    st.subheader("🛒 Q4 - d) SHAP e Explicabilidade")
    st.info("Use SHAP para entender a influência das variáveis.")

def q4_etapa5():
    st.subheader("🛒 Q4 - e) K-Means / DBSCAN")
    st.info("Clusterização e detecção de outliers por perfil.")

def q4_etapa6():
    st.subheader("🛒 Q4 - f) Decisão Estratégica")
    st.info("Sugira melhorias com base nos insights obtidos.")

# =============================
# 🔧 Funções - Questão 5
# =============================
def q5_etapa1():
    st.subheader("📌 Q5 - Em breve")
    st.info("Espaço reservado para uma nova questão.")


# =============================
# ▶️ APP Streamlit
# =============================
st.set_page_config(page_title="📊 Prova Final - Análise Estatística", layout="wide")
st.title("📚 Prova Final - Análise Estatística de Dados e Informações")
st.markdown("👩‍🎓 Desenvolvido por: [Seu Nome] &nbsp;&nbsp;&nbsp;&nbsp;📅 Julho 2025")

# MENU LATERAL
with st.sidebar:
    st.title("🧭 Menu da Prova")
    mostrar_todas = st.checkbox("✅ Mostrar todas as questões", value=False)

    # Questão 1
    with st.expander("🏠 Questão 1 - Regressão Linear (Imóveis)", expanded=mostrar_todas):
        show_q1_e1 = mostrar_todas or st.checkbox("1️⃣ Análise Descritiva", key="q1e1")
        show_q1_e2 = mostrar_todas or st.checkbox("2️⃣ Modelo de Regressão Linear", key="q1e2")
        show_q1_e3 = mostrar_todas or st.checkbox("3️⃣ Interpretação dos Resultados", key="q1e3")
        show_q1_e4 = mostrar_todas or st.checkbox("4️⃣ Ajustes no Modelo", key="q1e4")
        show_q1_e5 = mostrar_todas or st.checkbox("5️⃣ Tomada de Decisão", key="q1e5")

    # Questão 2
    with st.expander("🏨 Questão 2 - Regressão Logística (Reservas)", expanded=mostrar_todas):
        show_q2_e1 = mostrar_todas or st.checkbox("a) Análise Descritiva", key="q2e1")
        show_q2_e2 = mostrar_todas or st.checkbox("b) Modelo de Regressão Logística", key="q2e2")
        show_q2_e3 = mostrar_todas or st.checkbox("c) Análise das Features", key="q2e3")
        show_q2_e4 = mostrar_todas or st.checkbox("d) Justificativa do Método", key="q2e4")

    # Questão 3
    with st.expander("🌍 Questão 3 - ANOVA (Vendas por País)", expanded=mostrar_todas):
        show_q3_e1 = mostrar_todas or st.checkbox("a) Análise Descritiva", key="q3e1")
        show_q3_e2 = mostrar_todas or st.checkbox("b) ANOVA entre Países", key="q3e2")
        show_q3_e3 = mostrar_todas or st.checkbox("c) Ajustes no Modelo", key="q3e3")
        show_q3_e4 = mostrar_todas or st.checkbox("d) Interpretação e Decisão", key="q3e4")

    # Questão 4
    with st.expander("🛒 Questão 4 - Reclamações de Clientes", expanded=mostrar_todas):
        show_q4_e1 = mostrar_todas or st.checkbox("a) Discussão do Problema", key="q4e1")
        show_q4_e2 = mostrar_todas or st.checkbox("b) Análise Descritiva", key="q4e2")
        show_q4_e3 = mostrar_todas or st.checkbox("c) Seleção de Modelos", key="q4e3")
        show_q4_e4 = mostrar_todas or st.checkbox("d) SHAP e Explicabilidade", key="q4e4")
        show_q4_e5 = mostrar_todas or st.checkbox("e) K-Means / DBSCAN", key="q4e5")
        show_q4_e6 = mostrar_todas or st.checkbox("f) Decisão Estratégica", key="q4e6")

    # Questão 5
    with st.expander("📌 Questão 5 - [Reservado]", expanded=mostrar_todas):
        show_q5_e1 = mostrar_todas or st.checkbox("➡️ Em breve", key="q5e1")


# CHAMADAS DAS FUNÇÕES DE CADA ITEM
if show_q1_e1: q1_etapa1()
if show_q1_e2: q1_etapa2()
if show_q1_e3: q1_etapa3()
if show_q1_e4: q1_etapa4()
if show_q1_e5: q1_etapa5()

if show_q2_e1: q2_etapa1()
if show_q2_e2: q2_etapa2()
if show_q2_e3: q2_etapa3()
if show_q2_e4: q2_etapa4()

if show_q3_e1: q3_etapa1()
if show_q3_e2: q3_etapa2()
if show_q3_e3: q3_etapa3()
if show_q3_e4: q3_etapa4()

if show_q4_e1: q4_etapa1()
if show_q4_e2: q4_etapa2()
if show_q4_e3: q4_etapa3()
if show_q4_e4: q4_etapa4()
if show_q4_e5: q4_etapa5()
if show_q4_e6: q4_etapa6()

if show_q5_e1: q5_etapa1()

# Rodapé
st.markdown("---")
st.markdown("🧮 **Prova Final - Ciência de Dados Aplicada**  \n👩‍🏫 Professor(a): [Nome do Professor]  \n📊 Universidade XYZ - 2025")
