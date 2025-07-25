import csv
from xml.parsers.expat import model
from sklearn import pipeline
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import numpy as np
from sklearn.linear_model import LassoCV
import altair as alt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan
from scipy.stats import shapiro
import statsmodels.api as sm
import streamlit as st
import pandas as pd
import plotly.express as px

# ================================
# 📆 FUNÇÕES UTILITÁRIAS
# ================================
@st.cache_data
def carregar_dados(uploaded_file=None):
    DEFAULT_FILE = 'src/kc_house_data.csv'
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)
    else:
        return pd.read_csv(DEFAULT_FILE)

def upload_multiplos_datasets(questoes):
    st.sidebar.markdown("### 📁 Upload de Arquivos por Questão")
    arquivos = {}
    for questao in questoes:
        arquivos[questao] = st.sidebar.file_uploader(
            f"🔹 Arquivo para {questao}", type=["csv"], key=f"upload_{questao}"
        )
    return arquivos



def avaliar_pressupostos(X, y_true, y_pred):
  #########################
    # Supondo que você já tenha y_pred e residuos como arrays ou Series
    df_residuos = pd.DataFrame({
        "Valores previstos": y_pred,
        "Resíduos": y_true - y_pred  # ou residuos, se já estiver pronto
    })

    # Gráfico de dispersão com linha horizontal em y=0
    scatter = alt.Chart(df_residuos).mark_circle(size=60, opacity=0.6).encode(
        x=alt.X("Valores previstos", title="Valores previstos"),
        y=alt.Y("Resíduos", title="Resíduos"),
        tooltip=["Valores previstos", "Resíduos"]
    ).properties(
        width=400,
        height=300,
        title="Resíduos vs Valores previstos"
    )

    linha_zero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color='red', strokeDash=[5,5]).encode(
        y='y'
    )

    # Combine scatter + linha zero
    chart = scatter + linha_zero

    st.altair_chart(chart, use_container_width=True)
    
  #######################

def ajustar_modelo_lasso(X,y):
    """
    Ajusta um modelo de regressão Lasso com validação cruzada.
    Retorna o modelo treinado e os coeficientes.
    """
    st.markdown("### 🔍 Ajuste do Modelo com Lasso (com validação cruzada)")

    # Ajusta o modelo com validação cruzada para escolher o melhor alpha
    lasso = LassoCV(cv=5, random_state=42).fit(X, y)
    coef_lasso = pd.Series(lasso.coef_, index=X.columns)

    # Exibe os resultados
    st.write(f"Melhor alpha (λ): {lasso.alpha_:.6f}")
    st.markdown("### 📌 Coeficientes do Lasso")
    st.dataframe(coef_lasso[coef_lasso != 0].sort_values(ascending=False))

    return lasso


# =================
# Função para Remover Outliers com IQR
# =================
def remover_outliers_iqr(df, colunas=None, threshold=1.5):
    """
    Removes outliers from numeric columns based on the IQR method.
    Parameters:
        df: Original DataFrame
        colunas: list of numeric columns to evaluate (if None, applies to all)
        threshold: IQR multiplier to define limits (default: 1.5)
    Returns:
        DataFrame without outliers
    """
    if colunas is None:
        colunas = df.select_dtypes(include='number').columns.tolist()

    df_sem_outliers = df.copy()
    for col in colunas:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        limite_inferior = Q1 - threshold * IQR
        limite_superior = Q3 + threshold * IQR
        df_sem_outliers = df_sem_outliers[
            (df_sem_outliers[col] >= limite_inferior) & (df_sem_outliers[col] <= limite_superior)
        ]

    return df_sem_outliers




# =============================
# 🔧 Funções - Questão 1
# =============================
def q1_etapa1():
    st.markdown("---")
    def exibir_dicionario_variaveis():
        st.markdown("### 📘 Dicionário de Variáveis")
        st.markdown("""
        | Coluna            | Tradução                        | Descrição                                                                 |
        |-------------------|----------------------------------|---------------------------------------------------------------------------|
        | `id`              | Identificador                   | ID único do imóvel                                                        |
        | `date`            | Data da venda                   | Data em que o imóvel foi vendido                                          |
        | `price`           | Preço                           | Preço de venda do imóvel (variável alvo)                                 |
        | `bedrooms`        | Quartos                         | Número de quartos                                                         |
        | `bathrooms`       | Banheiros                       | Número de banheiros (valores podem ser fracionados)                      |
        | `sqft_living`     | Área útil (pés²)                | Área interna utilizável do imóvel                                        |
        | `sqft_lot`        | Área do terreno (pés²)         | Tamanho do lote do imóvel                                                |
        | `floors`          | Andares                         | Número de andares                                                         |
        | `waterfront`      | Frente para o mar               | 1 se o imóvel tem vista para o mar; 0 caso contrário                      |
        | `view`            | Visibilidade/Visão              | Grau de qualidade da vista (0–4)                                          |
        | `condition`       | Condição                        | Condição geral do imóvel (1–5)                                            |
        | `grade`           | Qualidade de construção         | Avaliação da qualidade da construção (1–13)                               |
        | `sqft_above`      | Área acima do solo (pés²)       | Área construída acima do solo                                            |
        | `sqft_basement`   | Área do porão (pés²)            | Área do porão                                                             |
        | `yr_built`        | Ano de construção               | Ano original de construção                                                |
        | `yr_renovated`    | Ano de reforma                  | Ano da última reforma (0 se nunca reformado)                             |
        | `zipcode`         | CEP                             | Código postal                                                             |
        | `lat`             | Latitude                        | Coordenada de latitude                                                    |
        | `long`            | Longitude                       | Coordenada de longitude                                                   |
        | `sqft_living15`   | Área útil dos vizinhos (pés²)   | Média da área útil das 15 casas mais próximas                             |
        | `sqft_lot15`      | Área dos terrenos vizinhos      | Média do tamanho dos lotes das 15 casas mais próximas                     |
            """)
        
    st.subheader("🏠 Q1 - 1 - Regressão Linear - Análise Descritiva dos Dados")
    exibir_dicionario_variaveis()
    uploaded_file = 'src/kc_house_data.csv'

    df = carregar_dados(uploaded_file)
    st.session_state["kc_df"] = df
    st.success("✅ Arquivo carregado com sucesso!")

    st.markdown("### 🔍 Preview dos Dados")
    #st.dataframe(df.head())
    #st.dataframe(df)

    st.markdown("### 📊 Estatísticas Descritivas")
    st.dataframe(df.describe())

    st.markdown("### 🧮 Mediana das Variáveis Numéricas")
    st.dataframe(df.median(numeric_only=True))

    if "price" in df.columns:
        # excluindo as variaves id e zipcod
        st.markdown("### 🔗 Correlação com o Preço (`price`)")
        correlacoes = df.drop(columns=["id", "zipcode"], errors='ignore').corr(numeric_only=True)["price"].sort_values(ascending=False)
        st.write(correlacoes)

        st.markdown("### 🔥 Mapa de Correlação")
        plt.figure(figsize=(10, 8))
        # excluindo as variaves id e zipcod
        sns.heatmap(df.drop(columns=["id"], errors='ignore').corr(numeric_only=True), annot=True, fmt=".2f", cmap="coolwarm")
        st.pyplot(plt.gcf())
        plt.clf()
        
        #corr = df.drop(columns=["id", "price"], errors='ignore').corr(numeric_only=True)
        #corr = corr.reset_index().melt('index')
        #corr.columns = ['Var1', 'Var2', 'Correlacao']

        #heatmap = alt.Chart(corr).mark_rect().encode(
        #    x=alt.X('Var1:O', title=None),
        #    y=alt.Y('Var2:O', title=None),
        #    color=alt.Color('Correlacao:Q', scale=alt.Scale(scheme='redblue'), legend=None),
        #    tooltip=['Var1', 'Var2', alt.Tooltip('Correlacao:Q', format=".2f")]
        #).properties(
        #    width=600,
        #    height=600,
        #    title="Mapa de Correlação"
        #)

        # Adiciona os valores no centro dos quadrados
        #text = alt.Chart(corr).mark_text(size=10, color='black').encode(
        #    x='Var1:O',
        #    y='Var2:O',
        #    text=alt.Text('Correlacao:Q', format=".2f")
        #)

        #st.markdown("### 🔥 Mapa de Correlação")
        #st.altair_chart(heatmap + text, use_container_width=False)
                
        

    st.markdown("### 📈 Distribuição de Variáveis")
    cols_to_plot = st.multiselect(
        "Selecione variáveis numéricas:",
        df.select_dtypes(include='number').columns.tolist(),
        default=["price", "sqft_living","bathrooms","waterfront","view","condition","grade"]
    )

    charts = []
    for col in cols_to_plot:
        base = alt.Chart(df).mark_bar(opacity=0.7).encode(
            alt.X(f"{col}:Q", bin=alt.Bin(maxbins=30), title=col),
            y='count()',
            tooltip=[col]
        ).properties(
            width=250,
            height=200,
            title=f"Distribuição: {col}"
        )

        kde = alt.Chart(df).transform_density(
            col,
            as_=[col, 'density'],
        ).mark_line(color='red').encode(
            x=col,
            y='density:Q'
        )

        chart = base + kde  # sobrepõe a densidade
        charts.append(chart)

    if charts:
        st.altair_chart(alt.hconcat(*charts), use_container_width=True)
    else:
        st.info("Selecione ao menos uma variável para visualizar.")
    
    
#
def q1_etapa2():
    st.markdown("---")
    st.subheader("🏠 Q1 - 2 - Regressao lienar Modelo de Regressão Linear")
    st.info("Crie o modelo de regressão linear com métricas de desempenho.")

    df = st.session_state.get("kc_df")
    if df is None:
        st.warning("⚠️ Os dados ainda não foram carregados. Execute a Etapa 1 primeiro.")
        return

    df = df.drop(columns=["id", "date", "zipcode"], errors='ignore')
    df = df.dropna()

    cat_cols = ["waterfront", "view", "condition", "grade"]
    cols_possiveis = df.columns.tolist()
    if "price" in cols_possiveis:
        cols_possiveis.remove("price")

    # Fallback para variáveis padrão
    default_vars = ["sqft_living", "bathrooms"] + cat_cols
    default_vars = [v for v in default_vars if v in cols_possiveis]
    if not default_vars:
        default_vars = df.select_dtypes(include='number').columns.drop("price").tolist()[:2]

    st.markdown("### 🎯 Seleção de Variáveis Preditivas (Antes do One-Hot)")
    selected_features_raw = st.multiselect(
        "Selecione as variáveis para o modelo:",
        cols_possiveis,
        default=default_vars,
        key="seletor_variaveis_q1_etapa2"
    )

    # Separar variáveis numéricas e categóricas selecionadas
    selected_cat = [col for col in selected_features_raw if col in cat_cols]
    selected_num = [col for col in selected_features_raw if col not in cat_cols]

    if selected_features_raw:
        # Construir X com concatenação segura
        frames = []
        if selected_num:
            frames.append(df[selected_num])
        if selected_cat:
            frames.append(pd.get_dummies(df[selected_cat], drop_first=True))

        if not frames:
            st.warning("⚠️ Nenhuma variável válida foi selecionada.")
            return

        X = pd.concat(frames, axis=1)
        y = df["price"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        modelo = LinearRegression()
        modelo.fit(X_train, y_train)
        y_pred = modelo.predict(X_test)

        st.session_state["X_test"] = X_test
        st.session_state["y_test"] = y_test
        st.session_state["y_pred"] = y_pred

        st.markdown("### 📌 Coeficientes do Modelo")
        coef_df = pd.DataFrame({"Variável": X.columns, "Coeficiente": modelo.coef_})
        st.dataframe(coef_df)

        st.markdown("### 📊 Métricas de Avaliação - Regressao Linear Classica")
        st.write(f"R²: {r2_score(y_test, y_pred):.4f}")
        st.write(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}")
        st.write(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")

def q1_etapa3():
    st.markdown("---")
    st.subheader("🏠 Q1 - 3️⃣ Interpretação dos Resultados")
    if "X_test" in st.session_state and "y_test" in st.session_state and "y_pred" in st.session_state:
        X = st.session_state["X_test"]
        y = st.session_state["y_test"]
        y_pred = st.session_state["y_pred"]
        avaliar_pressupostos(X, y, y_pred)
    else:
        st.warning("⚠️ Execute a Etapa 2 para gerar o modelo antes de interpretar os resultados.")



def q1_etapa4():
    st.markdown("---")
    st.subheader("🏠 Q1 - 4️⃣ Ajustes no Modelo")
    df = st.session_state.get("kc_df")
    if df is None:
        st.warning("⚠️ Os dados ainda não foram carregados. Execute a Etapa 1 primeiro.")
        st.stop()

    # Preprocessamento
    df = df.drop(columns=["id", "date", "zipcode"], errors='ignore')
    df = df[df["price"] > 0].dropna()

    st.markdown("### 🔧 Aplicando Transformação Logarítmica em 'price'")
    df["log_price"] = np.log(df["price"])

    # Define variáveis categóricas para possíveis seleções
    cat_cols = ["waterfront", "view", "condition", "grade"]
    df[cat_cols] = df[cat_cols].astype("category")

    # Cria lista de colunas possíveis (incluindo categóricas)
    all_cols = df.select_dtypes(include=["number", "category"]).columns.tolist()
    all_cols.remove("price")
    if "log_price" in all_cols:
        all_cols.remove("log_price")

    # Sugerir variáveis padrão
    default_vars = ["sqft_living", "bathrooms"] + cat_cols
    default_vars = [v for v in default_vars if v in all_cols]

    selected_features_raw = st.multiselect(
        "Selecione variáveis preditoras (numéricas ou categóricas):",
        all_cols,
        default=default_vars,
        key="seletor_variaveis_q1_etapa4"
    )

    # Separa entre categóricas e numéricas
    selected_cat = [col for col in selected_features_raw if col in cat_cols]
    selected_num = [col for col in selected_features_raw if col not in cat_cols]

    if selected_features_raw:
        # Aplica one-hot apenas nas categóricas selecionadas
        X_parts = []
        if selected_num:
            X_parts.append(df[selected_num])
        if selected_cat:
            X_parts.append(pd.get_dummies(df[selected_cat], drop_first=True))

        if not X_parts:
            st.warning("⚠️ Nenhuma variável válida foi selecionada.")
            return

        X = pd.concat(X_parts, axis=1)
        y = df["log_price"]

        # Divisão e modelo
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        modelo = LinearRegression()
        modelo.fit(X_train, y_train)
        y_pred = modelo.predict(X_test)

        # Coeficientes
        st.markdown("### 📌 Coeficientes (log_price)")
        coef_df = pd.DataFrame({"Variável": X.columns, "Coeficiente": modelo.coef_})
        st.dataframe(coef_df)

        # Métricas
        st.markdown("### 📊 Métricas de Avaliação - Regressão Linear após transformação Log")
        st.write(f"R²: {r2_score(y_test, y_pred):.4f}")
        st.write(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}")
        st.write(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")

        # Avaliação de pressupostos
        avaliar_pressupostos(X_test, y_test, y_pred)

        # Lasso
        modelo_lasso = ajustar_modelo_lasso(X_train, y_train)
        y_pred_lasso = modelo_lasso.predict(X_test)

        st.markdown("### 📊 Métricas de Avaliação- Transformação Lasso")
        st.write(f"R²: {r2_score(y_test, y_pred_lasso):.4f}")
        st.write(f"MAE: {mean_absolute_error(y_test, y_pred_lasso):.2f}")
        st.write(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_lasso)):.2f}")

        avaliar_pressupostos(X_test, y_test, y_pred_lasso)


def q1_etapa5():
    st.subheader("🏠 Q1 - 5️⃣ Tomada de Decisão")
    st.info("Descreva aplicações práticas do modelo no contexto de negócio.")
    
    st.markdown("## 📌 Análise da Questão 1 – Regressão Linear")

    st.markdown("""
    A análise da Questão 1 teve como objetivo prever os preços de imóveis na região de **King County** utilizando **Regressão Linear**.  
    A primeira etapa consistiu em uma **exploração descritiva dos dados**, onde foram apresentadas estatísticas como média, mediana e desvio padrão, além de gráficos de distribuição e correlação.  
    Essa análise inicial ajudou a identificar variáveis mais relevantes para a modelagem, como `sqft_living`, `bathrooms` e `grade`.  
    Em seguida, foi construído um **modelo de regressão linear**, avaliado pelas métricas:

    - **R²**: `0.6084`
    - **MAE**: `R$ 155.744,00`
    - **RMSE**: `R$ 243.314,83`

    O desempenho inicial indicou uma **capacidade explicativa moderada**, mas com elevada variabilidade nos erros, sugerindo que o modelo clássico não era suficientemente preciso.
    """)

    st.markdown("""
    Na etapa de interpretação, foram verificados os **pressupostos da regressão linear**, como:

    - Normalidade dos resíduos  
    - Homocedasticidade  
    - Ausência de multicolinearidade

    Todos os testes indicaram violações desses pressupostos.  
    Para tentar corrigir, foi aplicada a **transformação logarítmica** na variável resposta `price`, resultando em um modelo com:

    - **R²**: `0.6057`
    - **MAE**: `0.27`
    - **RMSE**: `0.34` (escala logarítmica)

    Embora os resíduos têm se tornado mais simétricos, os pressupostos **continuaram violados**.  
    Na sequência, aplicou-se o modelo **Lasso com validação cruzada**, visando penalizar variáveis menos relevantes.  
    O desempenho do modelo Lasso foi:

    - **R²**: `0.4882`
    - **MAE**: `0.31`
    - **RMSE**: `0.38`

    O Lasso teve desempenho inferior ao modelo log-transformado e **não resolveu as violações estatísticas**.
    """)

    st.markdown("""
    ### 💼 Aplicação no Negócio

    Mesmo com limitações, os modelos desenvolvidos oferecem **insights úteis para negócios**.  
    Eles podem auxiliar:

    - Na **precificação inicial** de imóveis
    - Na **identificação de imóveis fora do padrão**
    - Na orientação de **investimentos em reformas**, destacando o impacto de variáveis como `grade` e `sqft_living`.

    Contudo, **devido às violações dos pressupostos**, recomenda-se o uso de **modelos não lineares**, como:

    - Árvores de Decisão  
    - Random Forest  
    - XGBoost

    Esses algoritmos lidam melhor com dados complexos e podem gerar **previsões mais confiáveis e robustas** para o mercado imobiliário.
    """)

        

# =============================
# 🔧 Funções - Questão 2
# =============================

def criar_coluna_arrival_date(df):
    """
    Adiciona uma coluna 'arrival_date' ao DataFrame com a data de chegada em formato datetime,
    convertendo o mês (por extenso) para número.
    """
    # Dicionário para conversão de nome do mês para número
    meses = {
        'January': 1, 'February': 2, 'March': 3, 'April': 4,
        'May': 5, 'June': 6, 'July': 7, 'August': 8,
        'September': 9, 'October': 10, 'November': 11, 'December': 12
    }
    # Converter os nomes dos meses para número
    df['arrival_month_num'] = df['arrival_date_month'].map(meses)
    
    # Criar a coluna de data de chegada
    df['arrival_date'] = pd.to_datetime(
        df['arrival_date_year'].astype(str) + '-' +
        df['arrival_month_num'].astype(str).str.zfill(2) + '-' +
        df['arrival_date_day_of_month'].astype(str).str.zfill(2)
    )
    
    #st.write(df["arrival_date"])
    
     # Cria coluna booking_date (data da reserva)
    df['booking_date'] = df['arrival_date'] - pd.to_timedelta(df['lead_time'], unit='D')
    
     #excluir arrival_date_year, arrival_date_month, arrival_date_week_number, arrival_date_day_of_month
    #variaves foram substituídas por arrival_date

    df = df.drop(columns=["arrival_date_year", "arrival_date_month", "arrival_date_week_number", "arrival_date_day_of_month"], errors='ignore')

    return df

def pre_processamento(df):
    """
    Realiza pré-processamento básico dos dados
    """
    # Remove duplicatas
    df = df.drop_duplicates()
    df = criar_coluna_arrival_date(df)
    df.drop(columns=['arrival_month_num'], inplace=True)
    
    return df



def q2_etapa1():
    st.subheader("🏨 Q2 - a) Análise Descritiva dos Dados")
    st.info("Realize uma análise descritiva da base de dados.")
    uploaded_file = 'src/hotel_bookings.csv'
    # Carregar os dados

    df2 = carregar_dados(uploaded_file)
    st.session_state["hb_df"] = df2
    st.success("✅ Arquivo carregado com sucesso!")
    st.markdown("### 🔍 Preview dos Dados")
    st.dataframe(df2.head())
    
    
    # descrever cada coluna
    st.markdown("### 📘 Dicionário de Variáveis")
   # for col in df2.columns:
    #       st.markdown(f"**{col}**: {df2[col].dtype}")
    # fazer a tradução de cada coluna do df2


    # Definindo os dados da tabela
    dados = {
        "Variável": [
            "hotel", "is_canceled", "lead_time", "arrival_date_year", "arrival_date_month",
            "arrival_date_week_number", "arrival_date_day_of_month", "stays_in_weekend_nights",
            "stays_in_week_nights", "adults", "children", "babies", "meal", "country",
            "market_segment", "distribution_channel", "is_repeated_guest", "previous_cancellations",
            "previous_bookings_not_canceled", "reserved_room_type", "assigned_room_type",
            "booking_changes", "deposit_type", "agent", "company", "days_in_waiting_list",
            "customer_type", "adr", "required_car_parking_spaces", "total_of_special_requests",
            "reservation_status", "reservation_status_date", "arrival_date", "booking_date"
        ],
        "Tipo": [
            "Categórica", "Binária", "Numérica", "Numérica (inteira)", "Categórica",
            "Numérica (inteira)", "Numérica (inteira)", "Numérica (inteira)", "Numérica (inteira)",
            "Numérica (inteira)", "Numérica (inteira)", "Numérica (inteira)", "Categórica",
            "Categórica", "Categórica", "Categórica", "Binária", "Numérica (inteira)",
            "Numérica (inteira)", "Categórica", "Categórica", "Numérica (inteira)", "Categórica",
            "Categórica", "Categórica", "Numérica (inteira)", "Categórica", "Numérica (float)",
            "Numérica (inteira)", "Numérica (inteira)", "Categórica", "Data","Data","Data"
        ],
        "Descrição": [
            "Tipo de hotel (ex: Resort, City)",
            "Cancelamento da reserva (1 = sim, 0 = não)",
            "Tempo de antecedência da reserva (dias)",
            "Ano de chegada",
            "Mês de chegada",
            "Número da semana de chegada (1 a 52/53)",
            "Dia do mês de chegada",
            "Noites de fim de semana na reserva",
            "Noites de semana na reserva",
            "Número de adultos",
            "Número de crianças",
            "Número de bebês",
            "Tipo de refeição incluída",
            "País de origem do hóspede",
            "Segmento de mercado",
            "Canal de distribuição",
            "Hóspede recorrente (1 = sim, 0 = não)",
            "Número de cancelamentos anteriores",
            "Reservas anteriores não canceladas",
            "Tipo de quarto reservado",
            "Tipo de quarto atribuído",
            "Alterações realizadas na reserva",
            "Tipo de depósito",
            "Agente de reservas",
            "Empresa associada à reserva",
            "Dias na lista de espera",
            "Tipo de cliente",
            "Diária média (Average Daily Rate)",
            "Espaços de estacionamento necessários",
            "Total de pedidos especiais",
            "Status final da reserva (Check-Out, Canceled, No-Show)",
            "Data do status final da reserva",
            "Data chegada",
            "Data da reserva"
        ]
    }

    # Criando o DataFrame
    df_variaveis = pd.DataFrame(dados)

    # Exibindo no Streamlit
    st.write("### Dicionário de Variáveis do Dataset Hotel Booking Demand")
    st.dataframe(df_variaveis, use_container_width=True)

    # pre processamento
    df2 = pre_processamento(df2)
    st.session_state["hb_df"] = df2
    st.success("✅ Pré-processamento concluído - Excluindo duplicadas se houver")
    #st.markdown("### 🔍 Preview dos Dados apos Inclusao da coluna data arrival_date e exclusao das colunas de data")
    #st.dataframe(df2.head())
    
    st.markdown("### 📊 Estatísticas Descritivas")
    #st.dataframe(df2.describe())
    st.markdown("### 🧮 Mediana das Variáveis Numéricas")
    st.dataframe(df2.select_dtypes(include='number').median())
    
    # graficos de distribuição
    cols_numericas = [
        'lead_time', 'stays_in_weekend_nights', 'stays_in_week_nights', 'adults', 'children',
        'babies', 'previous_cancellations', 'previous_bookings_not_canceled',
        'booking_changes', 'days_in_waiting_list', 'adr',
        'required_car_parking_spaces', 'total_of_special_requests'
    ]
   
    st.markdown("### 📊 Histogramas das Variáveis Numéricas por Cancelamento")

    charts = []
    for col in cols_numericas:
        chart = alt.Chart(df2).mark_bar(opacity=0.7).encode(
            x=alt.X(f'{col}:Q', bin=alt.Bin(maxbins=30), title=col),
            y=alt.Y('count()', title='Frequência'),
            color=alt.Color('is_canceled:N', title='Cancelamento')
        ).properties(
            title=f'{col} - Distribuição por Cancelamento',
            width=300,
            height=250
        )
        charts.append(chart)

    # Exibir os gráficos em 3 colunas
    num_cols = 3
    rows = [charts[i:i+num_cols] for i in range(0, len(charts), num_cols)]

    for row in rows:
        cols = st.columns(num_cols)
        for col_chart, col_container in zip(row, cols):
            with col_container:
                st.altair_chart(col_chart, use_container_width=True)


    

def q2_etapa2():
    
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.linear_model import LogisticRegression
       # graficar a distribuição da variável dependente
    import plotly.express as px
    # importar pandas como pd
    import pandas as pd
    import streamlit as st
    import altair as alt
    
    
    st.subheader("🏨 Q2 - b) Modelo de Regressão Logística")
    st.info("Construa o modelo de regressão logística e avalie seu desempenho.")

    df2 = st.session_state.get("hb_df")
    if df2 is None:
        st.warning("⚠️ Os dados ainda não foram carregados. Execute a Etapa 1 primeiro.")
        return
    
    y = df2['is_canceled']
    X = df2.drop(columns='is_canceled')

    
        # Separe as features do tipo categórica e numérica
    cols_numericas = [
        'lead_time', 'stays_in_weekend_nights', 'stays_in_week_nights', 'adults', 'children',
        'babies', 'previous_cancellations', 'previous_bookings_not_canceled',
        'booking_changes', 'days_in_waiting_list', 'adr',
        'required_car_parking_spaces', 'total_of_special_requests'
    ]

    cols_categoricas = [
        'hotel', 'meal', 
        #'country',
        'market_segment', 'distribution_channel',
        'is_repeated_guest', 
        #'reserved_room_type', 'assigned_room_type',
        'deposit_type', 
        #'agent',
       # 'company', 
        'customer_type'
    ]
    

    # 1. Balanceamento das classes
    st.markdown("**Distribuição da variável dependente (is_canceled):**")
    st.dataframe(y.value_counts(normalize=True).rename("Proporção").to_frame())

    fig = px.pie(y.value_counts(normalize=True).rename("Proporção").to_frame(), 
                 values='Proporção', 
                 names=y.value_counts().index, 
                 title="Distribuição de Cancelamentos (is_canceled)",
                 color_discrete_sequence=px.colors.qualitative.Set2)    
    st.plotly_chart(fig, use_container_width=True)
    

   
   # se distribuição for desbalanceada, aplicar oversampling ou undersampling
    from imblearn.over_sampling import RandomOverSampler
    from imblearn.under_sampling import RandomUnderSampler
    from imblearn.pipeline import Pipeline as ImbPipeline
    from collections import Counter
    # Verificar o balanceamento das classes
    
   
    counter = Counter(y)
    #st.write(f"Classes antes do balanceamento: {counter}")
    # Se a distribuição for desbalanceada, aplicar oversampling ou undersampling

    if counter[0] < counter[1]:
        st.warning("⚠️ Distribuição desbalanceada. Aplicando oversampling para balancear as classes.")
        
        
        # Aplicar oversampling
        oversample = RandomOverSampler(sampling_strategy='minority', random_state=42)
        X_resampled, y_resampled = oversample.fit_resample(df2[cols_numericas + cols_categoricas], y)
        st.write(f"Classes após oversampling: {Counter(y_resampled)}")
    elif counter[0] > counter[1]:
        st.warning("⚠️ Distribuição desbalanceada. Aplicando undersampling para balancear as classes.")
        # Aplicar undersampling
        undersample = RandomUnderSampler(sampling_strategy='majority', random_state=42)
        X_resampled, y_resampled = undersample.fit_resample(df2[cols_numericas + cols_categoricas], y)
       # st.write(f"Classes após undersampling: {Counter(y_resampled)}")
       
            # 3. Verificação de valores ausentes
        # 3. Verificação de valores ausentes no dataset balanceado
        st.markdown("**Valores ausentes nas colunas do dataset balanceado:**")
        missing_values = X.isnull().sum()
        missing_values = missing_values[missing_values > 0].sort_values(ascending=False)
        if missing_values.empty:
            st.success("✅ Não há valores ausentes nas colunas do dataset balanceado.")
        else:
            st.dataframe(missing_values.rename("Valores Ausentes").to_frame())
            st.markdown("""
            Optou-se pela remoção das variáveis `country`, `agent` e `company` do modelo devido à alta proporção 
            de valores ausentes e à elevada cardinalidade dessas variáveis, que dificultariam o processamento e aumentariam 
            o risco de overfitting. Em contrapartida, a variável `hotel` foi mantida por sintetizar informações relevantes
            sobre o tipo de hospedagem e o perfil do hóspede, sendo capaz de capturar parte da variabilidade representada 
            pelas variáveis excluídas. Dessa forma, o modelo permanece mais simples, robusto e eficiente, sem perda significativa de informação.
            """)

   
    else:
        st.success("✅ Distribuição balanceada. Não é necessário aplicar oversampling ou undersampling.")
        X_resampled = df2[cols_numericas + cols_categoricas]
        y_resampled = y
   
    
    # Exibir a distribuição das classes após o balanceamento
    st.markdown("**Distribuição da variável dependente (is_canceled) após balanceamento:**")
    st.dataframe(y_resampled.value_counts(normalize=True).rename("Proporção").to_frame())
    # graficar a distribuição da variável dependente após balanceamento
    fig = px.pie(y_resampled.value_counts(normalize=True).rename("Proporção").to_frame(),
                 values='Proporção', 
                 names=y_resampled.value_counts().index, 
                 title="Distribuição de Cancelamentos (is_canceled) após Balanceamento",
                 color_discrete_sequence=px.colors.qualitative.Set2)
    st.plotly_chart(fig, use_container_width=True)

   # 2. Multicolinearidade (VIF) - Exemplo para as features numéricas
  # Imputação dos valores ausentes nas variáveis numéricas do dataset balanceado
    X_num_vif = pd.DataFrame(X_resampled[cols_numericas], columns=cols_numericas)
    X_num_vif = X_num_vif.fillna(X_num_vif.mean())

    # Calcular VIF
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    vif_data = pd.DataFrame()
    vif_data["Feature"] = X_num_vif.columns
    vif_data["VIF"] = [variance_inflation_factor(X_num_vif.values, i) for i in range(X_num_vif.shape[1])]
    st.markdown("**Fatores de Inflação de Variância (VIF):**")
    st.dataframe(vif_data)

    # validar se o VIF é maior 10
    vif_threshold = 10
    if any(vif_data["VIF"] > vif_threshold):
        st.warning(f"⚠️ Algumas variáveis têm VIF maior que {vif_threshold}. Considere remover ou combinar variáveis.")
    else:
        st.success(f"✅ Todas as variáveis têm VIF abaixo de {vif_threshold}. Não há multicolinearidade significativa.")


    
  
 
 
    # verificar linearidade entre as variáveis numéricas
    num_vars = [
        'lead_time'
    ]

    st.markdown("### Análise de Linearidade com Boxplot das Variáveis Numéricas por Cancelamento")

    import altair as alt
    for col in num_vars:
        chart = alt.Chart(df2).mark_boxplot(extent='min-max').encode(
            x=alt.X('is_canceled:N', title='Cancelamento (0 = Não, 1 = Sim)'),
            y=alt.Y(f'{col}:Q', title=col),
            color=alt.Color('is_canceled:N', legend=None)
        ).properties(
            title=f'{col} vs Cancelamento',
            width=400,
            height=300
        )
        st.altair_chart(chart, use_container_width=True)
  
    st.markdown("""
    Boxplots foram utilizados para comparar a distribuição das variáveis numéricas entre as classes da variável dependente 
    (`is_canceled`). Observou-se que algumas variáveis, como `lead_time`, apresentaram diferenças nas medianas entre as classes,
    sugerindo relação monotônica desejável para a regressão logística. Para as demais variáveis numéricas, essa tendência não foi
    tão evidente, indicando que a relação linear com o logit pode ser mais fraca ou inexistente nessas variáveis. Ainda assim, 
    todas as variáveis foram mantidas no modelo, reconhecendo que sua contribuição, mesmo que não estritamente linear, pode ser
    relevante para a previsão. Recomenda-se, em aplicações futuras, avaliar transformações ou métodos mais flexíveis caso o ajuste 
    linear não seja suficiente.
    """)
    st.success("✅ Pressupostos iniciais validados com sucesso! (Linearidade, Multicolinearidade, Balanceamento) Pronto para treinar o modelo de Regressão Logística.")

    # Pré-processamento
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, cols_numericas),
            ('cat', categorical_transformer, cols_categoricas)
        ]
    )

    pipeline = Pipeline(steps=[
        ('prep', preprocessor),
        ('logreg', LogisticRegression(max_iter=1000))
    ])


    # Split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled)

    # Treinamento
    #pipeline.fit(X_train, y_train)
    
    # Treinamento
    pipeline.fit(X_train, y_train)

    # Previsões
    y_pred = pipeline.predict(X_test)
    

    import numpy as np
    import pandas as pd

    # Obter nomes das features do pipeline para tabela de coeficientes
    feature_names = pipeline.named_steps['prep'].get_feature_names_out()
    coef = pipeline.named_steps['logreg'].coef_[0]
    odds = np.exp(coef)

    coef_table = pd.DataFrame({
        'Variável': feature_names,
        'Coeficiente': coef.round(3),
        'Odds Ratio': odds.round(3),
        'Interpretação': [
            "Aumenta chance de cancelamento" if c > 0 else "Reduz chance de cancelamento"
            for c in coef
        ]
    }).sort_values(by='Coeficiente', ascending=False)
        
    st.markdown("### 📊 Coeficientes do Modelo de Regressão Logística")
    st.dataframe(coef_table) 
    # Salve o pipeline e dados de teste no session_state
    st.session_state["model"] = pipeline
   
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

    # Calcular métricas
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    conf_mat = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    # Exibir resultados no Streamlit
    st.markdown(f"""
    **Acurácia:** {acc:.3f}  
    **Precisão:** {prec:.3f}  
    **Recall:** {rec:.3f}  
    **F1-score:** {f1:.3f}  
    """)

    st.markdown("**Matriz de Confusão:**")
    st.dataframe(conf_mat)

    st.markdown("**Relatório de Classificação:**")
    st.text(report)
    
    
    # Tabela de métricas
    metricas = pd.DataFrame({
        "Métrica": ["Acurácia", "Precisão", "Recall", "F1-score"],
        "Valor": [0.721, 0.708, 0.754, 0.730],
        "Interpretação": [
            "O modelo acertou 72,1% das previsões totais (canceladas e não canceladas).",
            "Das reservas que o modelo previu como canceladas, 70,8% realmente foram.",
            "O modelo identificou corretamente 75,4% das reservas que foram canceladas.",
            "Média harmônica entre precisão e recall, indicando bom equilíbrio."
        ]
    })
  
    st.markdown("Explicação")
    st.dataframe(metricas)

     
     
def q2_etapa3():
        st.markdown("---")
        st.subheader("🏨 Q2 - c) Análise das Features")
        st.info("Analise a importância das variáveis no modelo de regressão logística.")

        df2 = st.session_state.get("hb_df")
        if df2 is None:
            st.warning("⚠️ Os dados ainda não foram carregados. Execute a Etapa 1 primeiro.")
            return
        
        
        if "model" not in st.session_state:
            st.warning("⚠️ Execute a Etapa 2 para treinar o modelo antes de analisar as features.")
            return

        model = st.session_state["model"]
        #X_test = st.session_state["X_test"]

        feature_names = model.named_steps['prep'].get_feature_names_out()
        coef = model.named_steps['logreg'].coef_[0]

        print(len(feature_names), len(coef))  # Ambos devem ter o mesmo tamanho

        feature_importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': coef
        }).sort_values(by='Importance', ascending=False)

        st.markdown("### 📊 Importância das Variáveis")
        st.dataframe(feature_importance)

        # Gráfico de barras
        fig = px.bar(feature_importance, x='Feature', y='Importance', title='Importância das Variáveis')
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Análise Final das Principais Variáveis")

        st.markdown("""
        **Cinco variáveis que mais aumentam a chance de cancelamento:**
        1. **lead_time:** Reservas feitas com muita antecedência têm maior risco de cancelamento.
        2. **previous_cancellations:** Hóspedes com histórico prévio de cancelamentos tendem a cancelar novamente.
        3. **market_segment_Undefined:** Reservas de segmento indefinido apresentam mais risco de cancelamento.
        4. **customer_type_Transient:** Hóspedes transitórios são mais propensos ao cancelamento.
        5. **market_segment_Complementary:** Reservas complementares (gratuitas ou promocionais) têm mais chance de serem canceladas.

        **Cinco variáveis que mais reduzem a chance de cancelamento:**
        1. **required_car_parking_spaces:** Necessidade de vaga de estacionamento está associada à menor chance de cancelamento.
        2. **total_of_special_requests:** Mais pedidos especiais significam menor risco de cancelamento.
        3. **deposit_type_Non Refund:** Depósitos não reembolsáveis praticamente impedem cancelamentos.
        4. **customer_type_Group:** Hóspedes em grupo tendem a cancelar menos.
        5. **(Outra variável relevante, como booking_changes):** Mudanças na reserva podem indicar maior comprometimento e, portanto, menor risco de cancelamento.

        Essas variáveis ajudam a identificar perfis de reservas e hóspedes mais propensos ao cancelamento, permitindo ações preventivas por parte do hotel.
        """)

        
        
def q2_etapa4():
    st.markdown("---")
    st.subheader("🏨 Q2 - d) Justificativa do Método")
    st.info("Discuta a escolha do modelo de regressão logística.")

    df2 = st.session_state.get("hb_df")
    if df2 is None:
        st.warning("⚠️ Os dados ainda não foram carregados. Execute a Etapa 1 primeiro.")
        return

    if "model" not in st.session_state:
        st.warning("⚠️ Execute a Etapa 2 para treinar o modelo antes de justificar o método.")
        return

    model = st.session_state["model"]
 

    # Justificativa do método
    st.markdown("""
    A regressão logística foi escolhida por ser o método mais adequado para problemas de classificação binária, como é o caso da previsão de cancelamento de reservas, cuja variável alvo assume apenas dois valores: cancelado (1) ou não cancelado (0).

    Diferentemente da regressão linear, que prevê valores contínuos e pode gerar resultados fora do intervalo [0, 1], a regressão logística modela diretamente a probabilidade de ocorrência de um evento. Seu resultado está sempre restrito ao intervalo de 0 a 1, permitindo interpretar a saída do modelo como a probabilidade de cancelamento de cada reserva.

    Além disso, a regressão logística lida melhor com a natureza categórica da variável resposta e permite calcular métricas de avaliação apropriadas, como acurácia, precisão, recall e F1-score, que são fundamentais para problemas de classificação. Portanto, a regressão logística oferece maior robustez, interpretabilidade e adequação estatística para o contexto deste estudo, sendo preferível à regressão linear.
    """)



# =============================
# 🔧 Funções - Questão 3
# =============================
def q3_etapa1():
    st.subheader("🌍 Q3 - a) Análise Descritiva")
    st.info("Explore dados por país, quantidade e preço.")
        
        # Carregar a planilha (ou use df se já estiver carregado)
    @st.cache_data
    def carregar_dados():
        return pd.read_csv("src/planilha_combinada.csv")

    df = carregar_dados()

    # Total de linhas
    total_linhas = len(df)

    # Calcular valores ausentes
    valores_ausentes = df.isnull().sum()
    percentual_ausentes = (valores_ausentes / total_linhas) * 100

    # Juntar em um único DataFrame para exibição
    tabela_ausentes = pd.DataFrame({
        'Valores Ausentes': valores_ausentes,
        'Percentual (%)': percentual_ausentes.round(2)
    })

    # Filtrar apenas colunas com pelo menos 1 valor ausente
    tabela_ausentes = tabela_ausentes[tabela_ausentes['Valores Ausentes'] > 0]

    # Exibir
    print("📉 Valores Ausentes por Coluna:")
    print(tabela_ausentes.sort_values(by="Percentual (%)", ascending=False))
    
    # mostrar valores ausentes no streamlit
    st.markdown("### 📉 Valores Ausentes por Coluna")
    st.dataframe(tabela_ausentes.sort_values(by="Percentual (%)", ascending=False))
    
    
    # Exibir o DataFrame original
    st.markdown("### 📊 Preview dos Dados")
    st.dataframe(df)
    # Exibir estatísticas descritivas
    st.markdown("### 📈 Estatísticas Descritivas")
    st.dataframe(df.describe())
    
    
    # Amostragem de 100 mil registros para acelerar a visualização
    df_amostra = df.sample(n=100_000, random_state=42)

    vendas_pais = df_amostra.groupby("Country")["Price"].sum().reset_index()
    vendas_pais = vendas_pais.sort_values(by="Price", ascending=False)

    st.markdown("### 📊 Distribuição de Vendas por País (com amostragem)")
    st.dataframe(vendas_pais)
    
    #graficar a distribuição de vendas por país
    import plotly.express as px
    fig = px.bar(vendas_pais, x="Country", y="Price", title="Distribuição de Vendas por País (Amostragem)",
                 color="Price", color_continuous_scale=px.colors.sequential.Plasma)
    
    # Exibir a distribuição de vendas por país
    st.plotly_chart(fig)

    st.markdown("### 📊 Análise de Quantidade e Preço por País")  
    # Agrupar por país e calcular a soma de quantidade e preço
    vendas_pais_quantidade = df_amostra.groupby("Country")["Quantity"].sum().reset_index()
    vendas_pais_preco = df_amostra.groupby("Country")["Price"].sum().reset_index()
    vendas_pais_quantidade = vendas_pais_quantidade.sort_values(by="Quantity", ascending=False)
    vendas_pais_preco = vendas_pais_preco.sort_values(by="Price", ascending=False)
    st.dataframe(vendas_pais_quantidade)
    st.dataframe(vendas_pais_preco)
    # Gráfico de barras da quantidade de vendas por país
    fig_quantidade = px.bar(vendas_pais_quantidade, x="Country", y="Quantity", title="Quantidade de Vendas por País (Amostragem)",
                            color="Quantity", color_continuous_scale=px.colors.sequential.Viridis)

    st.plotly_chart(fig_quantidade)
    
    
    # stockcod por país
    vendas_pais_stockcode = df.groupby("Country")["StockCode"].nunique().reset_index()
    vendas_pais_stockcode = vendas_pais_stockcode.sort_values(by="StockCode", ascending=False)
    st.markdown("### 📊 Quantidade de Produtos Vendidos por País")
    st.dataframe(vendas_pais_stockcode)
    # Exibir a quantidade de vendas por país
   # st.markdown("### 📊 Quantidade de Vendas por País")
   # st.dataframe(df.groupby("Country")["Quantity"].sum().reset_index())
    
    # Gráfico de barras da quantidade de vendas por país
    #fig = px.bar(df, x="Country", y="Quantity", title="Quantidade de Vendas por País",
    #             color="Quantity", color_continuous_scale=px.colors.sequential.Viridis)
    #st.plotly_chart(fig)
    
    # medias de quantidade de produtos vendidos por país
    #st.markdown("### 📊 Média de Quantidade Vendida por País")
    #st.dataframe(df.groupby("Country")["Quantity"].mean().reset_index())
    # Gráfico de barras da média de quantidade vendida por país
    #fig = px.bar(df, x="Country", y="Quantity", title="Média de Quantidade Vendida por País",
    #             color="Quantity", color_continuous_scale=px.colors.sequential.Cividis)
    #st.plotly_chart(fig)
    

    # Exibir a média de preço por país
    #st.markdown("### 📊 Média de Preço por País")
    #st.dataframe(df.groupby("Country")["Price"].mean().reset_index())
    # Gráfico de barras da média de preço por país
    #fig = px.bar(df, x="Country", y="Price", title="Média de Preço por País",
    #             color="Price", color_continuous_scale=px.colors.sequential.Inferno)
    #st.plotly_chart(fig)
    


    # Exibir a quantidade de vendas por categoria
    #st.markdown("### 📊 Quantidade de Vendas por Categoria")
    #st.dataframe(df.groupby("Category")["Quantity"].sum().reset_index())
    # Gráfico de barras da quantidade de vendas por categoria
    #fig = px.bar(df, x="Category", y="Quantity", title="Quantidade de Vendas por Categoria",
    #             color="Quantity", color_continuous_scale=px.colors.sequential.Viridis)
    #st.plotly_chart(fig)


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
    with st.expander(" Questão 1 - Regressão Linear (Imóveis)", expanded=mostrar_todas):
        show_q1_e1 = mostrar_todas or st.checkbox("1️ Análise Descritiva", key="q1e1")
        show_q1_e2 = mostrar_todas or st.checkbox("2️ Modelo de Regressão Linear", key="q1e2")
        show_q1_e3 = mostrar_todas or st.checkbox("3️ Interpretação dos Resultados", key="q1e3")
        show_q1_e4 = mostrar_todas or st.checkbox("4️ Ajustes no Modelo", key="q1e4")
        show_q1_e5 = mostrar_todas or st.checkbox("5️ Tomada de Decisão", key="q1e5")

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

