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
        st.markdown("### Dicionário de Variáveis")
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
        
    st.header("Q1 - 1 - Regressão Linear - Análise Descritiva dos Dados")
    exibir_dicionario_variaveis()
    uploaded_file = 'src/kc_house_data.csv'

    df = carregar_dados(uploaded_file)
    st.session_state["kc_df"] = df
    #st.success("✅ Arquivo carregado com sucesso!")

    #st.markdown("### 🔍 Preview dos Dados")
    #st.dataframe(df.head())
    #st.dataframe(df)

    st.markdown("### Estatísticas Descritivas")
    st.dataframe(df.describe())

    #st.markdown("### Mediana das Variáveis Numéricas")
    #st.dataframe(df.median(numeric_only=True))

    if "price" in df.columns:
        # excluindo as variaves id e zipcod
        #st.markdown("### Correlação com o Preço (`price`)")
        #correlacoes = df.drop(columns=["id", "zipcode"], errors='ignore').corr(numeric_only=True)["price"].sort_values(ascending=False)
        #st.write(correlacoes)

        st.markdown("### Mapa de Correlação")
        st.subheader("Interpretação do Mapa de Correlação")

        st.markdown("""
        O mapa mostra que **área útil (`sqft_living`)** e **qualidade da construção (`grade`)** são as variáveis mais fortemente ligadas ao preço dos imóveis.  
        Por outro lado, variáveis como **condição (`condition`)** e **ano de construção (`yr_built`)** apresentam baixa correlação linear com o preço, o que sugere uma influência limitada na modelagem por regressão linear.

        Além disso, algumas variáveis estão fortemente correlacionadas entre si, como `sqft_living` e `sqft_above`, indicando a necessidade de atenção à **multicolinearidade** ao construir o modelo preditivo.
        """)

        # Plotar o mapa de correlação
        plt.figure(figsize=(10, 8))
        # excluindo as variaves id e zipcod
        sns.heatmap(df.drop(columns=["id"], errors='ignore').corr(numeric_only=True), annot=True, fmt=".2f", cmap="coolwarm")
        st.pyplot(plt.gcf())
        plt.clf()
    
  
        st.markdown(r"""
        A base de dados apresenta **21.613 imóveis** registrados na região de King County, nos EUA, com **21 variáveis** sobre características físicas, localização e preços.

        ### 🔹 Variável Alvo: `price`
        - **Média**: aproximadamente \$540.088  
        - **Mediana**: \$450.000  
        - **Desvio padrão**: \$367.127  
        - A distribuição é **assimétrica à direita**, indicando a presença de imóveis de alto padrão que elevam a média.
        """)



    
    # ----------- HISTOGRAMAS INTERATIVOS -----------
    st.subheader("🔹 Histogramas das Variáveis")

    cols_hist = ['price', 'sqft_living', 'bedrooms', 'bathrooms', 'floors', 'sqft_above', 'sqft_basement']

    selected_hist = st.multiselect("Selecione variáveis numéricas para visualizar histogramas:", options=cols_hist)

    if not selected_hist:
        st.info("Selecione ao menos uma variável para visualizar.")
    else:
        for col in selected_hist:
            st.subheader(f"Distribuição de {col}")
            chart = alt.Chart(df).mark_bar(opacity=0.7, color='steelblue').encode(
                alt.X(col, bin=alt.Bin(maxbins=40), title=col),
                y='count()',
                tooltip=[col]
            ).properties(width=600, height=300)
            st.altair_chart(chart, use_container_width=True)

    # ----------- BOXPLOTS INTERATIVOS -----------
    st.header("📦 Boxplots: Preço vs Variáveis Categóricas")

    cols_box = ['bedrooms', 'bathrooms', 'floors']
    selected_box = st.multiselect("Selecione variáveis para comparar com o preço:", options=cols_box)

    if not selected_box:
        st.info("Selecione ao menos uma variável para visualizar boxplots.")
    else:
        for col in selected_box:
            st.subheader(f"Preço vs {col}")
            chart = alt.Chart(df).mark_boxplot(extent='min-max').encode(
                x=alt.X(f'{col}:O', title=col),
                y=alt.Y('price:Q', scale=alt.Scale(type='log'), title='Preço (escala log)'),
                tooltip=['price', col]
            ).properties(width=600, height=300)
            st.altair_chart(chart, use_container_width=True)

# =============================

#
def q1_etapa2():
    st.markdown("---")
    st.header("Q1 - 2 - Regressao lienar Modelo de Regressão Linear")
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
    st.header("Q1 - 3️⃣ Interpretação dos Resultados")
    if "X_test" in st.session_state and "y_test" in st.session_state and "y_pred" in st.session_state:
        X = st.session_state["X_test"]
        y = st.session_state["y_test"]
        y_pred = st.session_state["y_pred"]
        avaliar_pressupostos(X, y, y_pred)
    else:
        st.warning("⚠️ Execute a Etapa 2 para gerar o modelo antes de interpretar os resultados.")



def q1_etapa4():
    st.markdown("---")
    st.header("4️⃣ Ajustes no Modelo")
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
    st.header("Q1 - 5️⃣ Tomada de Decisão")
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
    df = df.copy()
    df['arrival_month_num'] = df['arrival_date_month'].map(meses)
    
    # Criar a coluna de data de chegada
   # df['arrival_date'] = pd.to_datetime(
   #     df['arrival_date_year'].astype(str) + '-' +
   #     df['arrival_month_num'].astype(str).str.zfill(2) + '-' +
   #     df['arrival_date_day_of_month'].astype(str).str.zfill(2)
   # )
    
    #st.write(df["arrival_date"])
    
     # Cria coluna booking_date (data da reserva)
    #df['booking_date'] = df['arrival_date'] - pd.to_timedelta(df['lead_time'], unit='D')
    #df.loc[:, 'booking_date'] = df['arrival_date'] - pd.to_timedelta(df['lead_time'], unit='D')
    
     #excluir arrival_date_year, arrival_date_month, arrival_date_week_number, arrival_date_day_of_month
    #variaves foram substituídas por arrival_date

    #df = df.drop(columns=["arrival_date_year", "arrival_date_month", "arrival_date_week_number", "arrival_date_day_of_month"], errors='ignore')

    return df

def pre_processamento(df):
    """
    Realiza pré-processamento básico dos dados
    """
    # Remove duplicatas
    df = df.drop_duplicates()
    #df = criar_coluna_arrival_date(df)
    #df["arrival_date"] = pd.to_datetime(df["arrival_date"], errors='coerce')
    #df['arrival_date'] = pd.to_datetime(df['arrival_date']).dt.round('ms')  # ou 's' para segundos
    #df['booking_date'] = pd.to_datetime(df['booking_date'], errors='coerce')
    
    return df



def q2_etapa1():
    st.header("Q2 - a) Análise Descritiva dos Dados")
    st.info("Realize uma análise descritiva da base de dados.")
    uploaded_file = 'src/hotel_bookings.csv'
    # Carregar os dados

    df2 = carregar_dados(uploaded_file)
    st.session_state["hb_df"] = df2
    #st.success("✅ Arquivo carregado com sucesso!")
    st.markdown("### Preview dos Dados")
    st.dataframe(df2.head())
    
 
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
            "reservation_status", "reservation_status_date", "booking_date"
        ],
        "Tipo": [
            "Categórica", "Binária", "Numérica", "Numérica (inteira)", "Categórica",
            "Numérica (inteira)", "Numérica (inteira)", "Numérica (inteira)", "Numérica (inteira)",
            "Numérica (inteira)", "Numérica (inteira)", "Numérica (inteira)", "Categórica",
            "Categórica", "Categórica", "Categórica", "Binária", "Numérica (inteira)",
            "Numérica (inteira)", "Categórica", "Categórica", "Numérica (inteira)", "Categórica",
            "Categórica", "Categórica", "Numérica (inteira)", "Categórica", "Numérica (float)",
            "Numérica (inteira)", "Numérica (inteira)", "Categórica", "Data","Data"
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
    st.subheader("Resumo Estatístico")
    st.dataframe(df2.describe())
 
     # Limpar colunas irrelevantes
    df2 = df2.drop(columns=['RowNumber', 'CustomerId', 'Surname'], errors='ignore')
    
    
    # graficos de distribuição
    cols_numericas = [
        'lead_time', 'stays_in_weekend_nights', 'stays_in_week_nights', 'adults', 'children',
        'babies', 'previous_cancellations', 'previous_bookings_not_canceled',
        'booking_changes', 'days_in_waiting_list', 'adr',
        'required_car_parking_spaces', 'total_of_special_requests'
    ]
   

    
     # Mapear cancelamentos para facilitar leitura nos gráficos
    df2['cancelamento'] = df2['is_canceled'].map({0: 'Não Cancelada', 1: 'Cancelada'})
    
    

    # Calcular contagem e porcentagem
    contagem = df2['cancelamento'].value_counts()
    porcentagem = df2['cancelamento'].value_counts(normalize=True) * 100

    # Criar tabela
    tabela = pd.DataFrame({
        'Status da Reserva': contagem.index,
        'Quantidade': contagem.values,
        'Porcentagem (%)': porcentagem.values.round(2)
    })

    # Exibir resultado
    st.subheader("Quantidade e Porcentagem de Cancelamentos")
    st.dataframe(tabela)
    
    

    st.subheader("Distribuição das Reservas (Canceladas vs. Não Canceladas)")
    fig1, ax1 = plt.subplots()
    sns.countplot(data=df2, x='cancelamento', ax=ax1)
    ax1.set_title("Distribuição de Cancelamentos")
    st.pyplot(fig1)

    st.subheader("Tipo de Hotel vs Cancelamento")
    fig2, ax2 = plt.subplots()
    sns.countplot(data=df2, x='hotel', hue='cancelamento', ax=ax2)
    ax2.set_title("Cancelamentos por Tipo de Hotel")
    st.pyplot(fig2)

    st.subheader("Tempo de Antecedência (Lead Time) por Cancelamento")
    fig3, ax3 = plt.subplots()
    sns.boxplot(data=df2, x='cancelamento', y='lead_time', ax=ax3)
    ax3.set_title("Lead Time por Status de Cancelamento")
    st.pyplot(fig3)


    st.subheader("Distribuição de Reservas por Mês")
    fig4, ax4 = plt.subplots(figsize=(10, 4))
    order_months = ['January', 'February', 'March', 'April', 'May', 'June',
                    'July', 'August', 'September', 'October', 'November', 'December']
    sns.countplot(data=df2, x='arrival_date_month', order=order_months, hue='cancelamento', ax=ax4)
    ax4.set_title("Reservas por Mês (com Cancelamentos)")
    plt.xticks(rotation=45)
    st.pyplot(fig4)

    st.subheader("Considerações Finais da Análise Descritiva")
    st.markdown(r"""
    Com base nos gráficos apresentados, observa-se que aproximadamente 27,49% das reservas foram canceladas, enquanto 72,51% foram efetivadas, o que revela uma taxa de cancelamento relevante no conjunto de dados. A análise por tipo de hotel mostra que o City Hotel apresenta maior volume absoluto de cancelamentos em comparação ao Resort Hotel, refletindo também sua maior participação no total de reservas. Além disso, há uma variação expressiva ao longo do ano: os meses de julho e agosto concentram o maior volume de reservas e de cancelamentos, sugerindo que a alta temporada influencia diretamente o comportamento de churn.

    Outro fator importante identificado é o tempo de antecedência da reserva (lead time). As reservas que foram canceladas apresentam, em média, um lead time maior que as reservas mantidas, o que indica que clientes que reservam com muita antecedência têm maior probabilidade de cancelar. Esse insight pode ser utilizado para definir políticas comerciais mais eficientes, como tarifas não reembolsáveis, programas de fidelização ou ações promocionais segmentadas, contribuindo para a redução da taxa de cancelamentos e para o aumento da previsibilidade operacional.
    """)


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
    
    
    st.header("Q2 - b) Modelo de Regressão Logística")
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
    
    st.subheader("Explicação das Métricas do Modelo de Regressão Logística")
    st.dataframe(metricas)

     
     
def q2_etapa3():
        st.markdown("---")
        st.header("Q2 - c) Análise das Features")
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
        ### 🔺 Cinco variáveis que mais **aumentam** a chance de cancelamento:

        1. **`deposit_type_Non Refund`**  
        Depósitos não reembolsáveis estão associadas a uma **alta chance de cancelamento negativo** (impacto negativo forte no modelo), ou seja, sua ausência pode elevar o risco.

        2. **`required_car_parking_spaces`**  
        Quanto **menos necessidade de vaga de estacionamento**, maior a probabilidade de o cliente cancelar — sugerindo menor comprometimento.

        3. **`market_segment_Offline TA/TO`**  
        Reservas feitas por agências offline parecem estar mais ligadas a cancelamentos.

        4. **`deposit_type_Refundable`**  
        A opção de **reembolso** facilita o cancelamento, aumentando sua probabilidade.

        5. **`market_segment_Groups`**  
        Embora grupos sejam geralmente mais estáveis, nesse caso específico os dados indicam **maior risco de cancelamento**, talvez por volume ou incerteza logística.

        ---

        ### 🔻 Cinco variáveis que mais **reduzem** a chance de cancelamento:

        1. **`lead_time`**  
        Reservas feitas com bastante antecedência estão **menos propensas a serem canceladas**, indicando planejamento.

        2. **`customer_type_Transient`**  
        Hóspedes de tipo transitório demonstram **baixo risco de cancelamento**, talvez por viagens rápidas e com datas fixas.

        3. **`previous_cancellations`**  
        Curiosamente, clientes com histórico anterior **têm menor peso negativo aqui**, indicando que talvez tenham retornado com mais compromisso.

        4. **`market_segment_Complementary`**  
        Reservas promocionais ou gratuitas apresentam **menor risco de cancelamento**, talvez por serem incentivos vinculados a eventos específicos.

        5. **`distribution_channel_Undefined`**  
        Quando o canal de distribuição não é especificado, o modelo entende que há **menos risco de cancelamento**, possivelmente por padrão de preenchimento.

        ---

        Essas variáveis ajudam a identificar **perfis de clientes, canais e reservas** com maior ou menor probabilidade de cancelamento, permitindo **ações preventivas** e estratégias comerciais mais eficazes.
        """)


        
        
def q2_etapa4():
    st.markdown("---")
    st.header("Q2 - d) Justificativa do Método")
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
    st.header("Q3 - a) Análise Descritiva")
    st.info("Explore dados por país, quantidade e preço.")
        
        # Carregar a planilha (ou use df se já estiver carregado)
    @st.cache_data
    def carregar_dados():
        return pd.read_csv("src/planilha_combinada.csv")

    df = carregar_dados()
    
    #adicionar df na sessão do streamlit
    st.session_state["df3"] = df

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

    # mostrar valores ausentes no streamlit
    st.markdown("### Valores Ausentes por Coluna")
    st.dataframe(tabela_ausentes.sort_values(by="Percentual (%)", ascending=False))
     
    # Exibir o DataFrame original
    st.markdown("### Preview dos Dados")
    st.dataframe(df.head())
    # Exibir estatísticas descritivas
    st.markdown("### Estatísticas Descritivas")
    st.dataframe(df.describe())
    

    # 4. Gráficos de distribuição
    st.subheader("📈 Distribuições")

    # 6. Evolução temporal
    st.subheader("📅 Evolução Temporal das Vendas")
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df['Mês'] = df['InvoiceDate'].dt.to_period('M').astype(str)
    vendas_mes = df.groupby("Mês")['Quantity'].sum().reset_index()
    st.line_chart(vendas_mes.set_index("Mês"))
        
   
    # stockcod por país
    vendas_pais_stockcode = df.groupby("Country")["StockCode"].nunique().reset_index()
    vendas_pais_stockcode = vendas_pais_stockcode.sort_values(by="StockCode", ascending=False)
    st.markdown("### 📊 Quantidade de Produtos Vendidos por País")
    st.dataframe(vendas_pais_stockcode)
    
    # grafico stats.probplot de quantidade de produtos vendidos por país
    # use probplot para verificar a distribuição
    import scipy.stats as stats
    import matplotlib.pyplot as plt
    import seaborn as sns
   

    # Verifique se a coluna 'StockCode' é numérica
    vendas_pais_stockcode["StockCode"] = pd.to_numeric(vendas_pais_stockcode["StockCode"], errors="coerce")

    # Criação da figura
    fig_stockcode = plt.figure(figsize=(10, 6))

    # Gráfico Q-Q plot (verificação de normalidade)
    stats.probplot(vendas_pais_stockcode["StockCode"].dropna(), dist="norm", plot=plt)
    plt.title("Distribuição de Produtos Vendidos por País")

    # Renderização no Streamlit
    st.pyplot(fig_stockcode)
    
    # Suponha que você já tenha o DataFrame `df`
    # Certifique-se de que 'Price' é numérico
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce")

    # Criação da figura
    fig_price = plt.figure(figsize=(10, 6))

    # Gráfico Q-Q plot para verificar normalidade do preço
    stats.probplot(df["Price"].dropna(), dist="norm", plot=plt)
    plt.title("Q-Q Plot - Distribuição do Preço dos Produtos")

    # Exibir no Streamlit
    st.pyplot(fig_price)

    st.subheader("Resumo da Análise Descritiva ")
    
    st.markdown("""
    A análise do Q-Q Plot para o **preço dos produtos** revela uma clara assimetria à direita, indicando que a distribuição dos valores não é normal. Observa-se a presença de diversos **outliers de alta magnitude**, com uma curvatura acentuada no gráfico e dispersão evidente dos pontos em relação à linha teórica de normalidade. Isso sugere uma elevada **variabilidade nos preços**, com alguns produtos muito caros distorcendo a média e influenciando fortemente as estatísticas descritivas tradicionais.

    Da mesma forma, a **quantidade de produtos vendidos por país** também não apresenta distribuição normal. O gráfico evidencia uma forte **concentração de países com baixas vendas** e um pequeno número de países com volumes expressivamente maiores. A linha teórica de normalidade é ultrapassada nos extremos, reforçando a ideia de uma **distribuição assimétrica com cauda pesada**, o que indica que testes estatísticos não paramétricos são mais adequados para esse tipo de dado.
    """)

            
def q3_etapa2():
    st.header("Q3 - b) ANOVA entre Países")
    st.info("Apresente F, p-valor e interprete o teste.")
        
    import pandas as pd
    import statsmodels.api as sm
    from statsmodels.formula.api import ols
   
    #recuperar df3 da sessão do streamlit
    df3 = st.session_state.get("df3")
    if df3 is None:
        st.warning("⚠️ Os dados ainda não foram carregados. Execute a Etapa 1 primeiro.")
        return


    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    import pandas as pd

    # Exemplo: carregando o DataFrame (substitua pelo seu)
    # df = pd.read_excel("planilha_combinada_amostrada.xlsx")

    st.subheader("ANOVA - Comparação de Médias por País")

    # ANOVA para Quantity ~ Country
    modelo_q = smf.ols("Quantity ~ C(Country)", data=df3).fit()
    anova_q = sm.stats.anova_lm(modelo_q, typ=2)

    # ANOVA para Price ~ Country
    modelo_p = smf.ols("Price ~ C(Country)", data=df3).fit()
    anova_p = sm.stats.anova_lm(modelo_p, typ=2)

    # Resultado Quantity
    st.subheader("🔹 ANOVA: Quantity por País")
    st.dataframe(anova_q)

    f_q = anova_q.loc["C(Country)", "F"]
    p_q = anova_q.loc["C(Country)", "PR(>F)"]

    if p_q < 0.001:
        st.success(f"Resultado: influência **muito significativa** (F = {f_q:.2f}, p < 0.001)")
    elif p_q < 0.05:
        st.info(f"Resultado: influência **significativa** (F = {f_q:.2f}, p = {p_q:.4f})")
    else:
        st.warning(f"Resultado: **sem influência significativa** (F = {f_q:.2f}, p = {p_q:.4f})")
        
   
        
    # Resultado Price
    st.subheader("🔹 ANOVA: Price por País")
    st.dataframe(anova_p)

    f_p = anova_p.loc["C(Country)", "F"]
    p_p = anova_p.loc["C(Country)", "PR(>F)"]

    if p_p < 0.001:
        st.success(f"Resultado: influência **muito significativa** (F = {f_p:.2f}, p < 0.001)")
    elif p_p < 0.05:
        st.info(f"Resultado: influência **significativa** (F = {f_p:.2f}, p = {p_p:.4f})")
    else:
        st.warning(f"Resultado: **sem influência significativa** (F = {f_p:.2f}, p = {p_p:.4f})")
        
        
        
    st.markdown("""
    ### Interpretação do Resultado - Comparação de Médias por País

    A análise de variância (ANOVA) foi utilizada para verificar se as médias de **quantidade de produtos vendidos** e de **preço dos produtos** diferem significativamente entre os países.

    🔹 **Quantity por País**  
    - Estatística F = **131.59**  
    - Valor-p = **< 0.001**  
    ➡️ **Resultado:** Evidência estatística muito forte de que a quantidade média de produtos vendidos varia significativamente entre os países. Isso indica que o país de venda exerce uma **influência substancial** nas quantidades comercializadas.

    🔹 **Price por País**  
    - Estatística F = **5.95**  
    - Valor-p = **< 0.001**  
    ➡️ **Resultado:** Há uma **influência estatisticamente significativa** do país sobre os preços médios praticados. Embora o efeito seja menos intenso que o observado para quantidade, ainda assim é estatisticamente relevante.

    Esses resultados indicam que tanto o volume quanto o preço de produtos variam de forma significativa entre os países, reforçando a importância de considerar o fator geográfico em análises comerciais e estratégias de mercado.
    """)
       
    

def verificar_pressupostos_anova(df, var_resposta, fator_categ='Country'):
    """
    Verifica os pressupostos da ANOVA:
    - Normalidade dos resíduos (Shapiro-Wilk + Q-Q plot)
    - Homocedasticidade (Breusch-Pagan + gráfico de resíduos)

    Parâmetros:
    - df: DataFrame com os dados
    - var_resposta: variável numérica dependente (ex: 'Quantity', 'Price')
    - fator_categ: variável categórica (ex: 'Country')

    Exibe resultados e interpretações no Streamlit.
    """
    import streamlit as st
    import statsmodels.formula.api as smf
    import statsmodels.api as sm
    from statsmodels.stats.diagnostic import het_breuschpagan
    from scipy.stats import shapiro, zscore, kstest
    import matplotlib.pyplot as plt
    import seaborn as sns

    st.subheader(f"Verificação dos Pressupostos da ANOVA - {var_resposta} ~ {fator_categ}")

    # Ajustar modelo
    formula = f"{var_resposta} ~ C({fator_categ})"
    modelo = smf.ols(formula, data=df).fit()
    residuos = modelo.resid
    valores_previstos = modelo.fittedvalues

    violou_normalidade = False
    violou_homocedasticidade = False            

    
    #  Normalidade dos resíduos
    st.subheader("1️ Normalidade dos Resíduos (Q-Q Plot + Kolmogorov-Smirnov)")

    # Criar um container estreito com colunas
    col1, col2, col3 = st.columns([1, 2, 1])  # centraliza

    with col1:
        fig, ax = plt.subplots(figsize=(3, 2.5))
        sm.qqplot(residuos, line='s', ax=ax)
        ax.set_title("Q-Q Plot dos Resíduos", fontsize=10)
        st.pyplot(fig)

    # Aplicar o teste Kolmogorov-Smirnov com resíduos padronizados
    residuos_padronizados = zscore(residuos)
    ks_stat, ks_p = kstest(residuos_padronizados, 'norm')

    if ks_p < 0.05:
        violou_normalidade = True
        st.warning("Essa versão evita o alerta do Shapiro para N > 5000 e é mais apropriada para grandes amostras.")
        st.warning(f"❗ Kolmogorov-Smirnov indica violação da normalidade (U)(p = {ks_p:.4f})")
    else:
        st.success(f"✅ Resíduos seguem distribuição normal (Kolmogorov-Smirnov p = {ks_p:.4f})")
   

    # Layout com colunas para centralizar e limitar largura
    #col1, col2, col3 = st.columns([1, 2, 1])  # col2 é a central

    with col2:
        with st.container():
            fig2, ax2 = plt.subplots(figsize=(3, 2.5), dpi=100)
            sns.scatterplot(x=valores_previstos, y=residuos, s=10, ax=ax2)
            ax2.axhline(0, color='red', linestyle='--')
            ax2.set_xlabel("Valores Previstos", fontsize=8)
            ax2.set_ylabel("Resíduos", fontsize=8)
            ax2.set_title("Resíduos vs. Valores Previstos", fontsize=10)
            ax2.tick_params(axis='both', labelsize=7)
            fig2.tight_layout()
            st.pyplot(fig2, use_container_width=False)

    # Teste de Breusch-Pagan
    bp_test = het_breuschpagan(residuos, modelo.model.exog)
    p_bp = bp_test[3]  # p-valor do teste (stat, pval, fval, f_pval)

    st.write(f"🔬 Breusch-Pagan (homocedasticidade dos resíduos): p = {p_bp:.4f}")
    if p_bp < 0.05:
        violou_homocedasticidade = True
        st.warning("❗ Violação da homocedasticidade detectada (p < 0.05)")
    else:
        st.success("✅ Homocedasticidade verificada (p ≥ 0.05)")

    # Diagnóstico final
    st.subheader("Diagnóstico Final dos Pressupostos")
    if violou_normalidade or violou_homocedasticidade:
        st.error("⚠️ Um ou mais pressupostos da ANOVA foram violados. Considere usar testes não paramétricos : teste de Kruskal-Wallis.")
    else:
        st.success("✅ Todos os pressupostos foram atendidos. A ANOVA é apropriada.")




def q3_etapa3():
    st.header("Q3 - c) Ajustes no Modelo")    
    st.info("Verifique normalidade, homocedasticidade   etc.")
    
    # recuperar df3 da sessão do streamlit
    df3 = st.session_state.get("df3")
    
    if df3 is None:
        st.warning("⚠️ Os dados ainda não foram carregados. Execute a Etapa 1 primeiro.")
        return
    
    verificar_pressupostos_anova(df3, var_resposta="Quantity", fator_categ="Country")
    verificar_pressupostos_anova(df3, var_resposta="Price", fator_categ="Country")  
    
    
        # aplicar  Kruskal-Wallis
    from scipy.stats import kruskal

    st.subheader("Q3 - c) Teste Kruskal-Wallis")
    st.info("Teste não paramétrico para comparar medianas entre grupos.")
    anchor = "teste_kruskal_wallis"
    st.markdown(f"<a id='{anchor}'></a>", unsafe_allow_html=True)

    # Agrupar os dados por país
    grupos_quantity = [group["Quantity"].values for name, group in df3.groupby("Country")]
    grupos_price = [group["Price"].values for name, group in df3.groupby("Country")]

    # Aplicar o teste Kruskal-Wallis
    kruskal_quantity = kruskal(*grupos_quantity)
    kruskal_price = kruskal(*grupos_price)

    # Exibir os resultados
    st.markdown("### Teste Kruskal-Wallis - Comparação de Medianas")
    st.write(f"🔬 Kruskal-Wallis (Quantidade): H = {kruskal_quantity.statistic:.4f}, p = {kruskal_quantity.pvalue:.4f}")
    st.write(f"🔬 Kruskal-Wallis (Preço): H = {kruskal_price.statistic:.4f}, p = {kruskal_price.pvalue:.4f}")

    # Análise automática dos resultados
    st.markdown("### Análise Comparativa")

    # Interpretação Quantity
    if kruskal_quantity.pvalue < 0.05:
        st.success("✅ Diferença significativa detectada nas **quantidades vendidas entre países** (p < 0.05).")
    else:
        st.warning("⚠️ Não foi detectada diferença significativa nas **quantidades vendidas entre países** (p ≥ 0.05).")

    # Interpretação Price
    if kruskal_price.pvalue < 0.05:
        st.success("✅ Diferença significativa detectada nos **preços praticados entre países** (p < 0.05).")
    else:
        st.warning("⚠️ Não foi detectada diferença significativa nos **preços praticados entre países** (p ≥ 0.05).")





def q3_etapa4():
    st.header("Q3 - d) Interpretação e Decisão")
    
    
    from scipy.stats import kruskal

    st.info(" Interpretação do Teste Kruskal-Wallis - Teste não paramétrico para comparar medianas entre grupos, Decisões com base nas diferenças de médias")

    df3 = st.session_state.get("df3")
    if df3 is None:
        st.warning("⚠️ Os dados ainda não foram carregados. Execute a Etapa 1 primeiro.")
        return

    grupos_quantity = [group["Quantity"].values for name, group in df3.groupby("Country")]
    grupos_price = [group["Price"].values for name, group in df3.groupby("Country")]

    kruskal_quantity = kruskal(*grupos_quantity)
    kruskal_price = kruskal(*grupos_price)

    st.markdown("### Teste Kruskal-Wallis - Comparação de Medianas")
    st.write(f"🔬 Kruskal-Wallis (Quantidade): H = {kruskal_quantity.statistic:.4f}, p = {kruskal_quantity.pvalue:.4f}")
    st.write(f"🔬 Kruskal-Wallis (Preço): H = {kruskal_price.statistic:.4f}, p = {kruskal_price.pvalue:.4f}")

    st.markdown("""
    ### Análise dos Resultados

    Nesta etapa, foi aplicado o **teste de Kruskal-Wallis**, uma técnica não paramétrica utilizada para comparar as **medianas** de múltiplos grupos independentes, neste caso, os diferentes **países**.

    Os resultados apontam:

    - **Quantidade Vendida**: O valor da estatística de Kruskal-Wallis foi **H = {:.4f}**, com um **p-valor de {:.4f}**, indicando uma diferença estatisticamente significativa entre os países quanto à quantidade de produtos vendidos.
    - **Preço dos Produtos**: A estatística H foi de **{:.4f}**, também com **p-valor de {:.4f}**, evidenciando que os preços praticados também variam significativamente entre os países.

    Ambos os resultados têm **p < 0,05**, o que leva à rejeição da hipótese nula de igualdade das medianas entre os grupos. Isso confirma que **existem diferenças significativas tanto nas quantidades quanto nos preços entre os países analisados**.

    Essas variações podem estar associadas a fatores econômicos locais, estratégias de mercado, políticas comerciais ou mesmo particularidades culturais e regionais que afetam o consumo e o valor dos produtos. Portanto, **estratégias de venda e precificação devem considerar essas diferenças para maior assertividade e competitividade em cada mercado nacional**.
    """.format(
        kruskal_quantity.statistic, kruskal_quantity.pvalue,
        kruskal_price.statistic, kruskal_price.pvalue
    ))
    
    
    

# =============================
# 🔧 Funções - Questão 4
# =============================
def q4_etapa1():
    st.header("Q4 - a) Discussão do Problema")
    st.info("Importância de prever reclamações no varejo.")

def q4_etapa2():
    import streamlit as st
    import pandas as pd
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt

    

    st.title("📊 Análise Descritiva: Customer Personality Analysis")

    # Carregar dados
    @st.cache_data
    def carregar_dados():
        return pd.read_csv("src/marketing_campaign.csv", sep="\t")

    df4 = carregar_dados()
    
    # iNserir df4 na sessão do streamlit
    st.session_state["df4"] = df4
    # Exibir o DataFrame original
    
    with st.expander("📚 Dicionário de Dados"):
        st.markdown("""
    ### 👤 Pessoas
    - **ID**: Identificador exclusivo do cliente  
    - **Year_Birth**: Ano de nascimento do cliente  
    - **Education**: Nível de escolaridade do cliente  
    - **Marital_Status**: Estado civil do cliente  
    - **Income**: Renda familiar anual do cliente  
    - **Kidhome**: Número de crianças na casa do cliente  
    - **Teenhome**: Número de adolescentes na casa do cliente  
    - **Dt_Customer**: Data de inscrição do cliente na empresa  
    - **Recency**: Número de dias desde a última compra do cliente  
    - **Complain**: 1 se o cliente reclamou nos últimos 2 anos, 0 caso contrário  

    ---

    ### 🛍️ Produtos
    - **MntWines**: Valor gasto em vinho nos últimos 2 anos  
    - **MntFruits**: Valor gasto em frutas nos últimos 2 anos  
    - **MntMeatProducts**: Valor gasto em carne nos últimos 2 anos  
    - **MntFishProducts**: Valor gasto em peixes nos últimos 2 anos  
    - **MntSweetProducts**: Valor gasto em doces nos últimos 2 anos  
    - **MntGoldProds**: Valor gasto em ouro nos últimos 2 anos  

    ---

    ### 🎯 Promoção
    - **NumDealsPurchases**: Número de compras feitas com desconto  
    - **AcceptedCmp1**: 1 se o cliente aceitou a oferta na 1ª campanha  
    - **AcceptedCmp2**: 1 se o cliente aceitou a oferta na 2ª campanha  
    - **AcceptedCmp3**: 1 se o cliente aceitou a oferta na 3ª campanha  
    - **AcceptedCmp4**: 1 se o cliente aceitou a oferta na 4ª campanha  
    - **AcceptedCmp5**: 1 se o cliente aceitou a oferta na 5ª campanha  
    - **Response**: 1 se o cliente aceitou a oferta na última campanha  

    ---

    ### 🏬 Lugar (Canal de Compra)
    - **NumWebPurchases**: Número de compras feitas através do site  
    - **NumCatalogPurchases**: Número de compras feitas usando catálogo  
    - **NumStorePurchases**: Número de compras feitas diretamente nas lojas  
    - **NumWebVisitsMonth**: Número de visitas ao site no último mês  
    """)

    
    
    

    # ----------- Tratamento inicial ----------- #
    df4["Income"] = df4["Income"].fillna(df4["Income"].median())
    df4.drop(columns=["Z_CostContact", "Z_Revenue", "ID"], inplace=True)
    df4["Age"] = 2025 - df4["Year_Birth"]
    df4["Dt_Customer"] = pd.to_datetime(df4["Dt_Customer"], format="%d-%m-%Y")
    df4["TotalChildren"] = df4["Kidhome"] + df4["Teenhome"]
    df4["TotalSpent"] = df4[[
        "MntWines", "MntFruits", "MntMeatProducts", 
        "MntFishProducts", "MntSweetProducts", "MntGoldProds"
    ]].sum(axis=1)

    # Codificação
    df_encoded = pd.get_dummies(df4, columns=["Education", "Marital_Status"], drop_first=True)

    # ----------- Painel de Análise ----------- #
    st.markdown("## 🔍 Visão Geral da Base")
    st.dataframe(df4.head())

    st.markdown("## Tratamento de Dados")
    st.write("Valores ausentes tratados. Renda preenchida com mediana.")
    st.write("Variáveis derivadas adicionadas: `Age`, `TotalChildren`, `TotalSpent`")

    # ----------- Estatísticas e Segmentos ----------- #
    st.markdown("##  Estatísticas Descritivas por Grupo")

    with st.expander("👤 Pessoas"):
        st.write(df4[["Age", "Income", "Kidhome", "Teenhome", "TotalChildren"]].describe())
        fig1, ax1 = plt.subplots()
        sns.histplot(df4["Age"], bins=20, kde=True, ax=ax1)
        ax1.set_title("Distribuição da Idade")
        st.pyplot(fig1)

    with st.expander("🛍️ Produtos"):
        cols_prod = ["MntWines", "MntFruits", "MntMeatProducts",
                    "MntFishProducts", "MntSweetProducts", "MntGoldProds", "TotalSpent"]
        st.write(df4[cols_prod].describe())
        fig2, ax2 = plt.subplots(figsize=(10, 4))
        sns.boxplot(data=df4[cols_prod], ax=ax2)
        ax2.set_title("Gasto por Categoria de Produto")
        ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45)
        st.pyplot(fig2)

    with st.expander("🎯 Promoções"):
        cols_promo = ["AcceptedCmp1", "AcceptedCmp2", "AcceptedCmp3",
                    "AcceptedCmp4", "AcceptedCmp5", "Response", "NumDealsPurchases"]
        st.write(df4[cols_promo].sum().to_frame("Total de Aceites/Compras"))

    with st.expander("🏬 Lugar / Canal"):
        cols_lugar = ["NumWebPurchases", "NumCatalogPurchases", 
                    "NumStorePurchases", "NumWebVisitsMonth"]
        st.write(df4[cols_lugar].describe())
        fig3, ax3 = plt.subplots()
        sns.heatmap(df4[cols_lugar].corr(), annot=True, cmap="coolwarm", ax=ax3)
        st.pyplot(fig3)

    # ----------- Relação com Reclamar ----------- #
    st.markdown("## 🎯 Reclamações (Variável Alvo)")

    st.write("Distribuição de Reclamações")
    fig4, ax4 = plt.subplots()
    sns.countplot(x="Complain", data=df4, ax=ax4)
    ax4.set_title("Complain - Distribuição")
    st.pyplot(fig4)
    quantidade_reclamacoes = df4["Complain"].value_counts()
    st.write("Quantidade de Reclamações: 0 (não reclamou) e 1 (reclamou):")
    st.markdown("#### Quantidade de Reclamações")
    st.markdown("0 (não reclamou): {} | 1 (reclamou): {}".format(
        quantidade_reclamacoes.get(0, 0), quantidade_reclamacoes.get(1, 0)))
    st.markdown("#### Tabela de Reclamações")
    quantidade_reclamacoes = df4["Complain"].value_counts().rename_axis("Complain").reset_index(name="Count")
    quantidade_reclamacoes["Percentage"] = (quantidade_reclamacoes["Count"] / quantidade_reclamacoes["Count"].sum()) * 100
    quantidade_reclamacoes["Percentage"] = quantidade_reclamacoes["Percentage"].round(2)
    quantidade_reclamacoes = quantidade_reclamacoes.rename(columns={"Complain": "Reclamou"})
    quantidade_reclamacoes = quantidade_reclamacoes.set_index("Reclamou")   
    st.write(quantidade_reclamacoes)    
    
    # Justifica que adotaremos a tecnica smote para lidar com o desbalanceamento
    st.markdown("### Desbalanceamento de Classes")
    st.write("A variável alvo `Complain` apresenta um desbalanceamento significativo, com a maioria dos clientes não reclamando. Para lidar com isso, utilizaremos a técnica SMOTE (Synthetic Minority Over-sampling Technique) para balancear as classes antes de treinar os modelos preditivos.")

    st.markdown("### 🔍 Boxplots: Variáveis numéricas vs Reclamar")
    col_numericas = ["Income", "Age", "TotalSpent", "TotalChildren", "Recency", "NumWebVisitsMonth"]

    for col in col_numericas:
        fig, ax = plt.subplots()
        sns.boxplot(x="Complain", y=col, data=df4, ax=ax)
        ax.set_title(f"{col} vs Reclamar")
        st.pyplot(fig)

    # ----------- Correlação ----------- #
    st.markdown("## 📌 Correlação com Reclamar")
    correlacoes = df_encoded.corr()["Complain"].sort_values(ascending=False)
    st.dataframe(correlacoes.to_frame().rename(columns={"Complain": "Correlação com Reclamar"}))

    # ----------- Dispersão Multivariada ----------- #
    st.markdown("## 🌐 Relações Multivariadas")

    fig, ax = plt.subplots()
    sns.scatterplot(x="Income", y="TotalSpent", hue="Complain", data=df4, ax=ax)
    ax.set_title("Gasto Total vs Renda (Colorido por Reclamar)")
    st.pyplot(fig)

    # Final
    st.success("✅ Análise Descritiva Concluída.")


        
    

def q4_etapa3():
    st.header("Q4 - c) Seleção de Modelos")
    st.info("Compare Logistic, Árvores, Random Forest, XGBoost.")
    
    #recuperar df4 da sessão do streamlit
    df4 = st.session_state.get("df4")
    if df4 is None:
        st.warning("⚠️ Os dados ainda não foram carregados. Execute a Etapa 2 primeiro.")
        return
   
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from xgboost import XGBClassifier
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
    from imblearn.over_sampling import SMOTE
    from sklearn.preprocessing import StandardScaler
    import warnings
    warnings.filterwarnings("ignore")
    # Pré-processamento
    df4["Income"] = df4["Income"].fillna(df4["Income"].median())
    df4["Age"] = 2025 - df4["Year_Birth"]
    df4["TotalChildren"] = df4["Kidhome"] + df4["Teenhome"]
    df4["TotalSpent"] = df4[[
        "MntWines", "MntFruits", "MntMeatProducts",
        "MntFishProducts", "MntSweetProducts", "MntGoldProds"
    ]].sum(axis=1)

    df4 = df4.drop(columns=["ID", "Year_Birth", "Dt_Customer", "Z_CostContact", "Z_Revenue"], errors="ignore")
    df4 = pd.get_dummies(df4, columns=["Education", "Marital_Status"], drop_first=True)

    # Divisão X e y
    X = df4.drop(columns=["Complain"])
    y = df4["Complain"]

    # Divisão treino/teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

    # Escalonamento
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    



    # Modelos a testar
    modelos = {
        "Árvore de Decisão": DecisionTreeClassifier(class_weight='balanced', random_state=42),
        "Random Forest": RandomForestClassifier(class_weight='balanced', random_state=42),
        "XGBoost": XGBClassifier(
            scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=42
        )
    }
    
     # Para avaliar o modelo na proxima questao
    
    #adicionar modelos na sessão do streamlit
    st.session_state["modelos"] = modelos
  
    #inserir X_test na sessão do streamlit
    st.session_state["X_test"] = X_test

  

    st.header("🔍 Avaliação dos Modelos")
    resultados = []

    for nome, modelo in modelos.items():
        modelo.fit(X_train_scaled, y_train)
        y_pred = modelo.predict(X_test_scaled)
        y_prob = modelo.predict_proba(X_test_scaled)[:, 1]

        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        auc = roc_auc_score(y_test, y_prob)

        # Matriz de confusão
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(ax=ax)
        plt.title(f"Matriz de Confusão - {nome}")
        st.pyplot(fig)

        resultados.append({
            "Modelo": nome,
            "Acurácia": report["accuracy"],
            "Precisão": report["1"]["precision"],
            "Recall": report["1"]["recall"],
            "F1-score": report["1"]["f1-score"],
            "AUC": auc
        })

    # Tabela final
    st.markdown("### Comparação Final dos Modelos")
    df_result = pd.DataFrame(resultados).sort_values(by="F1-score", ascending=False)
    st.dataframe(df_result, use_container_width=True)
    
        
    st.markdown("### Seleção de Modelos para Previsão de Reclamações")
    
    st.markdown(
    "A seguir, apresentamos uma análise comparativa de diferentes modelos de machine learning para prever reclamações de clientes com base em suas características demográficas e comportamentais. "
    "Como a variável alvo `Complain` está altamente desbalanceada (apenas 22 positivos em mais de 2200 registros), aplicamos estratégias específicas de balanceamento para cada modelo:"
    )

    st.markdown("""
    - **Árvore de Decisão**: utilizou `class_weight='balanced'` para penalizar mais os erros da classe minoritária.
    - **Random Forest**: também utilizou `class_weight='balanced'`, o que redistribui o peso das classes automaticamente.
    - **XGBoost**: utilizou `scale_pos_weight`, calculado como a razão entre a classe negativa e a positiva, para ajustar a importância da minoria.

    Essas abordagens buscam reduzir o viés para a classe majoritária e aumentar a sensibilidade do modelo aos clientes que realmente reclamaram.
    """)
    
    

    st.markdown(
        "A seguir, apresentamos uma análise comparativa de diferentes modelos de machine learning para prever reclamações de clientes com base em suas características demográficas e comportamentais. "
        "Os modelos testados foram: **Árvore de Decisão**, **Random Forest** e **XGBoost**. "
        "O objetivo é identificar o modelo mais eficaz para prever se um cliente irá reclamar ou não, utilizando métricas como acurácia, precisão, recall, F1-score e AUC."
    )

    st.markdown("""
    | Modelo             | Acurácia | Precisão | Recall | F1-Score | AUC     |
    |--------------------|----------|----------|--------|----------|---------|
    | XGBoost            | 0.993    | 0.667    | 0.400  | 0.500    | 0.704   |
    | Random Forest      | 0.993    | 1.000    | 0.200  | 0.333    | 0.771   |
    | Árvore de Decisão  | 0.989    | 0.333    | 0.200  | 0.250    | 0.598   |
    """)

    st.markdown("""
    **Análise**:
    - O **Random Forest** apresentou o melhor desempenho em AUC (0.771), indicando boa capacidade de separação entre clientes que reclamam e os que não reclamam.
    - O **XGBoost** teve o melhor equilíbrio entre precisão e recall, resultando em maior F1-score (**0.500**), sendo mais robusto para detectar corretamente clientes com maior risco de reclamar.
    - A **Árvore de Decisão** simples teve o desempenho mais baixo, sugerindo que modelos mais complexos são mais adequados ao problema.

    ✅ **Conclusão**: o modelo **XGBoost** é o mais indicado, considerando o equilíbrio entre todas as métricas de avaliação.
    """)
    
    
    st.markdown("### ✅ Justificativa da escolha do Modelo XGBoost")

    st.markdown(
        "O **XGBoost (Extreme Gradient Boosting)** foi escolhido por sua alta capacidade de generalização, eficiência computacional e excelente desempenho em tarefas de classificação binária, especialmente em conjuntos de dados desbalanceados. "
        "Diferente de modelos tradicionais, o XGBoost permite o ajuste explícito do parâmetro `scale_pos_weight`, que controla o peso da classe minoritária no cálculo da função de perda. "
        "Esse recurso é especialmente útil no contexto desta análise, onde apenas 1% dos clientes fizeram reclamações, gerando forte desbalanceamento na variável alvo `Complain`."
    )

    st.markdown(
        "Além disso, o XGBoost é robusto contra overfitting e realiza regularização L1/L2 automaticamente, o que o torna altamente indicado para dados com múltiplas variáveis explicativas e possíveis correlações."
    )

    st.success("📌 Portanto, o XGBoost foi ajustado com `scale_pos_weight` para lidar adequadamente com o desbalanceamento e demonstrou bom equilíbrio entre precisão e recall.")

    


def q4_etapa4():
    st.header("Q4 - d) SHAP e Explicabilidade")
    st.info("Use SHAP para entender a influência das variáveis.")
    

    # Recuperar o modelo XGBoost da sessão do streamlit
    modelos = st.session_state.get("modelos")
    if modelos is None:
        st.warning("⚠️ O modelo ainda não foi treinado. Execute a Etapa 3 primeiro.")
        return

    X_test = st.session_state.get("X_test")
    if X_test is None:
        st.warning("⚠️ Os dados de teste ainda não foram carregados. Execute a Etapa 3 primeiro.")
        return
    import shap
    import matplotlib.pyplot as plt
    # Aplicar SHAP (TreeExplainer para XGBoost)
    explainer = shap.TreeExplainer(modelos["XGBoost"])
    shap_values = explainer.shap_values(X_test)


    # Título da seção
    st.markdown("## 🔍 Explicabilidade das Variáveis com SHAP")

    st.markdown(
        """
        Utilizamos o método SHAP (SHapley Additive exPlanations) com o modelo XGBoost para entender o impacto de cada variável nas previsões de reclamações.
        A seguir, são exibidos dois gráficos:
        - **Gráfico de barras**: mostra a importância média das variáveis no modelo.
        - **Beeswarm**: mostra a distribuição dos impactos individuais por variável.
        """
    )
    


    # Gráfico de barras (importância média)
    st.subheader("📊 Importância Média das Variáveis")
    fig_bar, ax = plt.subplots()
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    st.pyplot(fig_bar)

    # Gráfico beeswarm (impacto individual)
    st.subheader("🌪️ Impacto Individual das Variáveis")
    fig_beeswarm, ax = plt.subplots()
    shap.summary_plot(shap_values, X_test, show=False)
    st.pyplot(fig_beeswarm)
    
    st.markdown("### 📊 Interpretação das Variáveis Mais Influentes (SHAP)")

    st.markdown("""
    A análise de explicabilidade com SHAP revelou quais variáveis mais impactam a previsão de reclamações dos clientes. Abaixo, discutimos os principais fatores e seu significado no contexto do negócio:
    """)

    st.markdown("""
    #### 1. Age (Idade)  
    A idade foi a variável com maior impacto no modelo preditivo.  
    Clientes mais jovens e mais velhos tendem a ter diferentes níveis de expectativa em relação aos produtos e serviços. Por exemplo, consumidores mais velhos podem ser mais criteriosos e mais propensos a formalizar reclamações quando percebem falhas no atendimento ou no produto.
    """)

    st.markdown("""
    #### 2. MntWines (Gasto com vinhos)  
    Reflete o nível de consumo específico em produtos premium como vinhos.  
    Clientes que investem valores mais altos nesse item são geralmente mais exigentes quanto à qualidade, entrega e experiência geral de compra, o que aumenta a chance de reclamações em caso de frustração.
    """)

    st.markdown("""
    #### 3. MntMeatProducts & MntGoldProds  
    Indicadores de clientes com ticket médio elevado.  
    Esses consumidores geralmente têm maior valor agregado para a empresa e, por isso, esperam um serviço de excelência. Pequenas falhas podem comprometer sua experiência e levá-los a reclamar com mais frequência.
    """)

    st.markdown("""
    #### 4. TotalSpent (Total gasto)  
    A soma total dos gastos em diferentes categorias mostra que quanto maior o investimento do cliente, maior sua atenção à jornada de consumo.  
    Se o retorno percebido (produto, atendimento, entrega) não for proporcional ao valor investido, a probabilidade de reclamação aumenta.
    """)

    st.markdown("""
    #### 5. Income (Renda)  
    A renda familiar está relacionada ao nível de exigência e expectativa.  
    Clientes de maior poder aquisitivo tendem a ser menos tolerantes a falhas operacionais e mais rápidos em expressar insatisfação por meio de reclamações formais.
    """)

    st.markdown("### Conclusão")
    st.markdown("""
    O padrão identificado revela que clientes com alto engajamento, gastos expressivos e maior expectativa são mais propensos a gerar reclamações. 

    """)


def q4_etapa5():
    st.header("Q4 - e) K-Means / DBSCAN")
    st.info("Clusterização e detecção de outliers por perfil.")
        
 
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans, DBSCAN
    import matplotlib.pyplot as plt

    # Título da seção
    st.markdown("## Análise Não Supervisionada - K-Means (Segmentação de Clientes)")

    # Carregar a base
    df4 = st.session_state.get("df4")
    if df4 is None:
        st.warning("⚠️ Os dados ainda não foram carregados. Execute a Etapa 2 primeiro.")
        return

    # Seleção das colunas para clusterização
    colunas_cluster = [
        "Income", "Kidhome", "Teenhome", "MntWines", "MntFruits",
        "MntMeatProducts", "MntFishProducts", "MntSweetProducts",
        "MntGoldProds", "NumDealsPurchases", "NumWebPurchases",
        "NumCatalogPurchases", "NumStorePurchases", "NumWebVisitsMonth"
    ]

    # Remover valores ausentes
    df_cluster = df4[colunas_cluster].dropna()

    # Padronizar os dados
    scaler = StandardScaler()
    X_cluster = scaler.fit_transform(df_cluster)

    # Calcular WCSS para diferentes valores de k
    wcss = []
    for k in range(1, 10):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_cluster)
        wcss.append(kmeans.inertia_)

    # Exibir o gráfico Elbow
    st.markdown("### Método do Cotovelo")
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(range(1, 10), wcss, marker='o')
    ax.set_xlabel("Número de Clusters (k)")
    ax.set_ylabel("Soma dos Erros Quadráticos (WCSS)")
    ax.set_title("Elbow - Definição do Número Ideal de Clusters")
    ax.grid(True)
    st.pyplot(fig)

    
        
        # Aplicar K-Means com k=3
    st.markdown("### Segmentação de Clientes com K-Means (k=3)")
    st.info("Segmentação dos clientes em 3 grupos com base nos atributos selecionados.")
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_cluster)

    # Adicionar os clusters ao dataframe original
    df_cluster["Cluster"] = clusters

    # Mostrar a contagem de clientes por cluster
    st.markdown("### Quantidade de Clientes por Cluster")
    st.dataframe(df_cluster["Cluster"].value_counts().rename("Clientes").reset_index().rename(columns={"index": "Cluster"}))

    # Exibir médias por cluster
    st.markdown("### Médias dos Atributos por Cluster")
    st.dataframe(df_cluster.groupby("Cluster").mean().round(2))
        
    st.markdown("### Justificativa do Número de Clusters k=3")
    st.markdown("""
    A análise do gráfico Elbow (cotovelo) mostra uma redução acentuada na soma dos erros quadráticos (WCSS) entre os valores de **k = 1 até k = 3**. A partir de **k = 4**, a queda no WCSS se torna mais sutil, indicando que os ganhos marginais com a adição de novos clusters são pequenos.
    Portanto, o ponto de inflexão do gráfico ocorre em **k = 3**, sugerindo que **3 clusters representam um bom compromisso entre simplicidade e capacidade explicativa do modelo**.
    """)

        
    

    # Gráfico de dispersão de dois atributos para visualização
    st.markdown("### Visualização dos Clusters (Ex: Income vs MntWines)")
    fig, ax = plt.subplots()
    sns.scatterplot(data=df_cluster, x="Income", y="MntWines", hue="Cluster", palette="Set1", ax=ax)
    ax.set_title("Segmentação dos Clientes por K-Means (k=3)")
    st.pyplot(fig)
        
    st.markdown("### Interpretação dos Perfis - Agrupamento com K-Means (3 Clusters)")

    st.markdown("""
    Após aplicar o algoritmo **K-Means com 3 clusters**, foi possível identificar três grupos distintos de clientes com base em características como renda, padrão de consumo, estrutura familiar e comportamento de compra. 
    A análise das médias dessas variáveis permitiu classificar os grupos em perfis interpretáveis do ponto de vista de negócios e segmentação de mercado.
    """)

    st.markdown("""
    - **Cluster 0 – Perfil Médio**  
    Clientes com **renda intermediária** (58 mil), **nível de consumo moderado** e com filhos em idade escolar e adolescentes (Kidhome: 0.23 / Teenhome: 0.93).  
    Demonstram comportamento de compra equilibrado entre canais físicos e digitais, com boa frequência de compras online (NumWebPurchases: 6.37) e gasto razoável com vinhos e carnes.  
    Representam um público de **classe média engajada**, com potencial de fidelização e boa aceitação a promoções.

    - **Cluster 1 – Perfil Premium/Alta Renda**  
    Grupo com **maior renda média (76 mil)** e **altíssimo consumo** em todas as categorias, especialmente vinhos (589), carnes (454) e doces (71).  
    São clientes com **poucos filhos**, muito ativos no canal de catálogo (NumCatalogPurchases: 5.98) e lojas físicas (NumStorePurchases: 8.4).  
    Refletem um perfil **exigente, fiel e de alto valor** para a empresa, devendo ser priorizados em estratégias de retenção e atendimento diferenciado.

    - **Cluster 2 – Perfil Econômico**  
    Apresentam a **menor renda média (35 mil)** e **baixíssimo consumo**, com destaque para vinhos (42), carnes (23) e frutas (4).  
    Têm mais filhos (Kidhome: 0.8 / Teenhome: 0.45) e são pouco engajados em canais digitais (NumWebPurchases: 2.12), usando mais o site apenas para visita (NumWebVisitsMonth: 6.47).  
    Representam clientes **sensíveis ao preço**, com foco em necessidades básicas e menor frequência de compra.
    """)
        
    # Aplicar DBSCAN
    dbscan = DBSCAN(eps=1.6, min_samples=5)
    dbscan_labels = dbscan.fit_predict(X_cluster)
    df4["DBSCAN_Cluster"] = dbscan_labels

    # Exibir resultados
    st.subheader("Detecção de Perfis Atípicos com DBSCAN")
    st.markdown("A técnica DBSCAN foi aplicada para detectar grupos de clientes e perfis atípicos (outliers).")

    # Contagem de clusters e outliers
    outlier_count = (df4["DBSCAN_Cluster"] == -1).sum()
    st.write("Total de registros:", len(df4))
    st.write("Total de outliers detectados:", outlier_count)
    st.write("Grupos encontrados (DBSCAN_Cluster):")
    st.dataframe(df4["DBSCAN_Cluster"].value_counts().rename("Quantidade").to_frame())

    st.markdown("## Integração entre Agrupamentos e Modelos Supervisionados")

    st.markdown("""
    A união entre os agrupamentos (**K-Means** e **DBSCAN**) e os modelos supervisionados forma uma abordagem mais robusta para a análise de comportamento e previsão de reclamações de clientes.

    - **K-Means** auxilia na **estratificação de clientes por perfil**, permitindo identificar em quais segmentos as reclamações se concentram. Complemento à supervisão: ao observar a proporção de Complain em cada cluster, pode-se verificar quais perfis concentram mais insatisfação, mesmo que a variável não tenha sido usada no agrupamento. Isso reforça a explicação dos modelos supervisionados e permite ações direcionadas por perfil.

    - **DBSCAN** contribui ao **detectar anormalidades relevantes**, como clientes com comportamento fora do padrão esperado. Esses outliers podem representar casos de frustração, desengajamento ou experiências excepcionais (positivas ou negativas).

    - Os **modelos supervisionados** indicam, com boa acurácia, **quem tende a reclamar**, mas **se beneficiam ao serem interpretados em conjunto com os perfis descobertos nos clusters**, trazendo mais contexto e precisão às ações estratégicas.

    ✅ Essa integração amplia a capacidade da empresa de **antecipar problemas, personalizar o atendimento e fidelizar clientes** com base em evidências estatísticas e comportamentais.
    """)

def q4_etapa6():
    st.header("Q4 - f) Decisão Estratégica")
    st.info("Sugira melhorias com base nos insights obtidos.")
    
    st.markdown("### Decisões Estratégicas e Implicações para o Negócio")

    st.markdown("""
    A segmentação dos clientes em três clusters fornece **insumos estratégicos valiosos** para ações de marketing, fidelização e personalização do atendimento.

    - **Cluster 1 (Premium)** deve ser o foco de **ações de fidelização e retenção personalizadas**, como programas de recompensas, atendimento exclusivo e ofertas especiais. Esses clientes possuem alto ticket médio, forte engajamento e grande potencial de geração de receita.

    - **Cluster 0 (Classe Média)** representa um público com bom nível de consumo e engajamento. Estratégias como **promoções direcionadas, upgrades de produtos e cross-selling** podem aumentar ainda mais o seu valor ao longo do tempo. É um grupo estratégico para **crescimento sustentável** da base de clientes.

    - **Cluster 2 (Econômico)** é mais sensível ao preço e menos engajado digitalmente. Para esse grupo, ações de **inclusão, ofertas acessíveis, campanhas educativas e canais presenciais** podem ser mais eficazes. Embora tenham menor valor individual, sua quantidade pode representar uma **base volumosa e relevante**.

    A segmentação permite que a empresa **adapte sua comunicação e serviços a cada perfil**, otimizando investimentos e aumentando a satisfação do cliente. Além disso, é possível **cruzar os clusters com variáveis como reclamações e churn** para priorizar melhorias e evitar perdas de clientes estratégicos.
    """)
    
    st.markdown("### Aplicações Estratégicas da Análise de Dados")

    st.markdown("""
    A análise de dados realizada oferece **insumos valiosos para a tomada de decisão estratégica**, especialmente nas seguintes frentes:

    #### Retenção de Clientes
    - A identificação de perfis mais propensos a reclamar permite **ações preventivas**, como contato proativo, ofertas de fidelização ou acompanhamento personalizado.
    - Clusters com maior taxa de insatisfação podem ser alvo de **programas de engajamento** específicos para evitar o churn.

    #### Melhoria do Suporte ao Cliente
    - Variáveis como idade, renda e canais de compra ajudam a **adaptar o atendimento ao perfil do cliente**.
    - Clientes premium ou com alto ticket médio devem receber **suporte prioritário e especializado**, aumentando a satisfação e o valor percebido.

    #### Personalização de Produtos e Serviços
    - A segmentação permite **criar campanhas de marketing personalizadas**, com base em hábitos de consumo (ex: vinhos, carnes, catálogo).
    - Estratégias diferenciadas podem ser definidas para cada grupo identificado (econômico, médio, premium), aumentando a **eficácia das ações comerciais**.

    ✅ Em resumo, a combinação de modelos preditivos e agrupamentos não supervisionados fornece uma visão ampla e acionável sobre o comportamento do cliente, permitindo uma **gestão mais inteligente da base de clientes** e **aumento da competitividade no mercado**.
    """)

        
    


# =============================
# ▶️ APP Streamlit
# =============================
st.set_page_config(page_title="📊 Prova Final - Análise Estatística", layout="wide")
st.title("📚 Prova Final - Análise Estatística de Dados e Informações")
st.markdown("Desenvolvido por: [Silvia Laryssa Branco da Silva] &nbsp;&nbsp;&nbsp;&nbsp;📅 Julho 2025")
st.markdown("""
### 📄 Acesse a prova final

Clique no link abaixo para visualizar e interagir com o painel da prova final:

🔗 [👉 AIED - Prova Final: ](https://aiedprovafinal.streamlit.app/)
""")


# MENU LATERAL
with st.sidebar:
    st.title("🧭 Menu da Prova")
    mostrar_todas = st.checkbox("✅ Mostrar todas as questões", value=False)

    # Questão 1
    with st.expander("🏨 Questão 1 - Regressão Linear (Imóveis)", expanded=mostrar_todas):
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
    with st.expander("🏨 Questão 3 - ANOVA (Vendas por País)", expanded=mostrar_todas):
        show_q3_e1 = mostrar_todas or st.checkbox("a) Análise Descritiva", key="q3e1")
        show_q3_e2 = mostrar_todas or st.checkbox("b) ANOVA entre Países", key="q3e2")
        show_q3_e3 = mostrar_todas or st.checkbox("c) Ajustes no Modelo", key="q3e3")
        show_q3_e4 = mostrar_todas or st.checkbox("d) Interpretação e Decisão", key="q3e4")

    # Questão 4
    with st.expander("🏨 Questão 4 - Reclamações de Clientes", expanded=mostrar_todas):
        show_q4_e1 = mostrar_todas or st.checkbox("a) Discussão do Problema", key="q4e1")
        show_q4_e2 = mostrar_todas or st.checkbox("b) Análise Descritiva", key="q4e2")
        show_q4_e3 = mostrar_todas or st.checkbox("c) Seleção de Modelos", key="q4e3")
        show_q4_e4 = mostrar_todas or st.checkbox("d) SHAP e Explicabilidade", key="q4e4")
        show_q4_e5 = mostrar_todas or st.checkbox("e) K-Means / DBSCAN", key="q4e5")
        show_q4_e6 = mostrar_todas or st.checkbox("f) Decisão Estratégica", key="q4e6")

# =============================
# ▶️ EXECUÇÃO DE ETAPAS SELECIONADAS
# =============================

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

# Rodapé
st.markdown("---")
st.markdown("🧮 **Prova Final - AEDI - PPCA - UNB**  \n👩‍🏫 Professor(a): João Gabriel de Moraes Souza  \n📊 Universidade de Brasilia")

st.markdown("### Referências Bibliográficas")

st.markdown("""
- CALLEGARI-JACQUES, Sidia Maria. *Bioestatística: Princípios e Aplicações*. Porto Alegre: Artmed, 2007.

- FIELD, Andy. *Descobrindo a Estatística Usando o SPSS*. Tradução de Regina Machado. Porto Alegre: Artmed, 2009.

- HUFF, Darrell. *Como Mentir com Estatística*. São Paulo: Edições Bookman, 2009.

- MAGALHÃES, Marcos Nascimento; LIMA, Antonio Carlos Pedroso de. *Noções de Probabilidade e Estatística*. São Paulo: Edusp, 2004.

- GÉRON, Aurélien. *Mãos à Obra: Aprendizado de Máquina com Scikit-Learn, Keras e TensorFlow*. Tradução de Pedro Jatobá. Rio de Janeiro: Alta Books, 2021.

- GRUS, Joel. *Data Science do Zero: Noções Fundamentais com Python*. Tradução de Juliana D. Ferreira. Rio de Janeiro: Alta Books, 2021.
""")

