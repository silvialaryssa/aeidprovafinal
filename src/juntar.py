import pandas as pd

# Caminho do arquivo original
caminho_arquivo = "src/online_retail_II.xlsx"  # Substitua pelo caminho real do seu arquivo

# Número máximo de linhas por aba a serem lidas (ajuste conforme necessário)
limite_linhas = 538171

# Carregar o arquivo Excel
xls = pd.ExcelFile(caminho_arquivo)

# Listar as abas
abas = xls.sheet_names

# Lê cada aba com limite de linhas e adiciona à lista
dataframes = []
for aba in abas:
    print(f"Lendo aba: {aba}")
    df = xls.parse(aba, nrows=limite_linhas)
    #df["__origem_aba__"] = aba  # opcional: adiciona coluna para saber de qual aba veio
    dataframes.append(df)

# Combina todas as abas em um único DataFrame
df_combinado = pd.concat(dataframes, ignore_index=True)

# Salva o resultado em uma nova planilha Excel
arquivo_saida = "planilha_combinada_amostrada.xlsx"
df_combinado.to_csv("planilha_combinada.csv", index=False)

print(f"Arquivo combinado salvo como: {arquivo_saida}")
