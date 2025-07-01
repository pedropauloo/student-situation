import pandas as pd

def run():
    dados_pessoais = pd.read_csv("dados_pessoais_discentes.csv", sep=';', encoding='utf-8')
    matriculas = pd.read_csv("matriculas.csv", sep=';', encoding='utf-8')
    situacoes = pd.read_csv("situacao_discentes.csv", sep=';', encoding='utf-8')

    dados_pessoais.columns = dados_pessoais.columns.str.strip().str.lower()
    matriculas.columns = matriculas.columns.str.strip().str.lower()
    situacoes.columns = situacoes.columns.str.strip().str.lower()

    dados_pessoais['id_discente'] = dados_pessoais['id_discente'].astype(str).str.strip().str.lower()
    matriculas['discente'] = matriculas['discente'].astype(str).str.strip().str.lower()
    situacoes['id_discente'] = situacoes['id_discente'].astype(str).str.strip().str.lower()

    matriculas.rename(columns={"discente": "id_discente"}, inplace=True)
    matriculas["media_final"] = matriculas["media_final"].str.replace(",", ".")
    matriculas["media_final"] = pd.to_numeric(matriculas["media_final"], errors="coerce")
    matriculas["numero_total_faltas"] = pd.to_numeric(matriculas["numero_total_faltas"], errors="coerce")
    matriculas["reposicao"] = matriculas["reposicao"].map({"True": 1, "False": 0})
    matriculas["reposicao"] = pd.to_numeric(matriculas["reposicao"], errors="coerce").fillna(0)
    
    desempenho = matriculas.groupby("id_discente").agg({
        "media_final": "mean",
        "numero_total_faltas": "sum",
        "reposicao": "sum"
    }).reset_index()

    situacoes["data_alteracao_situacao"] = pd.to_datetime(situacoes["data_alteracao_situacao"], errors="coerce")
    situacoes_ultimas = situacoes.sort_values("data_alteracao_situacao").groupby("id_discente").tail(1)

    merged = pd.merge(dados_pessoais, desempenho, on="id_discente", how="left")
    merged = pd.merge(merged, situacoes_ultimas, on="id_discente", how="left")

    print(merged.head())

run()

# atributos necessários:
# quantidade de semestres
# numero de reprovas
# media de conclusao
# numero de trancamentos semestre
# numero de trancamentos matricula
# numero de faltas
# numero de reposicoes
# numero de disciplinas cursadas

# situacao final (curso)

# 1. juntar todas as tabelas (2015-2020)
# 2. remover atributos desnecessários
# 3. criar atributos necessários
# 4. treinar modelo