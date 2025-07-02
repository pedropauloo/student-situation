import pandas as pd
import os

def get_csv_from(pasta_csv):
    dfs = []
    arquivos = [f for f in os.listdir(pasta_csv) if f.endswith('.csv')]

    if not arquivos:
        print("Nenhum arquivo .csv encontrado na pasta.")
        return pd.DataFrame()

    for arquivo in arquivos:
        caminho_completo = os.path.join(pasta_csv, arquivo)
        try:
            df = pd.read_csv(caminho_completo, sep=';', encoding='utf-8')
            dfs.append(df)
        except Exception as e:
            print(f"Erro ao ler {arquivo}: {e}")

    df_concatenado = pd.concat(dfs, ignore_index=True)
    return df_concatenado

def get_dados_pessoais():
    dados_pessoais = get_csv_from("./dados/dados_pessoais_discentes")
    dados_pessoais.columns = dados_pessoais.columns.str.strip().str.lower()
    dados_pessoais['id_discente'] = dados_pessoais['id_discente'].astype(str).str.strip().str.lower()
    
    return dados_pessoais

def get_matriculas(dados_pessoais):
    matriculas = get_csv_from("./dados/matriculas")
    matriculas.columns = matriculas.columns.str.strip().str.lower()
    
    matriculas['discente'] = matriculas['discente'].astype(str).str.strip().str.lower()
    matriculas.rename(columns={"discente": "id_discente"}, inplace=True)
    
    matriculas["media_final"] = matriculas["media_final"].str.replace(",", ".")
    matriculas["media_final"] = pd.to_numeric(matriculas["media_final"], errors="coerce")
    
    matriculas["numero_total_faltas"] = pd.to_numeric(matriculas["numero_total_faltas"], errors="coerce")
    
    matriculas["reposicao"] = matriculas["reposicao"].map({"True": 1, "False": 0})
    matriculas["reposicao"] = pd.to_numeric(matriculas["reposicao"], errors="coerce").fillna(0)
    
    matriculas = matriculas[matriculas['id_discente'].isin(dados_pessoais['id_discente'])]

    return matriculas

def get_situacoes(dados_pessoais):
    situacoes = get_csv_from("./dados/situacao_discentes")
    situacoes.columns = situacoes.columns.str.strip().str.lower()
    situacoes['id_discente'] = situacoes['id_discente'].astype(str).str.strip().str.lower()
    
    situacoes = situacoes[situacoes['id_discente'].isin(dados_pessoais['id_discente'])]
    
    return situacoes

def run():
    dados_pessoais = get_dados_pessoais()
    matriculas = get_matriculas(dados_pessoais)
    situacoes = get_situacoes(dados_pessoais)

    df = pd.merge(dados_pessoais, matriculas, on='id_discente', how='inner')
    df = pd.merge(df, situacoes, on='id_discente', how='inner')

    df = df.sort_values(by='data_alteracao_situacao', ascending=False).drop_duplicates(subset=['id_discente', 'id_turma', 'unidade'], keep='first')
    df = df.sort_values(by=['id_discente','id_turma', 'unidade'])

    df.to_csv("dados_completos.csv", index=False, sep=';', encoding='utf-8')

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
