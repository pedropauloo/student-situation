import pandas as pd
import os
import numpy as np


def get_csv_from(pasta_csv):
    dfs = []
    arquivos = [f for f in os.listdir(pasta_csv) if f.endswith(".csv")]

    if not arquivos:
        print("Nenhum arquivo .csv encontrado na pasta.")
        return pd.DataFrame()

    for arquivo in arquivos:
        caminho_completo = os.path.join(pasta_csv, arquivo)
        try:
            df = pd.read_csv(caminho_completo, sep=";", encoding="utf-8")
            dfs.append(df)
        except Exception as e:
            print(f"Erro ao ler {arquivo}: {e}")

    df_concatenado = pd.concat(dfs, ignore_index=True)
    return df_concatenado


def get_dados_pessoais():
    dados_pessoais = get_csv_from("./dados/dados_pessoais_discentes")
    dados_pessoais.columns = dados_pessoais.columns.str.strip().str.lower()
    dados_pessoais["id_discente"] = (
        dados_pessoais["id_discente"].astype(str).str.strip().str.lower()
    )
    dados_pessoais.rename(columns={"status": "status_do_discente"}, inplace=True)
    return dados_pessoais


def get_matriculas(dados_pessoais):
    matriculas = get_csv_from("./dados/matriculas")
    matriculas.columns = matriculas.columns.str.strip().str.lower()
    matriculas["discente"] = matriculas["discente"].astype(str).str.strip().str.lower()
    matriculas.rename(columns={"discente": "id_discente"}, inplace=True)
    matriculas.rename(columns={"situacao": "situacao_matricula"}, inplace=True)

    matriculas["media_final"] = matriculas["media_final"].str.replace(",", ".")
    matriculas["media_final"] = pd.to_numeric(
        matriculas["media_final"], errors="coerce"
    )
    matriculas["numero_total_faltas"] = pd.to_numeric(
        matriculas["numero_total_faltas"], errors="coerce"
    )

    matriculas["reposicao"] = matriculas["reposicao"].map({"True": 1, "False": 0})
    matriculas["reposicao"] = pd.to_numeric(
        matriculas["reposicao"], errors="coerce"
    ).fillna(0)

    matriculas = matriculas[
        matriculas["id_discente"].isin(dados_pessoais["id_discente"])
    ]
    return matriculas


def get_situacoes(dados_pessoais):
    situacoes = get_csv_from("./dados/situacao_discentes")
    situacoes.columns = situacoes.columns.str.strip().str.lower()
    situacoes["id_discente"] = (
        situacoes["id_discente"].astype(str).str.strip().str.lower()
    )

    situacoes.rename(columns={"situacao": "situacao_discente_2022"}, inplace=True)
    situacoes = situacoes[situacoes["id_discente"].isin(dados_pessoais["id_discente"])]

    # Mantém apenas a linha mais recente por discente
    situacoes = situacoes.sort_values(by="data_alteracao_situacao", ascending=False)
    situacoes = situacoes.drop_duplicates(subset="id_discente", keep="first")

    return situacoes


def agrupar_disciplinas(matriculas):
    matriculas["descricao"] = matriculas["descricao"].str.upper().str.strip()

    def categorizar(desc):
        if desc in ["APROVADO", "APROVADO POR NOTA"]:
            return "aprovada"
        elif desc in [
            "REPROVADO",
            "REPROVADO POR FALTAS",
            "REPROVADO POR MÉDIA E POR FALTAS",
        ]:
            return "reprovada"
        elif desc in ["TRANCADO", "CANCELADO", "DESISTENCIA", "INDEFERIDO"]:
            return "trancada"
        return "outro"

    matriculas["categoria"] = matriculas["descricao"].apply(categorizar)

    # Deduplicar por id_discente + id_turma
    matriculas_deduplicadas = matriculas.sort_values(
        by="media_final", ascending=False
    ).drop_duplicates(subset=["id_discente", "id_turma"], keep="first")

    matriculas_deduplicadas["disciplina_aprovada"] = (
        matriculas_deduplicadas["categoria"] == "aprovada"
    ).astype(int)
    matriculas_deduplicadas["disciplina_reprovada"] = (
        matriculas_deduplicadas["categoria"] == "reprovada"
    ).astype(int)
    matriculas_deduplicadas["disciplina_trancada"] = (
        matriculas_deduplicadas["categoria"] == "trancada"
    ).astype(int)

    agrupado = matriculas_deduplicadas.groupby("id_discente", as_index=False).agg(
        {
            "media_final": "mean",
            "numero_total_faltas": "sum",
            "reposicao": "sum",
            "disciplina_aprovada": "sum",
            "disciplina_reprovada": "sum",
            "disciplina_trancada": "sum",
        }
    )

    agrupado["media_final"] = agrupado["media_final"].round(1)
    agrupado["numero_total_faltas"] = agrupado["numero_total_faltas"].round(1)
    agrupado["reposicao"] = agrupado["reposicao"].round(1)

    return agrupado


def run():
    dados_pessoais = get_dados_pessoais()
    matriculas = get_matriculas(dados_pessoais)
    situacoes = get_situacoes(dados_pessoais)
    dados_agrupados = agrupar_disciplinas(matriculas)

    df = pd.merge(dados_pessoais, dados_agrupados, on="id_discente", how="inner")
    df = pd.merge(df, situacoes, on="id_discente", how="inner")

    # Criar campo de semestre e contar semestres únicos
    df["semestre"] = (
        df["ano_alteracao_situacao"].astype(str)
        + "."
        + df["periodo_alteracao_situacao"].astype(str)
    )
    semestres_por_discente = (
        df.groupby("id_discente")["semestre"].nunique().reset_index()
    )
    semestres_por_discente.rename(
        columns={"semestre": "semestres_cursados"}, inplace=True
    )

    # Renomear colunas úteis
    df.rename(
        columns={
            "ano_ingresso_x": "ano_ingresso",
            "periodo_ingresso_x": "periodo_ingresso",
            "curso_x": "curso",
            "nivel_ensino_x": "nivel_ensino",
        },
        inplace=True,
    )

    # Remover colunas desnecessárias
    colunas_para_remover = [
        "id_discente",  # se quiser anonimizar, remova
        "bairro",
        "municipio",
        "estado",
        "estado_origem",
        "cidade_origem",
        "descricao_tipo_cota",
        "curso_y",
        "curso",
        "nivel_ensino_y",
        "nivel_ensino",
        "ano_ingresso_y",
        "periodo_ingresso_y",
        "id_curso",
        "situacao_discente_2022",
        "data_alteracao_situacao",
        "ano_alteracao_situacao",
        "periodo_alteracao_situacao",
        "semestre",
        "tipo_cota",
    ]
    # Adiciona a coluna de semestres cursados ANTES de remover id_discente
    df = pd.merge(df, semestres_por_discente, on="id_discente", how="left")

    # Agora remove colunas
    df.drop(
        columns=[col for col in colunas_para_remover if col in df.columns], inplace=True
    )

    # Reorganiza colunas para deixar o target no final (status)
    colunas = [col for col in df.columns if col != "status_do_discente"] + [
        "status_do_discente"
    ]
    df = df[colunas]

    # Exporta
    df.to_csv("dados_completos.csv", index=False, sep=";", encoding="utf-8")


if __name__ == "__main__":
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

# meu id 8514817b733fca321fabc5d81b146c21
