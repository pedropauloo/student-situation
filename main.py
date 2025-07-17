import os
import pandas as pd
import unicodedata
from datetime import datetime


def padronizar_colunas(df):
    df = df.rename(columns=str.strip).rename(columns=str.lower)
    df["id_discente"] = df["id_discente"].astype(str).str.strip().str.lower()

    return df


def agrupar(df, col, nome, agg="nunique"):
    return df.groupby("id_discente")[col].agg(agg).reset_index(name=nome)


def arredondar_colunas_numericas(df):
    colunas_float = df.select_dtypes(include=["float", "float64"]).columns
    df[colunas_float] = df[colunas_float].round(2)

    return df


def remover_acentos(texto):
    return (
        unicodedata.normalize("NFKD", texto).encode("ASCII", "ignore").decode("ASCII")
    )


def categorizar_descricao(descricao):
    if pd.isna(descricao):
        return "outros"

    desc = descricao.strip().upper()
    if "APROVADO" in desc or "DISPENSADO" in desc:
        return "aprovada"
    if "REPROVADO" in desc:
        return "reprovada"
    if any(x in desc for x in ["TRANCADO", "CANCELADO", "DESISTENCIA"]):
        return "trancada"
    if "INDEFERIDO" in desc:
        return "indeferido"
    return "outros"


def carregar_csvs_em_pasta(caminho_pasta):
    arquivos = [f for f in os.listdir(caminho_pasta) if f.endswith(".csv")]
    if not arquivos:
        print(f"Nenhum CSV encontrado em {caminho_pasta}")
        return pd.DataFrame()

    dfs = []
    for arquivo in arquivos:
        try:
            df = pd.read_csv(
                os.path.join(caminho_pasta, arquivo), sep=";", encoding="utf-8"
            )
            dfs.append(df)
        except Exception as e:
            print(f"Erro ao ler {arquivo}: {e}")

    return pd.concat(dfs, ignore_index=True)


def carregar_dados_pessoais():
    df = carregar_csvs_em_pasta("./dados/dados_pessoais_discentes")
    df = padronizar_colunas(df)

    df = df[df["nivel_ensino"].str.upper().str.strip() == "GRADUAÇÃO"]

    df.rename(columns={"status": "status_do_discente"}, inplace=True)

    return df


def carregar_matriculas(dados_pessoais):
    df = carregar_csvs_em_pasta("./dados/matriculas")

    df.rename(columns={"discente": "id_discente"}, inplace=True)

    df = padronizar_colunas(df)
    df = df.loc[:, ~df.columns.duplicated()]

    df["media_final"] = pd.to_numeric(
        df["media_final"].str.replace(",", "."), errors="coerce"
    )

    df["numero_total_faltas"] = pd.to_numeric(
        df["numero_total_faltas"], errors="coerce"
    )

    df["reposicao"] = pd.to_numeric(
        df["reposicao"].map({"True": 1, "False": 0}), errors="coerce"
    ).fillna(0)

    df = df[df["id_discente"].isin(dados_pessoais["id_discente"])]

    return df


def carregar_situacoes(dados_pessoais):
    df = carregar_csvs_em_pasta("./dados/situacao_discentes")

    if df.empty:
        return df

    df = padronizar_colunas(df)
    df = df[df["id_discente"].isin(dados_pessoais["id_discente"])]

    df["situacao"] = df["situacao"].str.upper().str.strip()
    df["situacao"] = df["situacao"].apply(
        lambda x: remover_acentos(x) if pd.notna(x) else x
    )

    return df


def construir_features(dados_pessoais, matriculas, situacoes):
    matriculas["categoria"] = matriculas["descricao"].apply(categorizar_descricao)

    features = dados_pessoais.copy()
    features["ano_nascimento"] = pd.to_numeric(
        features["ano_nascimento"], errors="coerce"
    )
    features["idade"] = datetime.now().year - features["ano_nascimento"]

    features = features.merge(
        agrupar(matriculas, "id_turma", "disciplinas_cursadas"),
        on="id_discente",
        how="left",
    )

    categorias = matriculas.pivot_table(
        index="id_discente",
        columns="categoria",
        values="id_turma",
        aggfunc="nunique",
        fill_value=0,
    )
    categorias.columns = [f"disciplina_{col}" for col in categorias.columns]
    features = features.merge(categorias.reset_index(), on="id_discente", how="left")

    features = features.merge(
        agrupar(matriculas, "media_final", "media_final_geral", agg="mean"),
        on="id_discente",
        how="left",
    )

    faltas_unicas = (
        matriculas.groupby(["id_discente", "id_turma"])["numero_total_faltas"]
        .first()
        .reset_index()
    )
    total_faltas = agrupar(
        faltas_unicas, "numero_total_faltas", "total_faltas", agg="sum"
    )
    total_reposicoes = agrupar(matriculas, "reposicao", "total_reposicoes", agg="sum")

    features = features.merge(total_faltas, on="id_discente", how="left")
    features = features.merge(total_reposicoes, on="id_discente", how="left")

    situacoes["semestre"] = (
        situacoes["ano_alteracao_situacao"].astype(str)
        + "."
        + situacoes["periodo_alteracao_situacao"].astype(str)
    )
    semestres_cursados = agrupar(situacoes, "semestre", "semestres_cursados")

    ultima_situacao = (
        situacoes.sort_values("data_alteracao_situacao").groupby("id_discente").tail(1)
    )
    ultima_situacao = ultima_situacao[["id_discente", "situacao"]]

    features = features.merge(semestres_cursados, on="id_discente", how="left")
    features = features.merge(ultima_situacao, on="id_discente", how="left")

    col_numericas = [
        "disciplinas_cursadas",
        "media_final_geral",
        "total_faltas",
        "total_reposicoes",
        "semestres_cursados",
    ]

    for col in col_numericas:
        if col in features.columns:
            features[col] = features[col].fillna(0)

    return features


def criar_target(df):
    df["aluno_evadio"] = (
        df["status_do_discente"]
        .str.upper()
        .apply(
            lambda x: int("CANCELADO" in x or "DESISTENCIA" in x) if pd.notna(x) else 0
        )
    )

    return df


def limpar_colunas_irrelevantes(df):
    colunas_remover = [
        "estado_origem",
        "cidade_origem",
        "estado",
        "municipio",
        "bairro",
        "nivel_ensino",
        "curso",
        "forma_ingresso",
        "tipo_cota",
        "descricao_tipo_cota",
        "status_do_discente",
        "situacao",
    ]

    df = df.drop(
        columns=[col for col in colunas_remover if col in df.columns], errors="ignore"
    )

    if "cotista" in df.columns:
        df["cotista"] = (
            df["cotista"]
            .astype(str)
            .str.strip()
            .str.lower()
            .map({"t": 1, "f": 0})
            .fillna(0)
            .astype(int)
        )

    return df


def remover_registros_ruidosos(df):
    cond_ruido = (
        ((df["periodo_ingresso"] == 1.0) | (df["periodo_ingresso"] == 2.0))
        & (df["cotista"] == 0)
        & (df["idade"].isna())
        & (df["disciplinas_cursadas"] == 0.0)
        & (df["disciplina_aprovada"].isna())
        & (df["disciplina_indeferido"].isna())
        & (df["disciplina_outros"].isna())
        & (df["disciplina_reprovada"].isna())
        & (df["disciplina_trancada"].isna())
        & (df["media_final_geral"] == 0.0)
        & (df["total_faltas"] == 0.0)
        & (df["total_reposicoes"] == 0.0)
        & (df["semestres_cursados"] == 0.0)
    )

    df = df[~cond_ruido]

    if "ano_ingresso" in df.columns:
        df = df[df["ano_ingresso"] >= 2010]

    if "sexo" in df.columns:
        df = df[df["sexo"].isin(["M", "F"])]

    return df


def pipeline():
    dados_pessoais = carregar_dados_pessoais()
    matriculas = carregar_matriculas(dados_pessoais)
    situacoes = carregar_situacoes(dados_pessoais)

    df = construir_features(dados_pessoais, matriculas, situacoes)
    df = criar_target(df)
    df = limpar_colunas_irrelevantes(df)
    df = arredondar_colunas_numericas(df)
    df = remover_registros_ruidosos(df)

    df.to_csv("dados_completos.csv", index=False, sep=";", encoding="utf-8")
    print("Arquivo salvo com sucesso")


if __name__ == "__main__":
    pipeline()
