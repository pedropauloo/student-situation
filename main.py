import pandas as pd
import os
import unicodedata

ANO_ATUAL = 2025


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
    
    dados_pessoais = dados_pessoais[
        dados_pessoais["nivel_ensino"].str.strip().str.upper() == "GRADUAÇÃO"
    ]

    return dados_pessoais


def get_matriculas(dados_pessoais):
    matriculas = get_csv_from("./dados/matriculas")
    matriculas.columns = matriculas.columns.str.strip().str.lower()

    matriculas.rename(columns={"discente": "id_discente"}, inplace=True)

    matriculas = matriculas.loc[:, ~matriculas.columns.duplicated()]

    matriculas["id_discente"] = (
        matriculas["id_discente"].astype(str).str.strip().str.lower()
    )

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
    if situacoes.empty:
        return situacoes
    situacoes.columns = situacoes.columns.str.strip().str.lower()
    situacoes["id_discente"] = (
        situacoes["id_discente"].astype(str).str.strip().str.lower()
    )
    situacoes.rename(columns={"situacao": "situacao_discente_2022"}, inplace=True)
    situacoes = situacoes[situacoes["id_discente"].isin(dados_pessoais["id_discente"])]

    situacoes["situacao_discente_2022"] = (
        situacoes["situacao_discente_2022"].str.strip().str.upper()
    )
    situacoes["situacao_discente_2022"] = situacoes["situacao_discente_2022"].apply(
        lambda x: (
            unicodedata.normalize("NFKD", x).encode("ASCII", "ignore").decode("ASCII")
            if pd.notna(x)
            else x
        )
    )

    return situacoes


def categorizar_descricao(descricao):
    if pd.isna(descricao):
        return "outros"
    desc = descricao.strip().upper()
    if "APROVADO" in desc or "DISPENSADO" in desc:
        return "aprovada"
    elif "REPROVADO" in desc:
        return "reprovada"
    elif "TRANCADO" in desc or "CANCELADO" in desc or "DESISTENCIA" in desc:
        return "trancada"
    elif "INDEFERIDO" in desc:
        return "indeferido"
    else:
        return "outros"


def get_dataset(dados_pessoais, matriculas, situacoes):
    for dataframe in [dados_pessoais, matriculas, situacoes]:
        dataframe["id_discente"] = (
            dataframe["id_discente"].astype(str).str.strip().str.lower()
        )

    matriculas["categoria"] = matriculas["descricao"].apply(categorizar_descricao)

    disciplinas_cursadas = (
        matriculas.groupby("id_discente")["id_turma"]
        .nunique()
        .reset_index(name="disciplinas_cursadas")
    )

    contagem_categorias = (
        matriculas.pivot_table(
            index="id_discente",
            columns="categoria",
            values="id_turma",
            aggfunc="nunique",
            fill_value=0,
        )
        .rename(columns=lambda col: f"disciplina_{col}")
        .reset_index()
    )

    media_geral = (
        matriculas.groupby("id_discente")["media_final"]
        .mean()
        .reset_index(name="media_final_geral")
    )

    std_media = (
        matriculas.groupby("id_discente")["media_final"]
        .std()
        .fillna(0)
        .reset_index(name="std_media_final")
    )

    faltas_unicas = (
        matriculas.groupby(["id_discente", "id_turma"])["numero_total_faltas"]
        .first()
        .reset_index()
    )

    total_faltas = (
        faltas_unicas.groupby("id_discente")["numero_total_faltas"]
        .sum()
        .reset_index(name="total_faltas")
    )

    total_reposicoes = (
        matriculas.groupby("id_discente")["reposicao"]
        .sum()
        .reset_index(name="total_reposicoes")
    )

    situacoes["semestre"] = (
        situacoes["ano_alteracao_situacao"].astype(str)
        + "."
        + situacoes["periodo_alteracao_situacao"].astype(str)
    )

    semestres_cursados = (
        situacoes.groupby("id_discente")["semestre"]
        .nunique()
        .reset_index(name="semestres_cursados")
    )

    ultima_situacao = (
        situacoes.sort_values("data_alteracao_situacao")
        .groupby("id_discente")
        .tail(1)[["id_discente", "situacao_discente_2022"]]
    )

    df_features = dados_pessoais.copy()

    df_features["ano_nascimento"] = pd.to_numeric(
        df_features["ano_nascimento"], errors="coerce"
    )
    df_features["idade"] = ANO_ATUAL - df_features["ano_nascimento"]

    # df_features = df_features[
    #     (df_features["idade"] >= 15) & (df_features["idade"] <= 60)
    # ]

    df_features = df_features.merge(disciplinas_cursadas, on="id_discente", how="left")
    df_features = df_features.merge(contagem_categorias, on="id_discente", how="left")
    df_features = df_features.merge(media_geral, on="id_discente", how="left")
    df_features = df_features.merge(std_media, on="id_discente", how="left")
    df_features = df_features.merge(total_faltas, on="id_discente", how="left")
    df_features = df_features.merge(total_reposicoes, on="id_discente", how="left")

    df_features = df_features.merge(semestres_cursados, on="id_discente", how="left")

    df_features = df_features.merge(ultima_situacao, on="id_discente", how="left")

    col_numericas = [
        "disciplinas_cursadas",
        "media_final_geral",
        "std_media_final",
        "total_faltas",
        "total_reposicoes",
        "semestres_cursados",
    ]
    for col in col_numericas:
        if col in df_features.columns:
            df_features[col] = df_features[col].fillna(0)

    return df_features


def adicionar_features(df):
    df["percentual_trancamentos"] = (
        df["disciplina_trancada"] / df["disciplinas_cursadas"]
    )
    df["percentual_reprovacoes"] = (
        df["disciplina_reprovada"] / df["disciplinas_cursadas"]
    )
    df["media_faltas_por_disciplina"] = df["total_faltas"] / df["disciplinas_cursadas"]
    df["media_reposicoes_por_disciplina"] = (
        df["total_reposicoes"] / df["disciplinas_cursadas"]
    )
    df["coeficiente_variacao_nota"] = df["std_media_final"] / df["media_final_geral"]
    return df


def limpar_colunas(df):
    colunas_para_remover = [
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
        "situacao_discente_2022",
    ]

    colunas_existentes = [col for col in colunas_para_remover if col in df.columns]
    df = df.drop(columns=colunas_existentes)

    if "cotista" in df.columns:
        df["cotista"] = (
            df["cotista"].astype(str).str.strip().str.lower().map({"t": 1, "f": 0})
        )
        df["cotista"] = df["cotista"].fillna(0).astype(int)

    return df


def cirar_target(df):
    df["curso_trancado"] = (
        df["status_do_discente"]
        .str.upper()
        .apply(lambda x: int("TRANCADO" in x or "CANCELADO" in x) if pd.notna(x) else 0)
    )
    return df


def round_numeric_columns(df):
    colunas_numericas = df.select_dtypes(include=["float", "float64"]).columns
    df[colunas_numericas] = df[colunas_numericas].round(2)
    return df


def run():
    dados_pessoais = get_dados_pessoais()
    matriculas = get_matriculas(dados_pessoais)
    situacoes = get_situacoes(dados_pessoais)

    df_final = get_dataset(dados_pessoais, matriculas, situacoes)
    df_final = adicionar_features(df_final)
    df_final = cirar_target(df_final)
    df_final = limpar_colunas(df_final)
    df_final = round_numeric_columns(df_final)

    df_final.to_csv("dados_completos.csv", index=False, sep=";", encoding="utf-8")


if __name__ == "__main__":
    run()
