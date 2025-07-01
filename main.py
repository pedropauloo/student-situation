import pandas as pd

# Caminhos dos arquivos CSV
caminho_discentes = "discentes.csv"
caminho_situacao = "situacao_discentes.csv"

# Leitura dos arquivos com separador ";"
df_discentes = pd.read_csv(caminho_discentes, sep=";")
df_situacao = pd.read_csv(caminho_situacao, sep=";")

# Merge (junção) com base nos campos comuns
df_merged = pd.merge(
    df_discentes,
    df_situacao,
    left_on=[
        "ano_ingresso",
        "periodo_ingresso",
        "nivel_ensino",
        "id_curso",
        "nome_curso",
    ],
    right_on=["ano_ingresso", "periodo_ingresso", "nivel_ensino", "id_curso", "curso"],
    how="inner",
)

# Ordena para garantir que o id_discente mais recente venha primeiro (se essa lógica for válida)
df_merged.sort_values(
    by=["matricula", "ano_alteracao_situacao"], ascending=[True, False], inplace=True
)

# Mantém apenas uma linha por matrícula (a mais recente)
df_resultado = df_merged[["matricula", "nome_discente", "id_discente"]].drop_duplicates(
    subset=["matricula"]
)

# Mostra os primeiros resultados no console
print(df_resultado.head())

# Salva o resultado em um novo CSV
df_resultado.to_csv("discentes_com_id.csv", sep=";", index=False)
