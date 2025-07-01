import pandas as pd

caminho_discentes = "discentes.csv"
caminho_situacao = "situacao_discentes.csv"

df_discentes = pd.read_csv(caminho_discentes, sep=";")
df_situacao = pd.read_csv(caminho_situacao, sep=";")

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

df_merged.sort_values(
    by=["matricula", "ano_alteracao_situacao"], ascending=[True, False], inplace=True
)

df_resultado = df_merged[["matricula", "nome_discente", "id_discente"]].drop_duplicates(
    subset=["matricula"]
)

print(df_resultado.head())

df_resultado.to_csv("discentes_com_id.csv", sep=";", index=False)
