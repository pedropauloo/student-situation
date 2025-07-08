import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

df = pd.read_csv("dados_completos.csv", sep=";")

categorical_cols = ["sexo", "raca"]
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# Preenchendo valores ausentes em colunas numéricas com a mediana
for col in df.select_dtypes(include=["float64", "int64"]):
    df[col].fillna(df[col].median(), inplace=True)

# Preenchendo valores ausentes em colunas categóricas com a moda
for col in categorical_cols:
    df[col].fillna(df[col].mode()[0], inplace=True)

X = df.drop(
    columns=[
        "id_discente",
        "aluno_evadio",
        "ano_nascimento",
    ]
)
y = df["aluno_evadio"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

num_cols = X.select_dtypes(include=["float64", "int64"]).columns
scaler = StandardScaler()

X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])

clf = RandomForestClassifier(class_weight='balanced', random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print(f"Acurácia do modelo: {acc:.4f}")


cm = confusion_matrix(y_test, y_pred)
print("\nMatriz de Confusão:")
print(cm)

print("\nRelatório de Classificação:")
report = classification_report(
    y_test, y_pred, target_names=["Não Evadiu", "Evadido"], zero_division=0
)
print(report)

# tentar balancear o dataset
# treinar outros modelos