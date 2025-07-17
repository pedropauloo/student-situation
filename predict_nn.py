import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt

# Configura dispositivo (GPU se disponível)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Usando dispositivo:", device)

# Carregamento e preparação dos dados
df = pd.read_csv("dados_completos.csv", sep=";")

categorical_cols = ["sexo", "raca"]
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

for col in df.select_dtypes(include=["float64", "int64"]):
    df[col] = df[col].fillna(df[col].median())

for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

X = df.drop(columns=["id_discente", "aluno_evadio", "ano_nascimento"])
y = df["aluno_evadio"].values

X_train, X_test, y_train, y_test = train_test_split(
    X.values, y, test_size=0.2, random_state=42, stratify=y
)

num_cols_idx = [
    X.columns.get_loc(c) for c in X.select_dtypes(include=["float64", "int64"]).columns
]
scaler = StandardScaler()
X_train[:, num_cols_idx] = scaler.fit_transform(X_train[:, num_cols_idx])
X_test[:, num_cols_idx] = scaler.transform(X_test[:, num_cols_idx])

# Converte para tensores e envia para dispositivo
X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)
y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1).to(device)

# Dataset e DataLoader
batch_size = 64
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


# Define modelo MLP sem sigmoid na saída
class MLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x)


model = MLP(X_train.shape[1]).to(device)

# Calcula pos_weight para classe positiva para balancear BCEWithLogitsLoss
num_pos = (y_train == 1).sum().item()
num_neg = (y_train == 0).sum().item()
pos_weight = torch.tensor(num_neg / num_pos).to(device)

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

num_epochs = 30
losses = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)  # raw logits
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    losses.append(epoch_loss)
    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

# Avaliação
model.eval()
with torch.no_grad():
    outputs = model(X_test)
    probs = torch.sigmoid(outputs)
    preds_class = (probs >= 0.5).float().cpu().numpy()
    y_true = y_test.cpu().numpy()

acc = accuracy_score(y_true, preds_class)
print(f"\nAcurácia no teste: {acc:.4f}")

print("Relatório de Classificação:")
print(classification_report(y_true, preds_class, digits=4))

print("Matriz de Confusão:")
print(confusion_matrix(y_true, preds_class))


plt.plot(range(1, num_epochs + 1), losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Curva de perda durante treinamento")
plt.show()
