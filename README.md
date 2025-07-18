# 🎓 Previsão de Evasão Estudantil com Aprendizado de Máquina

Este projeto tem como objetivo desenvolver um modelo preditivo capaz de identificar, com alta acurácia, quais estudantes têm maior probabilidade de evadir (abandonar o curso). A partir da análise de dados históricos e acadêmicos dos discentes, é possível auxiliar instituições de ensino na criação de estratégias preventivas e mais eficazes para combater a evasão.

---

## 📌 Contexto

A evasão estudantil é um desafio persistente em instituições de ensino superior. Com o uso de técnicas de machine learning, torna-se viável antecipar comportamentos de evasão a partir de padrões observados nos dados dos alunos, como:

- Desempenho acadêmico (disciplinas cursadas e aprovadas)
- Informações de matrícula e ingresso
- Características demográficas (idade, cotista ou não)

---

## 📊 Desempenho do Modelo

A avaliação do modelo mostrou resultados altamente satisfatórios, com métricas equilibradas entre as classes e excelente capacidade de generalização:

- ✅ **Acurácia geral:** 0.9067

### 🔍 Relatório de Classificação

| Classe       | Precision | Recall | F1-score | Suporte |
|--------------|-----------|--------|----------|---------|
| Não Evadiu   | 0.89      | 0.93   | 0.91     | 8.590   |
| Evadido      | 0.92      | 0.88   | 0.90     | 8.068   |

O modelo é particularmente eficiente na distinção entre alunos que permanecem e os que evadem. A leve assimetria observada (falsos negativos) aponta para a importância de estratégias complementares para captar casos mais sutis de evasão.

---

## 🧠 Abordagem Técnica

- 📌 Pré-processamento de dados com `pandas`, incluindo limpeza e normalização
- 🧹 Remoção de ruídos com base em regras específicas (ex: registros com 0 disciplinas e idade ausente)
- ⚖️ Balanceamento de classes e cálculo de pesos para lidar com desequilíbrios
- 📈 Treinamento de modelos supervisionados com `scikit-learn`
- 📊 Avaliação com matriz de confusão, classificação e análise de erros

---

## 🗃️ Estrutura dos Dados

Cada linha representa um estudante com os seguintes atributos:

- `periodo_ingresso`: Período em que o aluno ingressou
- `cotista`: Indica se o aluno entrou por sistema de cotas
- `idade`: Idade do aluno no momento do ingresso
- `disciplinas_cursadas`: Total de disciplinas que o aluno cursou
- `disciplina_aprovada`: Total de disciplinas que o aluno foi aprovado
- `evadiu`: Variável-alvo binária (1 = evadiu, 0 = não evadiu)

---

## 🚧 Limitações e Futuros Aperfeiçoamentos

- Incluir novos dados dos anos de 2019 até o presente para atualizar o modelo
- Trabalhar no tratamento de efeitos atípicos provocados pela pandemia
- Realizar testes com outros algoritmos e técnicas de explicabilidade
- Criar visualizações para facilitar o entendimento de padrões de evasão

---

## 🎯 Aplicações Possíveis

- Integração com sistemas acadêmicos para monitoramento em tempo real
- Geração de alertas para equipes pedagógicas e assistenciais
- Formulação de políticas públicas e institucionais baseadas em dados
