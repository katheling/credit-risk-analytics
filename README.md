# 📊 Análise de Risco de Crédito & Predição de Inadimplência

Projeto de Machine Learning e Business Intelligence para avaliação de risco de crédito e previsão de inadimplência.

## Visão Geral

- **Acurácia do Modelo (Random Forest):** 93,34%[cite: 10]
- **Taxa Média de Inadimplência:** 21,82%[cite: 10]
- **Volume Analisado:** $312,43 Mi[cite: 10]
- **Total de Solicitantes:** 32.581[cite: 10]

---

## Dashboard Executivo

![Dashboard de Risco de Crédito](credit-risk-dashboard.jpg)

---

## Etapas do Projeto

1. **Tratamento de Dados:** Preenchimento de valores ausentes pela mediana e codificação de variáveis categóricas (`get_dummies`).
2. **Modelagem:** Treinamento do modelo `Random Forest Classifier` com divisão de 70% para treino e 30% para teste.
3. **Visualização:** Construção de painel interativo no Power BI destacando inadimplência por idade, histórico, motivo do empréstimo e tipo de moradia.

---

## Tecnologias

- Python (Pandas, Scikit-Learn)
- Power BI
