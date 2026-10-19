import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 1. Importando a base em csv
df = pd.read_csv('credit_risk_dataset.csv')

# 2. Limpando os dados nulos, substituindo eles pela mediana
df['person_emp_length'] = df['person_emp_length'].fillna(df['person_emp_length'].median())
df['loan_int_rate'] = df['loan_int_rate'].fillna(df['loan_int_rate'].median())

# 3. Transformando variáveis categóricas em numéricas (One-Hot Encoding)
df_processado = pd.get_dummies(df)

# 4. Separando as características (X) e o alvo (y)
X = df_processado.drop('loan_status', axis=1) # Tudo menos a coluna de inadimplência
y = df_processado['loan_status']              # Apenas a coluna de inadimplência

# Verificação do tamanho das features e do alvo
print("Tamanho de X (features):", X.shape)
print("Tamanho de y (alvo):", y.shape)

# 5. Dividindo os dados: 70% para treino e 30% para teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Verificação do tamanho do treino e do teste
print("\nTamanho de X_train (treino):", X_train.shape)
print("Tamanho de X_test (teste):", X_test.shape)

# 6. Criar e treinar o modelo de Random Forest
modelo = RandomForestClassifier(random_state=42)
modelo.fit(X_train, y_train)

# 7. Fazer previsões e avaliar o modelo
previsoes = modelo.predict(X_test)
precisao = accuracy_score(y_test, previsoes)

# 8. Mapear a importância de cada variável
importancias = pd.Series(modelo.feature_importances_, index=X_train.columns)
importancias_ordenadas = importancias.sort_values(ascending=False)

# Resultado Final
print("\n--- Resultado Final ---")
print(f"Precisão do modelo: {precisao * 100:.2f}%")

print("\n--- Top 5 Variáveis Mais Importantes para o Modelo ---")
print(importancias_ordenadas.head(5))

df_processado.to_csv('credit_risk_processado.csv', index=False)
print("\nBase processada exportada com sucesso!")
