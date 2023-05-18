import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.metrics import classification_report

# Gerar um dataset de exemplo
X, y = make_classification(n_samples=1000, n_features=10, weights=[0.9, 0.1], random_state=42)
print("Contagem de classes antes do balanceamento:")
print(np.bincount(y))

# Dividir o dataset em treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Verificar se o dataset está balanceado
class_counts = np.bincount(y_train)
is_balanced = class_counts[0] == class_counts[1]
print(class_counts[0])
print(class_counts[1])
print("O dataset está balanceado? ", is_balanced)

# Realizar RandomOverSampling na classe minoritária
X_train_over, y_train_over = resample(X_train[y_train == 1], y_train[y_train == 1],
                                      replace=True, n_samples=class_counts[0], random_state=42)
X_train_balanced_over = np.vstack((X_train[y_train == 0], X_train_over))
y_train_balanced_over = np.hstack((y_train[y_train == 0], y_train_over))
print("Contagem de classes após RandomOverSampling:")
print(np.bincount(y_train_balanced_over))

# Realizar RandomUnderSampling na classe majoritária
X_train_under, y_train_under = resample(X_train[y_train == 0], y_train[y_train == 0],
                                        replace=False, n_samples=class_counts[1], random_state=42)
X_train_balanced_under = np.vstack((X_train_under, X_train[y_train == 1]))
y_train_balanced_under = np.hstack((y_train_under, y_train[y_train == 1]))
print("Contagem de classes após RandomUnderSampling:")
print(np.bincount(y_train_balanced_under))

# Agora você pode usar X_train_balanced_over e y_train_balanced_over para treinar seu modelo com RandomOverSampling,
# ou X_train_balanced_under e y_train_balanced_under para treinar seu modelo com RandomUnderSampling.

# Lembre-se de aplicar a mesma transformação de balanceamento no conjunto de teste, se necessário, antes de avaliar o modelo.

