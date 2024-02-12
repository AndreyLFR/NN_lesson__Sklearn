from sklearn.neural_network import MLPClassifier
import numpy as np
from sklearn.metrics import classification_report, roc_auc_score, roc_curve


def results(y_test, y_pred):
    report = classification_report(y_test, y_pred, target_names=['0', '1'])
    print(report)
    print('\nПлощадь под ROC-кривой - ' + str(round(roc_auc_score(y_test, y_pred), 4)))


def m_mlpc(X_train, y_train, X_test, y_test):
    """
    Метод обучает модель MLPClassifier без использования инструментов подбора гиперпараметров
    :param X_train: обучающий массив np значений атрибутов
    :param y_train: обучающий массив target
    :param X_test: тестовый массив np значений атрибутов
    :param y_test: тестовый массив target
    :return:
    """
    params = {'solver': 'lbfgs', 'max_iter': 1800, 'hidden_layer_sizes': 4, 'random_state': 1, 'alpha': 0.4699}
    nn = MLPClassifier(**params)
    #activation='logistic', solver='adam', hidden_layer_sizes=(100, 100)
    nn.fit(X_train, y_train)
    y_pred = nn.predict(X_test)
    print('MLPC')
    results(y_test, y_pred)


# Определим набор данных
X_train = np.array([
  [-2, -1],  # Алиса
  [25, 6],   # Боб
  [17, 4],   # Чарли
  [-15, -6], # Диана
])
y_train = np.array([
  1, # Алиса
  0, # Боб
  0, # Чарли
  1, # Диана
])

X_test = np.array([
    [-7, -3],  # Эмили
    [20, 2],  # Френк
    [183-135, 83-66],  # Александр
])
y_test = np.array([
  1,
  0,
  0
])

m_mlpc(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)