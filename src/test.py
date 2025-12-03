from utils import carregar_dataset, salvar_matriz_confusao, salvar_metricas
from tensorflow import keras
import numpy as np
import os

DATASET = "../datasets/potato-dataset"
RESULTADOS = "../resultados/"
MODELO_PATH = "../modelos/modelo_resnet_v2.keras"

test_dir = os.path.join(DATASET, "test")

test_gen = carregar_dataset(test_dir, shuffle=False)
modelo = keras.models.load_model(MODELO_PATH)

pred = modelo.predict(test_gen)
y_pred = np.argmax(pred, axis=1)
y_true = test_gen.classes
classes = list(test_gen.class_indices.keys())

salvar_matriz_confusao(y_true, y_pred, classes, RESULTADOS)
salvar_metricas(y_true, y_pred, classes, RESULTADOS)
