import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tensorflow import keras

def carregar_dataset(path, img_shape=(224,224), batch_size=32, shuffle=True):
    gen = keras.preprocessing.image.ImageDataGenerator(
        rescale=1/255,
        # rotation_range=25,           # Rotação aleatória até 25°
        # zoom_range=0.20,             # Zoom 20%
        # horizontal_flip=True,        # espelhamento
        # brightness_range=[0.5, 1.5], # imagens mais claras/escuras
        # channel_shift_range=30,      # pequena alteração de cor
        fill_mode="nearest"
        )
    
    return gen.flow_from_directory(
        path,
        target_size=img_shape,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=shuffle
    )

def salvar_historico(hist, out_dir):
    plt.figure(figsize=(10,6))
    plt.plot(hist.history["accuracy"], label="Treino")
    plt.plot(hist.history["val_accuracy"], label="Validação")
    plt.title("Acurácia ao Longo do Treinamento")
    plt.legend()
    plt.savefig(os.path.join(out_dir, "historico_treinamento.png"))
    plt.close()

def salvar_matriz_confusao(y_true, y_pred, classes, out_dir):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True, cmap="Blues")
    plt.xlabel("Previsto")
    plt.ylabel("Real")
    plt.title("Matriz de Confusão")
    plt.savefig(os.path.join(out_dir, "matriz_confusao.png"))
    plt.close()

def salvar_metricas(y_true, y_pred, classes, out_dir):
    report = classification_report(
        y_true, y_pred, target_names=classes, output_dict=True
    )

    for metric in ["precision", "recall", "f1-score"]:
        values = [report[c][metric] for c in classes]
        plt.figure(figsize=(10,5))
        sns.barplot(x=classes, y=values)
        plt.title(f"{metric} por Classe")
        plt.savefig(os.path.join(out_dir, f"{metric}.png"))
        plt.close()
