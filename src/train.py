from model import criar_modelo_resnet_v2
from utils import carregar_dataset, salvar_historico
import os
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

DATASET = "../datasets/potato-dataset"
RESULTADOS = "../resultados/"
MODELO_PATH = "../modelos/modelo_resnet_v2.keras"

train_dir = os.path.join(DATASET, "train")
val_dir = os.path.join(DATASET, "val")

train_gen = carregar_dataset(train_dir)
val_gen = carregar_dataset(val_dir)

y = train_gen.classes
classes = np.unique(y)
weights = compute_class_weight(class_weight="balanced", classes=classes, y=y)
class_weight = {i: weights[i] for i in range(len(classes))}

num_classes = train_gen.num_classes

modelo = criar_modelo_resnet_v2(num_classes)

hist = modelo.fit(
    train_gen,
    validation_data=val_gen,
    epochs=25,
    class_weight=class_weight
)

modelo.save(MODELO_PATH)
print("Modelo salvo em:", MODELO_PATH)

salvar_historico(hist, RESULTADOS)
