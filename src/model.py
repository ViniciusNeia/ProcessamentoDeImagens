import tensorflow as tf
from tensorflow import keras

def criar_modelo_resnet_v2(num_classes, input_shape=(224, 224, 3)):
    base = keras.applications.ResNet50V2(
        include_top=False,
        weights="imagenet",
        input_shape=input_shape
    )

    for layer in base.layers[:-20]:
        layer.trainable = False

    # base.trainable = False

    inputs = keras.Input(shape=input_shape)

    x = keras.applications.resnet_v2.preprocess_input(inputs)
    x = base(x, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(0.3)(x)
    x = keras.layers.Dense(128, activation="relu")(x)
    x = keras.layers.Dropout(0.5)(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs)

    model.compile(
        optimizer=keras.optimizers.Adam(0.0001),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.summary()
    
    return model
