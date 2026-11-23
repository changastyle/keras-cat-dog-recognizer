"""Entrena la CNN perros-vs-gatos y exporta a ONNX.

Reproduce el flujo del notebook de Colab:
1. Entrena modeloCNN3.keras: Conv2D(32/64/128, 3x3, valid) + MaxPool x3 ->
   GlobalAveragePooling -> Dropout -> Dense(1, sigmoide), con capas de
   aumento (RandomFlip/RandomRotation) solo activas en entrenamiento.
2. Reconstruye un modelo de inferencia sin aumentos y con Dropout fijo.
3. Exporta ONNX con model.export(format="onnx") (Keras 3 + tf2onnx).

La app web preprocesa cada fotograma a 100x100 escala de grises /255 y
trata la salida como P(perro). Clases: gatos -> 0, perros -> 1.

Uso:
    pip install -r requirements.txt
    python train.py --data data          # data/{gatos,perros}/*.jpg
    python train.py --tfds               # cats_vs_dogs de TF Datasets

Cada run exporta ../modelo/perros-gatos-cnn-ad3-YYYYMMDD.onnx y actualiza
el archivo canonico que sirve la web (salvo --no-deploy).
"""
import argparse
import datetime
import os
import shutil

import keras
import numpy as np
import tensorflow as tf

IMG = 100
BATCH = 32
EPOCHS = 15
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".gif", ".jfif")


def _list_image_files(root):
    """Class folders sorted alphabetically (gatos -> 0, perros -> 1)."""
    paths, labels = [], []
    for label, name in enumerate(sorted(os.listdir(root))):
        class_dir = os.path.join(root, name)
        if name.startswith("_") or not os.path.isdir(class_dir):
            continue
        for fname in sorted(os.listdir(class_dir)):
            if fname.lower().endswith(IMG_EXTS):
                paths.append(os.path.join(class_dir, fname))
                labels.append(label)
    return paths, labels


def _quarantine_corrupt(root, paths, labels):
    """One-time decode check; unreadable files move to <root>/_quarantine/."""
    marker = os.path.join(root, ".scan_done")
    if os.path.exists(marker):
        return paths, labels
    ok_p, ok_l, bad = [], [], 0
    for path, label in zip(paths, labels):
        try:
            tf.io.decode_image(
                tf.io.read_file(path), channels=3, expand_animations=False
            )
        except Exception:
            bad += 1
            qdir = os.path.join(
                root, "_quarantine", os.path.basename(os.path.dirname(path))
            )
            os.makedirs(qdir, exist_ok=True)
            shutil.move(path, os.path.join(qdir, os.path.basename(path)))
            continue
        ok_p.append(path)
        ok_l.append(label)
    if bad:
        print(f"Quarantined {bad} unreadable images -> {root}/_quarantine")
    with open(marker, "w") as f:
        f.write("done\n")
    return ok_p, ok_l


def _decode(path, label):
    # channels=3 decodes JPEG/PNG/GIF/BMP uniformly (some .jpg files are really BMP).
    img = tf.io.decode_image(
        tf.io.read_file(path), channels=3, expand_animations=False
    )
    img = tf.image.rgb_to_grayscale(tf.image.resize(img, (IMG, IMG))) / 255.0
    return img, tf.reshape(label, [1])


def _directory_datasets(root):
    paths, labels = _quarantine_corrupt(root, *_list_image_files(root))
    n = len(paths)
    order = np.random.RandomState(42).permutation(n)
    paths, labels = np.array(paths)[order], np.array(labels, dtype=np.float32)[order]
    n_val = int(0.2 * n)

    def make(paths_, labels_, shuffle):
        ds = tf.data.Dataset.from_tensor_slices((paths_, labels_))
        if shuffle:
            ds = ds.shuffle(len(paths_), seed=42, reshuffle_each_iteration=True)
        return ds.map(_decode, num_parallel_calls=tf.data.AUTOTUNE).batch(
            BATCH
        ).prefetch(tf.data.AUTOTUNE)

    return (
        make(paths[:-n_val], labels[:-n_val], shuffle=True),
        make(paths[-n_val:], labels[-n_val:], shuffle=False),
    )


def load_datasets(args):
    if args.tfds:
        # NOTE: tfds cats_vs_dogs generation is broken on Windows (zip path
        # separator bug); use --data with the extracted PetImages zip instead.
        import tensorflow_datasets as tfds

        (train_raw, val_raw), _ = tfds.load(
            "cats_vs_dogs",
            split=["train[:80%]", "train[80%:]"],
            as_supervised=True,
            with_info=True,
        )
        prep = lambda img, label: (  # noqa: E731
            tf.image.rgb_to_grayscale(tf.image.resize(img, (IMG, IMG))) / 255.0,
            tf.cast(label, tf.float32),  # 0=gato, 1=perro
        )
        return (
            train_raw.map(prep).shuffle(2048).batch(BATCH).prefetch(tf.data.AUTOTUNE),
            val_raw.map(prep).batch(BATCH).prefetch(tf.data.AUTOTUNE),
        )

    return _directory_datasets(args.data)


def build_model(dropout=0.3):
    """Arquitectura de modeloCNN3 (padding valid, cabeza GAP como el ONNX servido).

    Sin capa Rescaling: la app web ya normaliza a 0-1 antes de inferir.
    """
    return keras.Sequential(
        [
            keras.Input(shape=(IMG, IMG, 1)),
            keras.layers.RandomFlip("horizontal"),
            keras.layers.RandomRotation(0.1),
            keras.layers.Conv2D(32, 3, activation="relu"),
            keras.layers.MaxPooling2D(),
            keras.layers.Conv2D(64, 3, activation="relu"),
            keras.layers.MaxPooling2D(),
            keras.layers.Conv2D(128, 3, activation="relu"),
            keras.layers.MaxPooling2D(),
            keras.layers.GlobalAveragePooling2D(),
            keras.layers.Dropout(dropout),
            keras.layers.Dense(1, activation="sigmoid"),  # P(perro)
        ],
        name="modeloCNN3",
    )


def build_inference_model(modelo):
    """Copia del notebook: quita RandomFlip/RandomRotation y fija Dropout."""
    entrada = keras.Input(shape=(IMG, IMG, 1))
    x = entrada
    for capa in modelo.layers:
        if isinstance(capa, (keras.layers.RandomFlip, keras.layers.RandomRotation)):
            continue
        if isinstance(capa, keras.layers.Dropout):
            x = capa(x, training=False)
        else:
            x = capa(x)
    return keras.Model(inputs=entrada, outputs=x, name="perros_gatos_inferencia")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data", help="carpeta con subcarpetas gatos/ y perros/")
    p.add_argument("--tfds", action="store_true", help="usar cats_vs_dogs de TF Datasets")
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument(
        "--outdir",
        default=os.path.join(os.path.dirname(__file__), "..", "modelo"),
        help="carpeta de salida del ONNX (por defecto ../modelo)",
    )
    p.add_argument(
        "--no-deploy",
        action="store_true",
        help="no actualizar perros-gatos-cnn-ad3.onnx (el archivo que sirve la web)",
    )
    args = p.parse_args()

    train_ds, val_ds = load_datasets(args)
    modelo = build_model()
    modelo.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    modelo.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=[keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True)],
    )

    os.makedirs("checkpoints", exist_ok=True)
    modelo.save("checkpoints/modeloCNN3.keras")

    modelo_inferencia = build_inference_model(modelo)

    # Verificacion identica a la del notebook.
    prueba = np.random.default_rng(42).random((1, IMG, IMG, 1)).astype("float32")
    original = modelo(prueba, training=False).numpy()
    nuevo = modelo_inferencia(prueba, training=False).numpy()
    print("Prediccion original:", original)
    print("Prediccion nueva:", nuevo)
    assert np.allclose(original, nuevo), "el modelo de inferencia difiere del original"

    fecha = datetime.date.today().strftime("%Y%m%d")
    os.makedirs(args.outdir, exist_ok=True)
    dated = os.path.join(args.outdir, f"perros-gatos-cnn-ad3-{fecha}.onnx")
    modelo_inferencia.export(dated, format="onnx")
    print(f"ONNX exportado: {dated}")
    if not args.no_deploy:
        deploy = os.path.join(args.outdir, "perros-gatos-cnn-ad3.onnx")
        shutil.copyfile(dated, deploy)
        print(f"Desplegado para la web: {deploy}")
    modelo_inferencia.summary()


if __name__ == "__main__":
    main()
