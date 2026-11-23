# Entrenamiento

Documentación completa del pipeline en [README-training.md](README-training.md).

Entrena la CNN perros-vs-gatos y exporta el ONNX a `../modelo/` para servirlo en la web.

Reproduce el flujo del notebook de Colab:

1. Entrena `modeloCNN3` con capas de aumento (`RandomFlip`, `RandomRotation`) y `Dropout`.
2. Reconstruye un modelo de inferencia despojando los aumentos y fijando `Dropout(training=False)`.
3. Exporta con `modelo_inferencia.export("perros-gatos-cnn-ad3.onnx", format="onnx")`.

Arquitectura (la del ONNX servido): `Input(100,100,1)` → Conv2D(32,3,valid)+MaxPool → Conv2D(64,3)+MaxPool → Conv2D(128,3)+MaxPool → GlobalAveragePooling → Dropout → Dense(1, sigmoide). ~93K parámetros ≈ 400 KB, coherente con los 427 KB del ONNX actual.

El preprocesado debe coincidir con `../app.js`: imagen a **100×100 escala de grises, valores /255**, tensor `[1,100,100,1]`. La salida es `P(perro)`. No hay capa `Rescaling` dentro del modelo: la web normaliza antes de inferir.

## Datos

Opción A — carpeta local con subcarpetas por clase (orden alfabético: `gatos` → 0, `perros` → 1):

```
data/
  gatos/   *.jpg
  perros/  *.jpg
```

Opción B — dataset `cats_vs_dogs` de TF Datasets (se descarga solo): `python train.py --tfds`.

El dataset clásico de Kaggle es [dogs-vs-cats](https://www.kaggle.com/c/dogs-vs-cats/data) (25k imágenes). Otra fuente cómoda: [Microsoft Cats vs Dogs](https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip).

## Local

```bash
pip install -r requirements.txt
python train.py --data data          # o --tfds
cp perros-gatos-cnn-ad3.onnx ../modelo/
```

## Google Colab

```python
!pip install -q -U tf2onnx onnx onnxruntime tensorflow-datasets
# subir train.py o clonar el repo, luego:
!python train.py --tfds --epochs 15
# descargar perros-gatos-cnn-ad3.onnx y colocarlo en modelo/
```

El script guarda `checkpoints/modeloCNN3.keras` (modelo completo con aumentos) antes de exportar, igual que el flujo original donde se partía de un `.keras` ya entrenado.
