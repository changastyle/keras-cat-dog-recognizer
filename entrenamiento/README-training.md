# Cómo se entrenó el modelo perros-gatos

Este documento explica el pipeline completo: del dataset en Keras al `.onnx` que sirve la web.

## Pipeline original (Google Colab)

El modelo que hoy sirve la web (`modelo/perros-gatos-cnn-ad3.onnx`, ~427 KB) se generó así:

1. **Entrenamiento en Keras** — se entrenó una CNN y se guardó como `modeloCNN3.keras`. El modelo de entrenamiento incluía capas de aumento (`RandomFlip`, `RandomRotation`) y `Dropout`, que solo deben actuar durante el training.
2. **Reconstrucción para inferencia** — se recorrieron las capas del modelo cargado copiándolas sobre una nueva entrada `(100, 100, 1)`:
   - `RandomFlip` y `RandomRotation` se **descartan** (aumento ≠ inferencia).
   - `Dropout` se llama con `training=False` (queda en paso directo).
   - El resultado se verificó comparando la predicción del modelo original y del nuevo sobre el mismo tensor aleatorio.
3. **Exportación ONNX** — `modelo_inferencia.export("perros-gatos-cnn-ad3.onnx", format="onnx")` (API de Keras 3; usa tf2onnx por debajo).
4. **Despliegue** — el `.onnx` se copió a `modelo/` y la web lo ejecuta en el navegador con ONNX Runtime Web (WebAssembly), sin backend.

## Arquitectura

El ONNX servido (inspeccionado) contiene: 3 bloques `Conv2D + ReLU + MaxPool` (32/64/128 filtros, kernel 3×3, padding `valid`: 100→98→49→47→23→21→10) → `GlobalAveragePooling` → `Dense(1, sigmoide)` ≈ 93K parámetros.

| Capa | Salida |
|---|---|
| Input | (None, 100, 100, 1) |
| conv2d + pool ×3 | …→ (None, 10, 10, 128) |
| GlobalAveragePooling | (None, 128) |
| Dense sigmoide | (None, 1) = **P(perro)** |

## Contrato con la web

`../app.js` preprocesa cada fotograma de la webcam así:

- resize a **100×100**
- escala de grises con pesos de `cv2.COLOR_RGB2GRAY` (0.299/0.587/0.114)
- normalización **/255** → tensor `[1,100,100,1]` float32
- interpreta la salida como `P(perro)`; `1 - p` es `P(gato)`

Consecuencias para el entrenamiento:

- El modelo **no lleva `Rescaling`** — la web ya normaliza; entrenar con píxeles /255.
- Clases alfabéticas: `gatos` → 0, `perros` → 1.
- La web acepta `[1,100,100,1]` (NHWC) o `[1,1,100,100]` (NCHW).

## Cómo re-entrenar

```bash
pip install -r requirements.txt
python train.py --tfds --epochs 15        # descarga cats_vs_dogs
# o con dataset propio:
python train.py --data data               # data/{gatos,perros}/*.jpg
```

Cada ejecución exporta a **`../modelo/perros-gatos-cnn-ad3-YYYYMMDD.onnx`** (fecha del día) y actualiza `../modelo/perros-gatos-cnn-ad3.onnx`, el archivo que carga la web. Para exportar solo el artefacto fechado sin tocar el desplegado: `--no-deploy`.

También guarda `checkpoints/modeloCNN3.keras` (modelo completo con aumentos), por si quieres seguir el flujo original: cargar → limpiar → exportar.

En Colab:

```python
!pip install -q -U tf2onnx onnx onnxruntime tensorflow-datasets
!python train.py --tfds --epochs 15
# descargar el .onnx fechado y colocarlo en modelo/
```

## Datos

- `cats_vs_dogs` de TF Datasets (`--tfds`): el más cómodo, ~23k imágenes.
- Kaggle [dogs-vs-cats](https://www.kaggle.com/c/dogs-vs-cats/data): 25k imágenes, descomprimir en `data/gatos` y `data/perros`.
- Dataset propio: misma estructura de carpetas.

## Nota sobre la variante Flatten+Dense

El `summary` del notebook mostraba en una iteración `Flatten → Dense(128)` (~1.7M parámetros → ~6.5 MB). El ONNX desplegado usa `GlobalAveragePooling` (mucho más ligero para el navegador). `train.py` reproduce la variante GAP; si quieres la otra, sustituye `GlobalAveragePooling2D()` por `Flatten()` y añade `Dense(128, activation="relu")` antes del `Dropout`.
