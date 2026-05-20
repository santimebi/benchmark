# 📋 Documentación Funcional — Pipeline del Benchmark

## Visión General

Este benchmark implementa un pipeline completo para evaluar técnicas de
**machine unlearning** sobre un dataset sintético de espirales entrelazadas.
El objetivo es entrenar un modelo base, definir un subconjunto de datos a
"olvidar" (forget set), y posteriormente aplicar y evaluar métodos de
unlearning.

---

## Arquitectura del Proyecto

```
benchmark/
├── doc/
│   └── pipeline.md              ← Este fichero
├── models/
│   ├── base_nn.py               ← Definición del modelo BaseMLP
│   ├── weights/                 ← Pesos entrenados (.pth)
│   └── best_hp.json             ← Mejores HP (generado por Optuna)
├── tests/
│   ├── conftest.py              ← Configuración compartida de pytest
│   ├── test_base_nn.py          ← Tests del modelo
│   ├── test_create_spiral.py    ← Tests de generación del dataset
│   ├── test_split_dataset.py    ← Tests de partición
│   ├── test_hp_search_optuna.py ← Tests de búsqueda de HP
│   └── test_train_modelo_base.py← Tests del entrenamiento
├── utils/
│   └── config.py                ← Configuración centralizada (rutas)
├── datasets/                    ← Datos generados (CSV, NPZ)
├── 1_create_spiral.py           ← Paso 1: Generar dataset
├── 2_split_dataset.py           ← Paso 2: Particionar dataset
├── 00_hp_search_optuna.py       ← Paso 2.5: Buscar hiperparámetros
├── 3_train_model.py             ← Paso 3: Entrenar modelo (base, naive o personalizado)
└── requirements.txt             ← Dependencias del proyecto
```

---

## Pipeline de Ejecución

El pipeline se ejecuta en orden secuencial. Cada paso genera artefactos
que son consumidos por los pasos posteriores.

```mermaid
flowchart LR
    A["1_create_spiral.py"] --> B["2_split_dataset.py"]
    B --> C["00_hp_search_optuna.py"]
    C --> D["3_train_model.py"]
    
    A -. "spiral.csv" .-> B
    B -. "spiral_splits_seed_N.npz" .-> C
    C -. "best_hp.json" .-> D
    B -. "spiral_splits_seed_N.npz" .-> D
```

---

## Paso 1 — Generación del Dataset (`1_create_spiral.py`)

### Propósito
Genera un dataset sintético de **espirales entrelazadas** en 2D con N clases.

### Parámetros principales
| Parámetro | Valor por defecto | Descripción |
|-----------|-------------------|-------------|
| `n_points_per_class` | 400 | Puntos por espiral |
| `n_classes` | 3 | Número de espirales/clases |
| `noise` | 0.45 | Ruido angular (desviación estándar) |
| `rotations` | 2.5 | Vueltas completas de cada espiral |
| `random_state` | 42 | Semilla para reproducibilidad |

### Salida
- `datasets/spiral.csv` — CSV con columnas `x1, x2, label`

### Ejecución
```bash
python 1_create_spiral.py
```

---

## Paso 2 — Partición del Dataset (`2_split_dataset.py`)

### Propósito
Divide el dataset en **cuatro subconjuntos** para el benchmark de unlearning:

| Split | Descripción | Proporción aprox. |
|-------|-------------|-------------------|
| **Retain** | Datos que el modelo debe mantener | ~57% |
| **Forget** | Datos a olvidar (solo clase 0, cercanos al centro) | ~14% |
| **Validation** | Evaluación durante entrenamiento | ~8% |
| **Test** | Evaluación final | ~20% |

### Estrategia de Forget
El forget set se construye seleccionando las muestras de la **clase 0**
más cercanas al **centro de la espiral** (menor radio). Esto simula un
escenario realista donde se quiere eliminar un subgrupo coherente y
espacialmente localizado de los datos de entrenamiento.

### Salida
- `datasets/spiral_splits_seed_{N}.npz` — Archivo NumPy comprimido con
  las 8 arrays: `X_retain, y_retain, X_forget, y_forget, X_val, y_val, X_test, y_test`

### Ejecución
```bash
python 2_split_dataset.py
```

---

## Paso 2.5 — Búsqueda de Hiperparámetros (`00_hp_search_optuna.py`)

### Propósito
Encuentra la mejor combinación de hiperparámetros para el modelo base
usando **Optuna** con pruning automático (MedianPruner).

### Espacio de búsqueda
| Hiperparámetro | Tipo | Rango |
|----------------|------|-------|
| `hidden_dim` | Categórico | [8, 16, 32, 64, 128] |
| `lr` | Log-uniforme | [1e-4, 1e-1] |
| `batch_size` | Categórico | [16, 32, 64, 128] |
| `epochs` | Entero (paso 50) | [50, 300] |

### Métrica optimizada
- **Validation loss** (CrossEntropyLoss) — se minimiza.

### Pruning
Se usa `MedianPruner` con:
- `n_startup_trials=5`: los primeros 5 trials no se podan.
- `n_warmup_steps=20`: no se poda antes del epoch 20.

### Salida
- `models/best_hp.json` — JSON con los mejores HP encontrados.

### Ejecución
```bash
python 00_hp_search_optuna.py
```

---

## Paso 3 — Entrenamiento del Modelo (`3_train_model.py`)

### Propósito
Entrena cualquier modelo configurando dinámicamente su arquitectura, protocolo de entrenamiento/olvido, conjunto de splits y parámetros de hiperparámetros. 

Por defecto se entrena el modelo base original (`base_model_seed_{N}.pth`) con todos los datos (`retain` y `forget`), pero también permite entrenar el modelo ingenuo (`naive_model_seed_{N}.pth`) que excluye el split de `forget`.

### Parámetros principales (CLI)
| Parámetro | Valor por defecto | Descripción |
|-----------|-------------------|-------------|
| `--model_arch` | `models.base_nn.BaseMLP` | Import path de la clase del modelo a instanciar. |
| `--protocol` | `standard` | Protocolo de entrenamiento o unlearning a ejecutar (ej. `standard`). |
| `--train_splits` | `retain,forget` | Lista de splits separados por comas a incluir en el entrenamiento (ej. `retain` para naive). |
| `--model_name` | `base` | Prefijo para nombrar el archivo de pesos resultante en `models/weights/`. |
| `--hp_file` | `models/best_hp.json` | Ruta al archivo JSON de hiperparámetros generado por Optuna. |
| `--epochs` | *(Desde `hp_file` o `150`)* | Sobrescribe el número de épocas de entrenamiento. |
| `--batch_size` | *(Desde `hp_file` o `32`)* | Sobrescribe el tamaño de batch. |
| `--lr` | *(Desde `hp_file` o `1e-3`)* | Sobrescribe el learning rate. |
| `--hidden_dim` | *(Desde `hp_file` o `16`)* | Sobrescribe la dimensión oculta del modelo MLP. |
| `--seeds` | `0,1,2` | Semillas para las cuales entrenar el modelo, separadas por comas (ej. `0,1,2`). |

### Carga de Hiperparámetros
1. Si existe la ruta del `--hp_file` (por defecto `models/best_hp.json`) → se cargan sus valores.
2. Si no existe → se usan valores por defecto:
   - `hidden_dim=16`, `lr=1e-3`, `batch_size=32`, `epochs=150`
3. Cualquier argumento específico `--epochs`, `--batch_size`, `--lr` o `--hidden_dim` indicado por consola sobrescribirá los valores anteriores.

### Salidas
- `models/weights/{model_name}_model_seed_{N}.pth` — State dict del modelo entrenado.

### Ejecución
Para entrenar el modelo original completo (modelo base):
```bash
python 3_train_model.py --model_name base --train_splits retain,forget
```

Para entrenar el modelo que nunca vio los datos a olvidar (modelo naive):
```bash
python 3_train_model.py --model_name naive --train_splits retain
```

---

## Modelo Base — `BaseMLP` (`models/base_nn.py`)

### Arquitectura

```
┌─────────────────────────────────────────┐
│            feature_extractor            │
│  Linear(input_dim, hidden_dim) → ReLU   │
│  Linear(hidden_dim, hidden_dim) → ReLU  │
├─────────────────────────────────────────┤
│              classifier                 │
│  Linear(hidden_dim, output_dim)         │
└─────────────────────────────────────────┘
```

### Parámetros por defecto
- `input_dim=2` (x1, x2 del espiral)
- `hidden_dim=16`
- `output_dim=2`

### Feature extraction
```python
logits, features = model(x, return_features=True)
# features.shape = (batch_size, hidden_dim)
```

---

## Configuración (`utils/config.py`)

Define la ruta centralizada para el almacenamiento de datos:

```python
DATASETS_PATH = Path("/datasets")
```

> **Nota**: Todos los scripts importan esta constante para garantizar
> consistencia en las rutas de lectura/escritura.

---

## Tests

El proyecto incluye tests unitarios con **pytest** para cada componente:

| Fichero | Componente testeado | Nº tests |
|---------|---------------------|----------|
| `test_base_nn.py` | Arquitectura y forward pass del modelo | 10 |
| `test_create_spiral.py` | Generación del dataset espiral | 9 |
| `test_split_dataset.py` | Partición y forget set | 11 |
| `test_hp_search_optuna.py` | Búsqueda de hiperparámetros | 13 |
| `test_train_modelo_base.py` | Entrenamiento y checkpointing | 10 |

### Ejecución
```bash
pytest tests/ -v
```

### Estrategia de testing
- **Datos sintéticos**: todos los tests usan datos generados en `tmp_path`
  de pytest, sin depender de ficheros reales.
- **Carga dinámica**: los módulos con nombres numéricos (`1_`, `2_`, `3_`,
  `00_`) se cargan con `importlib` para evitar errores de sintaxis en imports.
- **Mocks**: las funciones de matplotlib se mockean para evitar ventanas
  gráficas durante los tests.

---

## Dependencias

Ver `requirements.txt`:

| Paquete | Versión | Uso |
|---------|---------|-----|
| numpy | 1.26.4 | Manipulación numérica, datasets |
| matplotlib | 3.9.2 | Visualización de espirales y splits |
| scikit-learn | 1.8.0 | `train_test_split` para particiones |
| torch | 2.6.0 | Modelo, entrenamiento, tensores |
| torchaudio | 2.6.0 | Dependencia del ecosistema PyTorch |
| torchvision | 0.21.0 | Dependencia del ecosistema PyTorch |
| optuna | 4.8.0 | Búsqueda de hiperparámetros |
| pytest | 9.0.3 | Framework de testing |

### Instalación
```bash
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu118
```

---

## Flujo Completo — Quick Start

```bash
# 1. Generar el dataset espiral
python 1_create_spiral.py

# 2. Crear los splits (retain/forget/val/test)
python 2_split_dataset.py

# 3. Buscar los mejores hiperparámetros (opcional pero recomendado)
python 00_hp_search_optuna.py

# 4. Entrenar el modelo base y el modelo naive con los HP óptimos
python 3_train_model.py --model_name base --train_splits retain,forget
python 3_train_model.py --model_name naive --train_splits retain

# 5. Ejecutar tests
pytest tests/ -v
```
