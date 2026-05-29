# 📋 Guía del Pipeline del Benchmark de Machine Unlearning

Esta documentación sirve como punto de partida para nuevos usuarios. En ella se detalla cómo estructurar, ejecutar y evaluar las técnicas de desaprendizaje (**Machine Unlearning**) disponibles en el benchmark.

---

## 🚀 Inicio Rápido (Método Recomendado)

La forma más sencilla de ejecutar todo el flujo del benchmark (generar hiperparámetros, entrenar modelos base/naive, aplicar desaprendizaje y evaluar métricas) es utilizando el script automatizador `run_pipeline.py`.

### 1. Preparar el Dataset
Antes de ejecutar el pipeline, debes generar los splits del dataset deseado. Ejecuta uno de los siguientes scripts según el caso:

*   **Para Espiral (Sintético 2D):**
    ```bash
    python 1_create_spiral.py
    python 2_split_dataset.py
    ```
*   **Para CIFAR-Nano (Subconjunto ligero de CIFAR-10, 100 imágenes):**
    ```bash
    python 1_create_cifar_nano.py
    ```
*   **Para CIFAR-10 completo (50,000 imágenes):**
    ```bash
    python 1_create_cifar10.py
    ```

### 2. Lanzar el Pipeline Completo
Una vez creados los splits, puedes ejecutar todo el flujo con un único comando.

*   **Ejemplo con el dataset `spiral` (Modelo MLP simple):**
    ```bash
    python run_pipeline.py --dataset spiral --model_arch models.base_nn.BaseMLP --n_trials 20 --seeds 0,1,2
    ```
*   **Ejemplo con el dataset `cifar_nano` (Modelo ResNet18):**
    ```bash
    python run_pipeline.py --dataset cifar_nano --model_arch models.resnet.ResNet18 --n_trials 5 --seeds 0,1,2
    ```

Este script ejecutará secuencialmente la búsqueda de hiperparámetros con Optuna, el entrenamiento de los modelos de base y de referencia (naive), la aplicación de los algoritmos de unlearning (`cfk`, `cfgk`, `euk`, `rurk`), la evaluación de las métricas (incluyendo RK y RK_micro) y finalmente la generación de la tabla resumen.

---

## 🛠️ Ejecución Paso a Paso (Manual)

Si prefieres ejecutar cada paso manualmente o depurar partes específicas del flujo:

### Paso 1: Búsqueda de Hiperparámetros (`00_hp_search.py`)
Optimiza los hiperparámetros del entrenamiento estándar u optimiza los parámetros específicos de desaprendizaje con **Optuna**.
```bash
# Optimización de hiperparámetros de entrenamiento
python 00_hp_search.py --protocol standard --dataset spiral --model_arch models.base_nn.BaseMLP --n_trials 30

# Optimización de hiperparámetros para desaprendizaje CFK
python 00_hp_search.py --protocol cfk --dataset spiral --model_arch models.base_nn.BaseMLP --n_trials 30
```

### Paso 2: Entrenar Modelos de Referencia (`3_train_model.py`)
Entrena el modelo completo (`base`) y el modelo que simula el olvido ideal entrenado desde cero sin el forget set (`naive`).
```bash
# Entrenar modelo Base (con retain + forget)
python 3_train_model.py --model_name base --dataset spiral --model_arch models.base_nn.BaseMLP --train_splits retain,forget --seeds 0,1,2

# Entrenar modelo Naive (solo con retain)
python 3_train_model.py --model_name naive --dataset spiral --model_arch models.base_nn.BaseMLP --train_splits retain --seeds 0,1,2
```

### Paso 3: Aplicar Protocolos de Desaprendizaje (`3_train_model.py`)
Aplica uno de los métodos de desaprendizaje sobre el modelo `base` previamente entrenado.
```bash
# Aplicar protocolo CFK
python 3_train_model.py --model_name cfk --protocol cfk --dataset spiral --model_arch models.base_nn.BaseMLP --train_splits retain --pretrained_weights "models/weights/base_model_seed_{seed}.pth" --hp_file "models/best_cfk_hp.json" --seeds 0,1,2
```

### Paso 4: Evaluar Métricas (`4_metricas.py`)
Calcula las precisiones absolutas y ratios relativos (`RR`, `RF`, `RT`, `TR`), así como la métrica de conocimiento residual **Residual Knowledge (RK)** y **RK_micro**.
```bash
python 4_metricas.py --unlearned_name cfk --dataset spiral --model_arch models.base_nn.BaseMLP --seeds 0,1,2
```

### Paso 5: Compilar Tablas Comparativas (`5_generate_tables.py`)
Compila todos los JSON de resultados guardados en `results/` en una tabla formateada en Markdown para fácil visualización.
```bash
python 5_generate_tables.py
```

---

## 🎛️ Parámetros de Consola Clave

A continuación se listan los argumentos de consola más importantes de los scripts principales:

### `run_pipeline.py` (Script de automatización global)
*   `--dataset`: Nombre del dataset a usar (`spiral`, `cifar_nano`, `cifar10`).
*   `--model_arch`: Ruta de importación de la arquitectura del modelo (ej: `models.resnet.ResNet18` o `models.base_nn.BaseMLP`).
*   `--n_trials`: Número de iteraciones de Optuna en la búsqueda de hiperparámetros.
*   `--seeds`: Semillas a evaluar (por ejemplo: `0,1,2`).

### `3_train_model.py` (Entrenamiento y Unlearning)
*   `--model_name`: Nombre identificativo de salida del modelo (ej: `base`, `naive`, `cfk`, `rurk`).
*   `--protocol`: Protocolo a aplicar (`standard`, `cfk`, `cfgk`, `euk`, `rurk`).
*   `--train_splits`: Splits de datos a usar para el entrenamiento (ej: `retain` o `retain,forget`).
*   `--pretrained_weights`: Ruta a los pesos iniciales (requerido para métodos de desaprendizaje).
*   `--hp_file`: Archivo JSON con los mejores hiperparámetros cargados para el entrenamiento.

### `4_metricas.py` (Cálculo de Métricas y RK)
*   `--unlearned_name`: Nombre del modelo desentrenado que se va a evaluar (ej: `cfk`, `cfgk`, `euk`, `rurk`).
*   `--rk_tau`: Desviación estándar de las perturbaciones gaussianas (por defecto `0.03`).
*   `--rk_c`: Número de perturbaciones Monte Carlo por muestra para evaluar RK (por defecto `100`).
*   `--rk_chunk_size`: Tamaño del bloque para evaluación vectorizada en GPU (por defecto `100`).

---

## 🧪 Pruebas Unitarias

El proyecto incluye tests unitarios automatizados ejecutables mediante **pytest** para asegurar el correcto funcionamiento de las métricas y los algoritmos:
```bash
pytest tests/ -v
```
