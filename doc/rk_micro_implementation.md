# Implementación de la métrica RK_micro para Residual Knowledge

## 1. Objetivo

Añadir al pipeline una métrica de **Residual Knowledge agregada por conteos globales**, denominada `RK_micro`, para evaluar si un modelo *unlearned* conserva conocimiento residual sobre el *forget set* `S_f` bajo perturbaciones gaussianas.

La métrica compara, sobre perturbaciones locales de las muestras olvidadas, cuántas veces el modelo *unlearned* `m` predice correctamente la etiqueta real frente a cuántas veces lo hace el modelo de referencia *re-trained* `a`, entrenado desde cero únicamente sobre el *retain set* `S_r`.

La diferencia clave respecto a la métrica macro definida en el artículo es que **no se calcula un ratio por muestra**, sino que primero se agregan todos los conteos y después se divide. Esto evita que muestras individuales con denominador cero propaguen valores `NaN`.

---

## 2. Definición matemática

Dado un *forget set*:

```text
S_f = {(x_i, y_i)}_{i=1}^{n_f}
```

para cada muestra `x_i` se generan `c` perturbaciones gaussianas:

```text
x'_{i,j}, con j = 1, ..., c
```

donde:

```text
m = modelo unlearned
a = modelo re-trained, entrenado desde cero únicamente sobre S_r
y_i = etiqueta real de la muestra olvidada
```

Definimos los conteos globales:

```text
M_total = sum_i sum_j 1( m(x'_{i,j}) = y_i )

A_total = sum_i sum_j 1( a(x'_{i,j}) = y_i )
```

Entonces:

```text
RK_micro_tau(S_f) = M_total / A_total
```

donde `tau` controla la magnitud de las perturbaciones gaussianas.

### Interpretación

```text
RK_micro > 1
```

El modelo *unlearned* reconoce perturbaciones del *forget set* más a menudo que el modelo *re-trained*. Esto sugiere presencia de *residual knowledge*.

```text
RK_micro ≈ 1
```

El comportamiento del modelo *unlearned* es similar al modelo *re-trained*.

```text
RK_micro < 1
```

El modelo *unlearned* reconoce menos perturbaciones que el modelo *re-trained*. Según la interpretación principal del artículo, este caso no es evidencia de *residual knowledge*.

Para usarlo como objetivo de optimización, se recomienda penalizar solo el exceso:

```text
RK_micro_excess = max(0, RK_micro - 1)
```

o, de forma más estable:

```text
RK_micro_log_excess = max(0, log(RK_micro))
```

---

## 3. Decisión metodológica

La métrica debe implementarse como **ratio de conteos agregados**, no como media de ratios por muestra.

### Correcto

```python
rk_micro = total_correct_unlearned / total_correct_retrained
```

### Incorrecto

```python
rk_micro = mean(correct_unlearned_per_sample / correct_retrained_per_sample)
```

La segunda opción reproduce el problema original de denominadores cero por muestra.

---

## 4. Inputs necesarios

La función debe recibir como mínimo:

```python
unlearned_model
retrained_model
forget_loader
num_perturbations
tau
device
```

Opcionalmente:

```python
noise_type = "gaussian"
clip_min = 0.0
clip_max = 1.0
batch_size_perturbations = None
return_details = True
seed = None
normalize_fn = None
```

---

## 5. Requisitos de los modelos

Ambos modelos deben cumplir:

- estar en modo `eval()`;
- estar en el mismo `device`;
- recibir inputs con el mismo preprocesamiento;
- devolver logits o scores de clase con forma `[batch_size, num_classes]`.

La predicción se calcula como:

```python
pred = logits.argmax(dim=1)
```

---

## 6. Requisitos del DataLoader

El `forget_loader` debe devolver batches:

```python
x, y
```

donde:

```text
x tiene forma [B, C, H, W]
y tiene forma [B]
```

En el caso actual del pipeline, las imágenes se asumen en rango `[0, 1]`.

Si en alguna parte del pipeline se normaliza CIFAR-10 con media y desviación típica, debe decidirse explícitamente dónde se añade el ruido.

La opción recomendada para este pipeline es:

1. El `DataLoader` devuelve imágenes en `[0, 1]`.
2. Se genera ruido gaussiano en espacio de píxel.
3. Se aplica clipping a `[0, 1]`.
4. Si el modelo espera inputs normalizados, se aplica `normalize_fn` justo antes del forward.

---

## 7. Generación de perturbaciones

Para cada imagen `x_i`, generar `c` perturbaciones:

```python
noise = torch.randn_like(x) * tau
x_perturbed = x + noise
x_perturbed = torch.clamp(x_perturbed, 0.0, 1.0)
```

Para esta implementación, la perturbación queda definida como:

```text
Gaussian perturbation + clipping to valid image range.
```

No debe mezclarse esta métrica con otras familias de perturbaciones en el mismo resultado experimental.

---

## 8. Output recomendado

La función no debería devolver únicamente un `float`. Debe devolver un diccionario con la métrica y diagnósticos.

Formato recomendado:

```python
{
    "rk_micro": float,
    "rk_micro_smoothed": float,
    "rk_micro_for_objective": float,
    "rk_micro_excess": float,
    "rk_micro_log_excess": float,
    "total_correct_unlearned": int,
    "total_correct_retrained": int,
    "total_perturbations": int,
    "num_forget_samples": int,
    "num_perturbations_per_sample": int,
    "tau": float,
    "zero_global_denominator": bool,
    "unlearned_perturbed_accuracy": float,
    "retrained_perturbed_accuracy": float
}
```

Donde:

```python
unlearned_perturbed_accuracy = M_total / total_perturbations

retrained_perturbed_accuracy = A_total / total_perturbations
```

Esto es importante porque `RK_micro` por sí solo puede ocultar si ambos modelos tienen accuracies muy bajas bajo perturbación.

---

## 9. Manejo del caso A_total = 0

Aunque `RK_micro` reduce el problema de denominadores cero por muestra, todavía existe un caso extremo:

```python
A_total == 0
```

Es decir, el modelo *re-trained* no acierta ninguna perturbación de ninguna muestra del *forget set*.

Este caso debe tratarse explícitamente.

### Política recomendada

Calcular dos valores:

```python
rk_micro
rk_micro_for_objective
```

Con la siguiente lógica:

```python
if A_total > 0:
    rk_micro = M_total / A_total
else:
    if M_total > 0:
        rk_micro = float("inf")
    else:
        rk_micro = float("nan")
```

Para rankings automáticos u Optuna conviene evitar `inf` y `nan`. Por eso se calcula también una versión suavizada:

```python
rk_micro_smoothed = (M_total + 0.5) / (A_total + 0.5)
```

Entonces:

```python
rk_micro_for_objective = rk_micro if is_finite(rk_micro) else rk_micro_smoothed
```

Decisión recomendada:

- Para reporting científico: reportar `rk_micro` estricto y marcar si `A_total == 0`.
- Para optimización automática: usar `rk_micro_for_objective`, finito y estable.
- En condiciones normales, `A_total > 0`, por tanto ambas versiones coinciden salvo por la métrica auxiliar suavizada.

---

## 10. Pseudocódigo

```text
def compute_rk_micro(
    unlearned_model,
    retrained_model,
    forget_loader,
    num_perturbations,
    tau,
    device,
    normalize_fn=None,
    clip_min=0.0,
    clip_max=1.0,
    seed=None,
):
    set both models to eval mode

    if seed is not None:
        set torch generator seed

    total_correct_unlearned = 0
    total_correct_retrained = 0
    total_perturbations = 0
    num_forget_samples = 0

    with torch.no_grad():
        for x, y in forget_loader:

            move x, y to device
            batch_size = x.shape[0]
            num_forget_samples += batch_size

            for j in range(num_perturbations):

                generate gaussian noise with std tau
                x_perturbed = x + noise
                clip x_perturbed to [clip_min, clip_max]

                if normalize_fn is not None:
                    x_model = normalize_fn(x_perturbed)
                else:
                    x_model = x_perturbed

                logits_m = unlearned_model(x_model)
                logits_a = retrained_model(x_model)

                pred_m = argmax(logits_m)
                pred_a = argmax(logits_a)

                total_correct_unlearned += number of pred_m == y
                total_correct_retrained += number of pred_a == y
                total_perturbations += batch_size

    if total_correct_retrained > 0:
        rk_micro = total_correct_unlearned / total_correct_retrained
    else:
        if total_correct_unlearned > 0:
            rk_micro = inf
        else:
            rk_micro = nan

    rk_micro_smoothed = (
        (total_correct_unlearned + 0.5)
        / (total_correct_retrained + 0.5)
    )

    rk_micro_for_objective = (
        rk_micro if is_finite(rk_micro) else rk_micro_smoothed
    )

    rk_micro_excess = max(0, rk_micro_for_objective - 1)
    rk_micro_log_excess = max(0, log(rk_micro_for_objective))

    return dictionary with metrics and diagnostics
```

---

## 11. Implementación PyTorch recomendada

```python
import math
import torch


@torch.no_grad()
def compute_rk_micro(
    unlearned_model,
    retrained_model,
    forget_loader,
    num_perturbations: int = 100,
    tau: float = 0.1,
    device: str | torch.device = "cuda",
    normalize_fn=None,
    clip_min: float = 0.0,
    clip_max: float = 1.0,
    seed: int | None = None,
) -> dict:
    """
    Compute RK_micro over the forget set using Gaussian perturbations.

    RK_micro = total_correct_unlearned / total_correct_retrained

    The metric aggregates counts over all forget samples and all perturbations
    before taking the ratio. This avoids per-sample zero denominators.

    Parameters
    ----------
    unlearned_model:
        Model after unlearning.

    retrained_model:
        Reference model trained from scratch only on the retain set.

    forget_loader:
        DataLoader over the forget set. It must return (x, y).

    num_perturbations:
        Number of Gaussian perturbations per forget sample.

    tau:
        Standard deviation of the Gaussian perturbation.

    device:
        Device used for model evaluation.

    normalize_fn:
        Optional preprocessing function applied after perturbation and clipping.
        Use this if the model expects normalized inputs.

    clip_min, clip_max:
        Valid input range after perturbation.

    seed:
        Optional random seed for reproducible perturbations.

    Returns
    -------
    dict
        Dictionary with RK_micro and diagnostic quantities.
    """

    if num_perturbations <= 0:
        raise ValueError("num_perturbations must be positive.")

    if tau < 0:
        raise ValueError("tau must be non-negative.")

    device = torch.device(device)

    unlearned_model = unlearned_model.to(device)
    retrained_model = retrained_model.to(device)

    unlearned_model.eval()
    retrained_model.eval()

    generator = None
    if seed is not None:
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)

    total_correct_unlearned = 0
    total_correct_retrained = 0
    total_perturbations = 0
    num_forget_samples = 0

    for x, y in forget_loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        batch_size = x.shape[0]
        num_forget_samples += batch_size

        for _ in range(num_perturbations):
            if generator is not None:
                noise = torch.randn(
                    x.shape,
                    device=device,
                    dtype=x.dtype,
                    generator=generator,
                )
            else:
                noise = torch.randn_like(x)

            x_perturbed = x + tau * noise
            x_perturbed = torch.clamp(x_perturbed, clip_min, clip_max)

            if normalize_fn is not None:
                x_model = normalize_fn(x_perturbed)
            else:
                x_model = x_perturbed

            logits_unlearned = unlearned_model(x_model)
            logits_retrained = retrained_model(x_model)

            pred_unlearned = logits_unlearned.argmax(dim=1)
            pred_retrained = logits_retrained.argmax(dim=1)

            total_correct_unlearned += (pred_unlearned == y).sum().item()
            total_correct_retrained += (pred_retrained == y).sum().item()
            total_perturbations += batch_size

    if total_perturbations == 0:
        raise ValueError("forget_loader produced zero samples.")

    if total_correct_retrained > 0:
        rk_micro = total_correct_unlearned / total_correct_retrained
        zero_global_denominator = False
    else:
        zero_global_denominator = True
        if total_correct_unlearned > 0:
            rk_micro = float("inf")
        else:
            rk_micro = float("nan")

    rk_micro_smoothed = (
        (total_correct_unlearned + 0.5)
        / (total_correct_retrained + 0.5)
    )

    if math.isfinite(rk_micro):
        rk_micro_for_objective = rk_micro
    else:
        rk_micro_for_objective = rk_micro_smoothed

    rk_micro_excess = max(0.0, rk_micro_for_objective - 1.0)
    rk_micro_log_excess = max(0.0, math.log(rk_micro_for_objective))

    unlearned_perturbed_accuracy = (
        total_correct_unlearned / total_perturbations
    )
    retrained_perturbed_accuracy = (
        total_correct_retrained / total_perturbations
    )

    return {
        "rk_micro": rk_micro,
        "rk_micro_smoothed": rk_micro_smoothed,
        "rk_micro_for_objective": rk_micro_for_objective,
        "rk_micro_excess": rk_micro_excess,
        "rk_micro_log_excess": rk_micro_log_excess,
        "total_correct_unlearned": int(total_correct_unlearned),
        "total_correct_retrained": int(total_correct_retrained),
        "total_perturbations": int(total_perturbations),
        "num_forget_samples": int(num_forget_samples),
        "num_perturbations_per_sample": int(num_perturbations),
        "tau": float(tau),
        "zero_global_denominator": bool(zero_global_denominator),
        "unlearned_perturbed_accuracy": float(unlearned_perturbed_accuracy),
        "retrained_perturbed_accuracy": float(retrained_perturbed_accuracy),
    }
```

---

## 12. Integración en el pipeline

### Nombre interno recomendado

Usar nombres explícitos para evitar confusión con la métrica macro original del artículo:

```text
rk_micro
rk_micro_excess
rk_micro_log_excess
rk_micro_smoothed
```

Si se añade como objetivo de Optuna:

```text
objective10_rk_micro
```

o, si se quiere extender el objetivo existente:

```text
objective10_plus_rk_micro
```

Evitar llamarlo simplemente `rk`, porque puede confundirse con la definición macro original.

---

## 13. Dónde calcularlo

La métrica debe calcularse después de disponer de:

1. modelo original, si se usa para otras métricas;
2. modelo *unlearned*;
3. modelo *naive* o *retrained* sobre retain set;
4. `forget_loader`;
5. configuración de perturbaciones.

Conceptualmente encaja en la fase donde ya se calculan:

- retain accuracy;
- forget accuracy;
- validation accuracy;
- test accuracy;
- MIA / indiscernibility;
- residual metrics.

No se recomienda calcularla durante entrenamiento salvo que se quiera usar como métrica online, porque es computacionalmente cara.

---

## 14. Configuración Hydra sugerida

```yaml
metrics:
  residual_knowledge:
    enabled: true
    variant: rk_micro
    num_perturbations: 100
    tau: 0.1
    noise_type: gaussian
    clip_min: 0.0
    clip_max: 1.0
    seed: 123
    use_same_perturbations_for_models: true
    objective_value: rk_micro_log_excess
```

Campo especialmente importante:

```yaml
use_same_perturbations_for_models: true
```

Esto debe cumplirse siempre. Para cada perturbación `x'_{i,j}`, ambos modelos `m` y `a` deben evaluarse sobre exactamente el mismo tensor perturbado. Si no, se introduce ruido Monte Carlo adicional en la comparación.

---

## 15. Uso como objetivo de Optuna

Para optimización multiobjetivo, no se recomienda usar directamente `rk_micro`, porque valores menores que 1 no deberían ser premiados indefinidamente.

Objetivo minimizable recomendado:

```python
objective_rk = rk_micro_log_excess
```

donde:

```python
rk_micro_log_excess = max(0, log(rk_micro_for_objective))
```

Interpretación:

```text
si RK_micro <= 1, penalización 0
si RK_micro > 1, penalización creciente
```

La escala logarítmica evita que casos extremos dominen completamente el objetivo.

Alternativa más simple:

```python
objective_rk = rk_micro_excess
```

donde:

```python
rk_micro_excess = max(0, rk_micro_for_objective - 1)
```

Esta versión puede ser menos estable si aparecen ratios grandes.

Recomendación para el pipeline:

```text
Usar rk_micro_log_excess como objetivo Optuna.
Reportar rk_micro como métrica científica.
Guardar también rk_micro_smoothed y los conteos.
```

---

## 16. Logging mínimo obligatorio

Cada run debe guardar como mínimo:

```json
{
  "rk_micro": 1.37,
  "rk_micro_smoothed": 1.37,
  "rk_micro_for_objective": 1.37,
  "rk_micro_excess": 0.37,
  "rk_micro_log_excess": 0.3147,
  "total_correct_unlearned": 8231,
  "total_correct_retrained": 6008,
  "total_perturbations": 100000,
  "num_forget_samples": 1000,
  "num_perturbations_per_sample": 100,
  "tau": 0.1,
  "zero_global_denominator": false,
  "unlearned_perturbed_accuracy": 0.08231,
  "retrained_perturbed_accuracy": 0.06008
}
```

Guardar los conteos es importante porque permite reproducir o auditar la métrica sin volver a evaluar modelos.

---

## 17. Tests unitarios recomendados

### Test 1 — Caso básico

Si:

```python
M_total = 80
A_total = 40
```

entonces:

```python
rk_micro = 2.0
rk_micro_excess = 1.0
rk_micro_log_excess = log(2.0)
```

---

### Test 2 — No penaliza si RK <= 1

Si:

```python
M_total = 30
A_total = 60
```

entonces:

```python
rk_micro = 0.5
rk_micro_excess = 0.0
rk_micro_log_excess = 0.0
```

---

### Test 3 — Denominador global cero con numerador positivo

Si:

```python
M_total = 10
A_total = 0
```

entonces:

```python
rk_micro = inf
rk_micro_smoothed = 21.0
rk_micro_for_objective = 21.0
zero_global_denominator = True
```

Porque:

```python
(10 + 0.5) / (0 + 0.5) = 21
```

---

### Test 4 — Denominador global cero con numerador cero

Si:

```python
M_total = 0
A_total = 0
```

entonces:

```python
rk_micro = nan
rk_micro_smoothed = 1.0
rk_micro_for_objective = 1.0
zero_global_denominator = True
```

Este caso debe marcarse como experimentalmente degenerado, porque ninguno de los dos modelos reconoce ninguna perturbación.

---

### Test 5 — Número total de perturbaciones

Si el *forget set* tiene `n_f = 1000` muestras y `c = 100`, entonces:

```python
total_perturbations = 100000
```

Este test detecta errores de acumulación por batch.

---

## 18. Recomendación final de implementación

Para añadir `RK_micro` de forma limpia al pipeline, implementar una función independiente:

```python
compute_rk_micro(...)
```

y hacer que el pipeline la llame solo si:

```yaml
metrics.residual_knowledge.enabled: true
metrics.residual_knowledge.variant: rk_micro
```

La métrica principal para tablas debe ser:

```text
rk_micro
```

La métrica para Optuna debe ser:

```text
rk_micro_log_excess
```

Y siempre deben guardarse:

```text
total_correct_unlearned
total_correct_retrained
total_perturbations
unlearned_perturbed_accuracy
retrained_perturbed_accuracy
zero_global_denominator
```

Con esta implementación se evita la propagación de `NaN`, se mantiene una interpretación estadística clara y no se ocultan casos relevantes de *residual knowledge*.
