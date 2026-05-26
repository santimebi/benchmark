# Especificación de implementación: Residual Knowledge (RK)

## Alcance

Este documento especifica únicamente la métrica **Residual Knowledge (RK)** tal como aparece definida en el paper *The Unseen Threat: Residual Knowledge in Machine Unlearning under Perturbed Samples*.

No se incluyen métricas derivadas no definidas explícitamente en el paper, como `rk_smooth_macro`, `rk_excess_macro`, `rk_log_excess_macro`, objetivos de Optuna basados en exceso, suavizados con `alpha`, ni penalizaciones adicionales. La implementación debe calcular exclusivamente:

- RK por muestra olvidada.
- RK agregado sobre el forget set mediante promedio de RK por muestra.

Cualquier tratamiento adicional, ranking, transformación, smoothing o métrica auxiliar debe quedar fuera de esta implementación.

## Notación usada por el paper

Sea:

- `S`: dataset original de entrenamiento.
- `S_f`: forget set, subconjunto de muestras que deben olvidarse.
- `S_r = S \ S_f`: retain set.
- `(x, y) in S_f`: muestra olvidada y su etiqueta real.
- `m`: modelo unlearned obtenido mediante un método de machine unlearning.
- `a`: modelo re-trained, entrenado desde cero sobre `S_r`.
- `B_p(x, tau)`: bola de perturbaciones alrededor de `x`, con norma `p` y radio `tau`.
- `x'`: muestra perturbada generada de forma independiente dentro de `B_p(x, tau)`.
- `1(condition)`: función indicadora, igual a 1 si la condición se cumple y 0 en caso contrario.

## Definición de RK por muestra

Para cada muestra olvidada `(x, y) in S_f`, el paper define Residual Knowledge como el cociente:

```text
r_tau((x, y)) = Pr[m(x') = y] / Pr[a(x') = y],
where x' i.i.d. ~ B_p(x, tau).
```

Es decir, RK compara la probabilidad de que el modelo unlearned clasifique correctamente perturbaciones de una muestra olvidada frente a la probabilidad de que el modelo re-trained clasifique correctamente esas mismas perturbaciones.

## Definición de RK sobre el forget set

El paper extiende la definición anterior al conjunto completo `S_f` promediando el RK por muestra:

```text
r_tau(S_f) = (1 / |S_f|) * sum_{(x, y) in S_f} r_tau((x, y)).
```

Esta es la métrica agregada que debe reportarse para un modelo, método, seed, dataset, split y valor de `tau`.

## Estimador Monte Carlo definido en el paper

Como las probabilidades anteriores no se conocen analíticamente, el paper indica que `r_tau((x, y))` puede estimarse mediante Monte Carlo.

Para cada muestra olvidada `(x, y)`, se generan `c` perturbaciones independientes:

```text
x'_1, x'_2, ..., x'_c ~ B_p(x, tau)
```

El estimador empírico por muestra es:

```text
r_hat_tau((x, y)) =
    sum_{j=1}^c 1[m(x'_j) = y]
    /
    sum_{j=1}^c 1[a(x'_j) = y]
```

De forma equivalente, si se definen:

```text
u_i = número de perturbaciones de la muestra i correctamente clasificadas por m
r_i = número de perturbaciones de la muestra i correctamente clasificadas por a
```

entonces:

```text
r_hat_tau_i = u_i / r_i
```

El estimador agregado sobre `S_f` es:

```text
r_hat_tau(S_f) = (1 / |S_f|) * sum_i r_hat_tau_i.
```

## Configuración experimental descrita en el paper para evaluación RK

Para la evaluación de RK, el paper describe las siguientes configuraciones de perturbación:

### Gaussian noise

Para Gaussian noise, el paper usa perturbaciones gaussianas y reporta curvas de RK con:

```text
p = 2
c = 100
x'_j generado mediante Gaussian noise alrededor de x
```

En la implementación descrita en el apéndice, se indica el uso de `torchattacks.GN(model, std=tau)`.

### FGSM

El paper también evalúa RK usando FGSM dirigido:

```text
p = infinity
c = 100
torchattacks.FGSM(model, eps=tau)
```

### PGD

El paper también evalúa RK usando PGD dirigido:

```text
p = infinity
c = 100
torchattacks.PGD(model, eps=tau, alpha=2/255, steps=pgd_epoch, random_start=True)
```

## Configuración mínima recomendada para reproducir la métrica principal del paper

Para una primera integración estricta de RK en el pipeline, sin añadir variantes no definidas en el paper, se debe implementar al menos:

```text
metric_name: residual_knowledge
perturbation: gaussian
p: 2
c: 100
tau: configurable
models_compared: unlearned model m vs re-trained model a
aggregation: mean over forget samples
```

El valor de `tau` debe ser un parámetro explícito de la métrica. El paper usa curvas sobre varios valores de `tau` y en algunos experimentos utiliza valores alrededor de `tau = 0.03`.

## Requisitos de entrada

La función de evaluación RK debe recibir:

```text
model_unlearned:
    Modelo m obtenido por el método de unlearning.

model_retrained:
    Modelo a entrenado desde cero usando únicamente S_r.

forget_loader:
    DataLoader de S_f.
    Debe devolver pares (x, y).

c:
    Número de perturbaciones Monte Carlo por muestra.
    Para reproducir la configuración del paper: c = 100.

tau:
    Radio o escala de perturbación.

perturbation:
    Tipo de perturbación usado para generar x'.
    Valores documentados en el paper: gaussian, fgsm, pgd.

device:
    cpu o cuda.

normalization / denormalization:
    Transformaciones necesarias para generar perturbaciones en el espacio correcto de entrada y evaluar los modelos en el formato esperado.
```

## Procedimiento algorítmico

Para cada muestra `(x, y)` de `S_f`:

1. Generar `c` perturbaciones independientes `x'_1, ..., x'_c` alrededor de `x`, según el mecanismo de perturbación elegido.
2. Evaluar el modelo unlearned `m` sobre cada perturbación `x'_j`.
3. Evaluar el modelo re-trained `a` sobre exactamente las mismas perturbaciones `x'_j`.
4. Calcular:

```text
u_i = sum_{j=1}^c 1[argmax(m(x'_j)) = y]
r_i = sum_{j=1}^c 1[argmax(a(x'_j)) = y]
```

5. Calcular RK empírico por muestra:

```text
r_hat_tau_i = u_i / r_i
```

6. Repetir para todas las muestras del forget set.
7. Calcular el RK agregado:

```text
r_hat_tau(S_f) = mean_i(r_hat_tau_i)
```

## Tratamiento de denominador cero

El paper define RK como un cociente. No introduce smoothing, constante `alpha`, clipping, sustitución por 1, sustitución por infinito, ni ninguna regla alternativa para el caso en que el denominador empírico sea cero.

Por tanto, para mantener la implementación estrictamente alineada con el paper:

```text
if r_i > 0:
    r_hat_tau_i = u_i / r_i
else:
    r_hat_tau_i is undefined by the Monte Carlo estimator
```

La implementación no debe aplicar smoothing ni modificar el cociente. En código, el caso `r_i = 0` debe tratarse como valor no definido para la métrica empírica de esa muestra. Una política de software aceptable es devolver `NaN` para esa muestra y, si existe algún `NaN`, devolver `NaN` para el promedio agregado `r_hat_tau(S_f)`, salvo que el benchmark defina explícitamente fuera de esta métrica una política de filtrado.

El conteo de denominadores cero puede guardarse como información de control de calidad de la ejecución, pero no debe presentarse como una métrica RK adicional.

## Pseudocódigo PyTorch

```python
def compute_residual_knowledge(
    model_unlearned,
    model_retrained,
    forget_loader,
    tau,
    c=100,
    perturbation="gaussian",
    device="cuda",
    normalize_fn=None,
    denormalize_fn=None,
    seed=0,
):
    """
    Compute Residual Knowledge exactly as the Monte Carlo estimator described in the paper.

    Returns:
        {
            "rk_tau_forget_set": float or NaN,
            "rk_tau_per_sample": list[float or NaN],
            "u_counts": list[int],
            "r_counts": list[int],
            "tau": float,
            "c": int,
            "perturbation": str,
        }

    Notes:
        - No smoothing is applied.
        - No RK excess or log-excess is computed.
        - If r_i = 0 for a sample, RK for that sample is undefined by the empirical ratio.
    """

    import torch

    model_unlearned.eval()
    model_retrained.eval()

    rng = torch.Generator(device=device)
    rng.manual_seed(seed)

    rk_values = []
    u_counts = []
    r_counts = []

    for x, y in forget_loader:
        x = x.to(device)
        y = y.to(device)

        batch_size = x.shape[0]

        if denormalize_fn is not None:
            x_base = denormalize_fn(x)
        else:
            x_base = x

        correct_unlearned = torch.zeros(batch_size, device=device, dtype=torch.long)
        correct_retrained = torch.zeros(batch_size, device=device, dtype=torch.long)

        for _ in range(c):
            if perturbation == "gaussian":
                noise = torch.randn(
                    x_base.shape,
                    generator=rng,
                    device=device,
                    dtype=x_base.dtype,
                ) * tau
                x_pert = x_base + noise

                # For image datasets with bounded pixel domain.
                x_pert = torch.clamp(x_pert, 0.0, 1.0)

            else:
                raise NotImplementedError(
                    "This minimal implementation only includes Gaussian RK. "
                    "FGSM and PGD require attack-specific implementation."
                )

            if normalize_fn is not None:
                x_pert_model = normalize_fn(x_pert)
            else:
                x_pert_model = x_pert

            with torch.no_grad():
                logits_m = model_unlearned(x_pert_model)
                logits_a = model_retrained(x_pert_model)

                pred_m = logits_m.argmax(dim=1)
                pred_a = logits_a.argmax(dim=1)

            correct_unlearned += (pred_m == y).long()
            correct_retrained += (pred_a == y).long()

        for b in range(batch_size):
            u_i = int(correct_unlearned[b].item())
            r_i = int(correct_retrained[b].item())

            u_counts.append(u_i)
            r_counts.append(r_i)

            if r_i > 0:
                rk_values.append(u_i / r_i)
            else:
                rk_values.append(float("nan"))

    rk_tensor = torch.tensor(rk_values, dtype=torch.float64)

    if torch.isnan(rk_tensor).any():
        rk_tau_forget_set = float("nan")
    else:
        rk_tau_forget_set = rk_tensor.mean().item()

    return {
        "rk_tau_forget_set": rk_tau_forget_set,
        "rk_tau_per_sample": rk_values,
        "u_counts": u_counts,
        "r_counts": r_counts,
        "tau": tau,
        "c": c,
        "perturbation": perturbation,
    }
```

## Salidas de la métrica

La implementación debe producir como salida principal:

```text
rk_tau_forget_set
```

También puede guardar el vector por muestra:

```text
rk_tau_per_sample
```

y los conteos necesarios para reproducibilidad:

```text
u_counts
r_counts
tau
c
perturbation
```

Los conteos `u_counts` y `r_counts` no son métricas adicionales; son los componentes empíricos del estimador Monte Carlo de RK.

## Elementos explícitamente excluidos

Esta especificación excluye los siguientes elementos porque no forman parte de la definición de RK del paper:

```text
rk_smooth_macro
rk_excess_macro
rk_log_excess_macro
alpha smoothing
max(0, RK - 1)
max(0, log(RK))
clipping de RK
sustitución de denominadores cero por constantes
uso de RK como objetivo transformado de Optuna
penalizaciones adicionales derivadas de RK
```

También se excluyen interpretaciones cualitativas adicionales. La implementación debe limitarse al cálculo de `r_hat_tau((x, y))` y `r_hat_tau(S_f)`.

## Checklist de validación

Antes de integrar la métrica en el pipeline, comprobar:

```text
[ ] El modelo unlearned m está en modo eval().
[ ] El modelo re-trained a está en modo eval().
[ ] Ambos modelos reciben exactamente las mismas perturbaciones x'_j.
[ ] Se usa el forget set S_f, no retain ni test.
[ ] Para cada muestra se generan c perturbaciones.
[ ] El numerador es el número de veces que m predice y.
[ ] El denominador es el número de veces que a predice y.
[ ] No se aplica smoothing.
[ ] No se calcula RK Excess ni Log-Excess.
[ ] El agregado es la media aritmética de RK por muestra.
[ ] Los casos r_i = 0 se marcan como no definidos, sin alterar el cociente.
```
