# CFGK — Decisiones cerradas y búsqueda exploratoria de hiperparámetros

## Estado de la especificación

Esta versión incorpora la decisión de permitir regiones de búsqueda que incluyen valores negativos para `c` y `gamma`, con el objetivo de evaluar empíricamente si esos valores pueden tener algún sentido práctico dentro del pipeline.

La búsqueda se interpreta como **búsqueda directa sobre los valores de `c` y `gamma`**, no como búsqueda sobre exponentes logarítmicos, salvo que en el futuro se indique explícitamente lo contrario.

---

# 1. Regularización CFGK

La pérdida total usada durante el proceso de unlearning es:

```text
loss_total = loss_task + penalty_cfgk
```

donde:

```text
loss_task = CrossEntropy(logits, labels)
```

y:

```text
penalty_cfgk = c * exp( - (norm + beta)^gamma )
```

con:

```text
norm = || theta_S - theta_0,S ||_2
```

---

# 2. Conjunto de parámetros regularizados `S`

La regularización se calcula únicamente sobre los pesos de las capas no congeladas.

El conjunto `S` está formado por parámetros que cumplen simultáneamente:

```text
parameter_name corresponde a un tensor weight
parameter.requires_grad == True
parameter pertenece a un bloque descongelado por CFGK
```

Por tanto, no se regularizan parámetros congelados.

---

# 3. Exclusión de bias y buffers

La regularización se aplica exclusivamente sobre tensores:

```text
weight
```

No se incluyen:

```text
bias
running_mean
running_var
num_batches_tracked
```

Esto implica que, si se recorre `model.named_parameters()`, deben seleccionarse únicamente parámetros cuyo nombre corresponda inequívocamente a pesos entrenables.

---

# 4. Norma L2 global

Se usa una única norma L2 global sobre todos los pesos seleccionados.

No se calcula una penalización independiente por capa ni por bloque.

La norma se calcula como:

```text
norm = sqrt( sum_{p in S} sum( (p - p0)^2 ) )
```

donde:

```text
p  = peso actual del modelo durante unlearning
p0 = peso correspondiente del modelo base pre-unlearning
```

---

# 5. Bloques en ResNet18

En ResNet18, la congelación y descongelación se realiza exclusivamente por bloques.

La estructura considerada es:

```text
1 - 4 - 4 - 4 - 4 - 1
```

Conceptualmente:

```text
bloque inicial
layer1
layer2
layer3
layer4
clasificador final
```

Solo tiene sentido congelar o descongelar bloques completos.

Por tanto:

```text
bloque congelado    -> todos sus parámetros tienen requires_grad = False
bloque descongelado -> sus pesos weight entrenables tienen requires_grad = True
```

La regularización CFGK se calcula únicamente sobre los pesos `weight` de los bloques descongelados.

---

# 6. Modelo de referencia `theta_0`

El modelo de referencia `theta_0` es el modelo base en el momento inmediatamente previo a empezar a aplicar el método de unlearning.

No es:

```text
modelo naive
modelo reentrenado desde cero
modelo tras una época de unlearning
modelo tras aplicar parcialmente CFGK
```

Debe guardarse justo antes del bucle de unlearning:

```python
reference_state_dict = {
    name: tensor.detach().clone()
    for name, tensor in model.state_dict().items()
}
```

Esta copia debe permanecer fija durante todo el proceso.

---

# 7. Parámetro `beta`

`beta` queda fijado a:

```text
beta = 1e-12
```

Su función es exclusivamente evitar problemas de cálculo numérico durante la ejecución.

No se considera hiperparámetro de búsqueda.

---

# 8. Región de búsqueda de hiperparámetros

La región de búsqueda queda fijada como:

```text
c ∈ [-2, 8]
gamma ∈ [-1, 10]
```

Esta región se usa para valorar empíricamente si valores negativos de `c` o `gamma` pueden tener sentido práctico.

La expectativa experimental es que, si esos valores no son útiles, la búsqueda de hiperparámetros debería desplazarse rápidamente hacia regiones positivas.

---

# 9. Interpretación técnica de valores negativos

Definimos:

```text
x = norm + beta
```

con:

```text
x > 0
```

La penalización es:

```text
P(x) = c * exp( - x^gamma )
```

Su derivada respecto a `x` es:

```text
dP/dx = - c * gamma * x^(gamma - 1) * exp( - x^gamma )
```

Como:

```text
x^(gamma - 1) > 0
exp( - x^gamma ) > 0
```

el signo de la derivada depende de:

```text
- c * gamma
```

Por tanto, el comportamiento cualitativo depende del signo de `c * gamma`.

---

## 9.1. Caso `c > 0`, `gamma > 0`

Este es el comportamiento estándar esperado de CFGK.

La penalización decrece cuando aumenta la distancia entre el modelo actual y el modelo base.

Al minimizar:

```text
loss_total = loss_task + penalty_cfgk
```

el término de regularización empuja a aumentar la distancia `norm`.

Interpretación:

```text
presión de alejamiento respecto al modelo base
```

---

## 9.2. Caso `c < 0`, `gamma > 0`

La penalización es negativa y aumenta hacia cero cuando crece la distancia.

Al minimizar la loss, el optimizador tiende a preferir valores más negativos de la penalización, lo que favorece distancias pequeñas.

Interpretación:

```text
presión de permanencia cerca del modelo base
```

Este comportamiento es contrario a la intuición principal de CFGK como regularización de alejamiento.

Puede tener sentido únicamente como región exploratoria o como control experimental.

---

## 9.3. Caso `c > 0`, `gamma < 0`

En este caso:

```text
x^gamma = 1 / x^|gamma|
```

La penalización tiende a ser muy pequeña cuando `norm` está cerca de cero, y aumenta hacia `c` cuando `norm` crece.

Al minimizar, se favorecen distancias pequeñas.

Interpretación:

```text
presión de permanencia cerca del modelo base
```

Además, aunque `beta = 1e-12` evita división exacta por cero, este caso puede producir comportamientos numéricos delicados cerca de `norm = 0`.

---

## 9.4. Caso `c < 0`, `gamma < 0`

En este caso, la penalización es negativa y tiende hacia `c` cuando la distancia crece.

Como `c < 0`, al minimizar la loss puede existir presión para aumentar la distancia.

Interpretación:

```text
presión de alejamiento, pero mediante una recompensa negativa acotada
```

Este caso podría tener algún sentido práctico exploratorio, pero no debe interpretarse como la forma estándar positiva de la penalización CFGK.

---

## 9.5. Caso `gamma = 0`

Si:

```text
gamma = 0
```

entonces:

```text
(norm + beta)^gamma = 1
```

y por tanto:

```text
penalty_cfgk = c * exp(-1)
```

La penalización es constante e independiente de los pesos del modelo.

Consecuencia:

```text
no hay efecto regularizador real sobre los parámetros
```

Este caso puede permitirse dentro de la búsqueda, pero debe interpretarse como degenerado.

---

## 9.6. Caso `c = 0`

Si:

```text
c = 0
```

entonces:

```text
penalty_cfgk = 0
loss_total = loss_task
```

En este caso CFGK se reduce a CFK sin regularización exponencial adicional.

---

# 10. Criterio de interpretación de la búsqueda

Dado que se permite explorar:

```text
c ∈ [-2, 8]
gamma ∈ [-1, 10]
```

la evaluación posterior debe registrar no solo el mejor valor de la métrica objetivo, sino también la región de signos encontrada.

Se recomienda registrar para cada trial:

```text
c
gamma
sign(c)
sign(gamma)
sign(c * gamma)
cfgk_norm_initial
cfgk_norm_final
loss_cfgk_penalty_initial
loss_cfgk_penalty_final
```

Esto permite distinguir entre:

```text
1. CFGK estándar: c > 0, gamma > 0
2. CFGK degenerado: c = 0 o gamma = 0
3. CFGK de permanencia: c * gamma < 0
4. CFGK exploratorio de recompensa negativa: c < 0, gamma < 0
```

---

# 11. Recomendación de implementación

La implementación debe permitir los rangos decididos:

```text
c ∈ [-2, 8]
gamma ∈ [-1, 10]
```

pero debe añadir comprobaciones de estabilidad numérica.

Como mínimo, después de calcular la penalización debe verificarse:

```python
if not torch.isfinite(penalty_cfgk):
    raise FloatingPointError("CFGK penalty became non-finite.")
```

También se recomienda verificar:

```python
if not torch.isfinite(loss_total):
    raise FloatingPointError("CFGK total loss became non-finite.")
```

No se debe forzar automáticamente `c >= 0` ni `gamma > 0` si el objetivo experimental es estudiar si valores negativos tienen sentido práctico.

---

# 12. Pseudocódigo implementable

```text
Entrada:
    model
    reference_state_dict
    c
    gamma
    beta = 1e-12

Procedimiento:
    1. logits = model(inputs)

    2. ce_loss = CrossEntropy(logits, labels)

    3. squared_sum = 0

    4. Para cada (name, parameter) en model.named_parameters():

           si parameter.requires_grad == False:
               continuar

           si name no corresponde a weight:
               continuar

           si parameter no pertenece a bloque descongelado:
               continuar

           p0 = reference_state_dict[name]

           squared_sum += sum((parameter - p0)^2)

    5. norm = sqrt(squared_sum)

    6. penalty_cfgk = c * exp( - (norm + beta)^gamma )

    7. loss_total = ce_loss + penalty_cfgk

    8. verificar que penalty_cfgk y loss_total son finitos

    9. loss_total.backward()

    10. optimizer.step()

Salida:
    loss_total
    ce_loss
    penalty_cfgk
    norm
```

---

# 13. Implementación PyTorch de referencia

```python
import torch
import torch.nn as nn


class CFGKRegularizedLoss(nn.Module):
    def __init__(
        self,
        reference_state_dict: dict,
        c: float,
        gamma: float,
        beta: float = 1e-12,
    ):
        super().__init__()

        self.reference_state_dict = {
            name: tensor.detach().clone()
            for name, tensor in reference_state_dict.items()
        }

        self.c = float(c)
        self.gamma = float(gamma)
        self.beta = float(beta)
        self.ce = nn.CrossEntropyLoss()

    def _is_regularized_weight(self, name: str, parameter: torch.nn.Parameter) -> bool:
        if not parameter.requires_grad:
            return False

        if not name.endswith("weight"):
            return False

        return True

    def cfgk_penalty(self, model: nn.Module):
        device = next(model.parameters()).device
        squared_sum = torch.zeros((), device=device)

        for name, parameter in model.named_parameters():
            if not self._is_regularized_weight(name, parameter):
                continue

            if name not in self.reference_state_dict:
                raise KeyError(f"Parameter {name} not found in reference_state_dict.")

            reference_parameter = self.reference_state_dict[name].to(
                device=parameter.device,
                dtype=parameter.dtype,
            )

            if parameter.shape != reference_parameter.shape:
                raise ValueError(
                    f"Shape mismatch for parameter {name}: "
                    f"current={tuple(parameter.shape)}, "
                    f"reference={tuple(reference_parameter.shape)}"
                )

            squared_sum = squared_sum + torch.sum((parameter - reference_parameter) ** 2)

        norm = torch.sqrt(squared_sum)
        penalty = self.c * torch.exp(-torch.pow(norm + self.beta, self.gamma))

        if not torch.isfinite(penalty):
            raise FloatingPointError(
                f"CFGK penalty became non-finite. "
                f"c={self.c}, gamma={self.gamma}, beta={self.beta}, norm={norm.item()}"
            )

        return penalty, norm

    def forward(self, model: nn.Module, logits: torch.Tensor, labels: torch.Tensor):
        ce_loss = self.ce(logits, labels)
        penalty, norm = self.cfgk_penalty(model)
        total_loss = ce_loss + penalty

        if not torch.isfinite(total_loss):
            raise FloatingPointError(
                f"CFGK total loss became non-finite. "
                f"ce_loss={ce_loss.item()}, penalty={penalty.item()}, "
                f"c={self.c}, gamma={self.gamma}, norm={norm.item()}"
            )

        logs = {
            "loss_total": total_loss.detach(),
            "loss_ce": ce_loss.detach(),
            "loss_cfgk_penalty": penalty.detach(),
            "cfgk_norm": norm.detach(),
            "cfgk_c": self.c,
            "cfgk_gamma": self.gamma,
            "cfgk_beta": self.beta,
            "cfgk_sign_c_gamma": float(self.c * self.gamma),
        }

        return total_loss, logs
```

---

# 14. Tests mínimos adicionales

## Test 1 — `c = 0`

Debe cumplirse:

```text
penalty_cfgk = 0
loss_total = ce_loss
```

## Test 2 — `gamma = 0`

Debe cumplirse:

```text
penalty_cfgk = c * exp(-1)
```

independientemente de `norm`.

## Test 3 — `c > 0`, `gamma > 0`

Debe verificarse que la penalización decrece cuando aumenta `norm`.

## Test 4 — `c < 0`, `gamma > 0`

Debe verificarse que la penalización favorece normas pequeñas.

## Test 5 — `c > 0`, `gamma < 0`

Debe verificarse que la penalización favorece normas pequeñas.

## Test 6 — `c < 0`, `gamma < 0`

Debe verificarse que la penalización puede favorecer normas grandes, pero como recompensa negativa acotada.

## Test 7 — valores no finitos

Debe verificarse que la implementación detecta y detiene el entrenamiento si `penalty_cfgk` o `loss_total` se vuelven no finitos.

---

# 15. Resumen final

CFGK se implementa como una extensión de CFK que añade una regularización exponencial a la CrossEntropy. La regularización se calcula sobre la norma L2 global entre los pesos actuales y los pesos del modelo base pre-unlearning, restringida a los tensores `weight` de los bloques descongelados de ResNet18 con `requires_grad=True`.

La arquitectura ResNet18 se trata por bloques `1-4-4-4-4-1`, y solo se congela o descongela por bloque. `beta` queda fijado a `1e-12`.

La región de búsqueda queda fijada como:

```text
c ∈ [-2, 8]
gamma ∈ [-1, 10]
```

Estos valores se permiten de forma exploratoria para comprobar si regiones negativas tienen algún sentido práctico. La interpretación posterior debe distinguir entre comportamiento estándar de CFGK, casos degenerados y configuraciones que inducen permanencia cerca del modelo base en lugar de alejamiento.
