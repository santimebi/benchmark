# Documentación funcional y técnica: RURK-Gaussian

## 1. Objetivo funcional

Implementar el método de olvido **RURK — Robust Unlearning against Residual Knowledge** como un método de fine-tuning que parte de un modelo original entrenado sobre el dataset completo y produce un modelo desaprendido.

El método recibe:

```text
original_model = A(S)
retain_set = S_r
forget_set = S_f
loss_function = CrossEntropyLoss
```

y devuelve:

```text
unlearned_model = w_RURK
```

El método RURK está propuesto en el paper como una estrategia de fine-tuning para reducir la capacidad del modelo de reconocer perturbaciones locales de las muestras del forget set. La formulación del método aparece en la sección 4.2 y el algoritmo operativo se proporciona en el Algorithm 1 del Appendix B.2.

---

## 2. Alcance exacto de esta implementación

Esta implementación debe cubrir exclusivamente:

```text
Método: RURK
Variante: Gaussian
Dataset objetivo: CIFAR-10
Arquitectura objetivo: ResNet18
Entrada del DataLoader: imágenes en rango [0, 1]
Capas congeladas: ninguna
Parámetros entrenables: todos los parámetros del modelo
Referencia algorítmica: Algorithm 1 del paper
```

Queda fuera de alcance:

```text
- RURK-FGSM
- RURK-PGD
- variantes con capas congeladas
- variantes combinadas con CFK, EUK o CFGK
- búsqueda de hiperparámetros
- uso de RK como loss directa
- smoothing o métricas derivadas no definidas en el paper
- modificaciones del objetivo de Algorithm 1
```

---

## 3. Entradas obligatorias

La clase o función de RURK debe recibir:

```text
original_model:
    Modelo PyTorch ya entrenado sobre el dataset completo S.

retain_loader:
    DataLoader de S_r.
    Debe devolver batches (retain_data, retain_targets).

forget_loader:
    DataLoader de S_f.
    Debe devolver batches (forget_data, forget_targets).

device:
    "cuda" o "cpu".

config:
    Configuración del método.
```

Los tensores de imagen procedentes de ambos loaders deben estar en rango:

```text
[0, 1]
```

No se debe aplicar ruido sobre tensores normalizados con mean/std, porque en esta implementación se fija que el DataLoader devuelve directamente imágenes en escala píxel `[0, 1]`.

---

## 4. Salida obligatoria

La función debe devolver:

```text
unlearned_model:
    Copia entrenada de original_model después de aplicar RURK.
```

Opcionalmente, puede devolver también un diccionario de logs, pero eso no forma parte del método matemático; es solo instrumentación de ingeniería.

---

## 5. Configuración para CIFAR-10 + ResNet18

Usar la configuración descrita en el Appendix B.2 para CIFAR-10:

```python
rurk_config = {
    "num_epochs": 2,
    "batch_size": 128,
    "tau": 0.03,
    "v": 1,
    "lambda_f": 0.03,
    "lambda_a": 0.00045,
    "learning_rate": 0.01,
    "momentum": 0.90,
    "weight_decay": 5e-4,
    "max_total_iterations": 200,
    "gradient_clip_norm": 1.0,
}
```

El paper especifica para RURK el uso de:

```text
loss = CrossEntropyLoss
optimizer = SGD
learning_rate = 0.01
momentum = 0.90
weight_decay = 5e-4
scheduler = CosineAnnealingLR
gradient_clip_norm = 1.0
max_total_iterations = 200
tau = 0.03
v = 1
```

Para CIFAR-10 fija:

```text
N = 2
lambda_f = 0.03
lambda_a = 0.00045
```

---

## 6. Loss de RURK según Algorithm 1

En cada iteración se calculan tres pérdidas:

```text
RLoss:
    CrossEntropyLoss sobre un batch del retain set.

FLoss:
    CrossEntropyLoss sobre un batch del forget set original.

AdvFLoss:
    CrossEntropyLoss sobre un batch perturbado del forget set.
```

La loss total debe ser:

```text
Loss = RLoss - lambda_f * FLoss - lambda_a * AdvFLoss
```

Esta es la forma explícita que aparece en Algorithm 1. En consecuencia, aunque la sección 4.2 presenta la formulación general con retain loss y pérdida adversarial sobre perturbaciones vulnerables, para esta implementación debe seguirse el pseudocódigo operativo del appendix.

---

## 7. Construcción de las perturbaciones Gaussian

La variante que se implementa es **RURK-Gaussian**.

Para cada batch del forget set:

```text
forget_data:
    Tensor de shape [B, C, H, W].

forget_targets:
    Tensor de shape [B].
```

Se genera una perturbación:

```text
noise ~ Normal(0, tau^2)
adv_forget_data = forget_data + noise
adv_forget_data = clamp(adv_forget_data, 0, 1)
```

Con esta especificación:

```text
tau = 0.03
v = 1
```

Por tanto, solo se genera **una** muestra perturbada por cada imagen del forget batch.

El paper indica que, para mejorar la eficiencia, se puede aproximar el vulnerable set tomando:

```text
V((x, y), tau) = B_p(x, tau)
```

y fija:

```text
v = 1
```

De esta forma, solo se extrae una muestra perturbada de una distribución gaussiana multivariante centrada en `x` con desviación típica `tau`.

---

## 8. Algoritmo funcional

```text
Input:
    original_model
    retain_loader
    forget_loader
    config

Output:
    unlearned_model

Procedure:
    1. Copiar original_model para obtener model.
    2. Poner model en modo train.
    3. Crear CrossEntropyLoss.
    4. Crear optimizador SGD.
    5. Crear scheduler CosineAnnealingLR.
    6. Crear iterador sobre forget_loader.

    7. Para epoch en {1, ..., num_epochs}:
           Para cada batch del retain_loader:

               7.1. Obtener retain_data, retain_targets.

               7.2. Obtener forget_data, forget_targets usando next(forget_iterator).
                    Si forget_iterator se agota, reiniciarlo.

               7.3. Calcular:
                        retain_logits = model(retain_data)
                        RLoss = CE(retain_logits, retain_targets)

               7.4. Calcular:
                        forget_logits = model(forget_data)
                        FLoss = CE(forget_logits, forget_targets)

               7.5. Generar adv_forget_data:
                        noise ~ Normal(0, tau^2)
                        adv_forget_data = clamp(forget_data + noise, 0, 1)

               7.6. Calcular:
                        adv_forget_logits = model(adv_forget_data)
                        AdvFLoss = CE(adv_forget_logits, forget_targets)

               7.7. Calcular:
                        Loss = RLoss - lambda_f * FLoss - lambda_a * AdvFLoss

               7.8. Hacer backward.

               7.9. Aplicar gradient clipping con norma máxima 1.0.

               7.10. Hacer optimizer.step().

               7.11. Hacer scheduler.step().

               7.12. Incrementar contador de iteraciones.

               7.13. Si se alcanza max_total_iterations = 200, parar.

    8. Devolver model.
```

---

## 9. Pseudocódigo PyTorch implementable

```python
import copy
import torch
import torch.nn as nn
import torch.optim as optim


def generate_gaussian_perturbation(x: torch.Tensor, tau: float) -> torch.Tensor:
    """
    RURK-Gaussian perturbation.

    Assumption:
        x is in pixel range [0, 1].

    Returns:
        x_adv = clamp(x + noise, 0, 1),
        where noise ~ Normal(0, tau^2).
    """
    noise = torch.randn_like(x) * tau
    x_adv = x + noise
    x_adv = torch.clamp(x_adv, min=0.0, max=1.0)
    return x_adv


def rurk_gaussian_unlearning(
    original_model: torch.nn.Module,
    retain_loader,
    forget_loader,
    device: torch.device,
    num_epochs: int = 2,
    tau: float = 0.03,
    v: int = 1,
    lambda_f: float = 0.03,
    lambda_a: float = 0.00045,
    learning_rate: float = 0.01,
    momentum: float = 0.90,
    weight_decay: float = 5e-4,
    max_total_iterations: int = 200,
    gradient_clip_norm: float = 1.0,
):
    """
    Implements RURK-Gaussian for CIFAR-10 + ResNet18 following Algorithm 1.

    This implementation:
        - starts from the original model A(S);
        - uses retain batches from S_r;
        - uses forget batches from S_f;
        - generates Gaussian perturbations around forget samples;
        - optimizes Loss = RLoss - lambda_f * FLoss - lambda_a * AdvFLoss;
        - updates all trainable parameters;
        - does not freeze layers.
    """

    if v != 1:
        raise ValueError("This paper-only implementation is restricted to v = 1.")

    model = copy.deepcopy(original_model)
    model = model.to(device)
    model.train()

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(
        model.parameters(),
        lr=learning_rate,
        momentum=momentum,
        weight_decay=weight_decay,
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max_total_iterations,
    )

    forget_iterator = iter(forget_loader)
    total_iterations = 0

    for epoch in range(num_epochs):
        for retain_data, retain_targets in retain_loader:

            if total_iterations >= max_total_iterations:
                break

            retain_data = retain_data.to(device)
            retain_targets = retain_targets.to(device)

            try:
                forget_data, forget_targets = next(forget_iterator)
            except StopIteration:
                forget_iterator = iter(forget_loader)
                forget_data, forget_targets = next(forget_iterator)

            forget_data = forget_data.to(device)
            forget_targets = forget_targets.to(device)

            optimizer.zero_grad(set_to_none=True)

            retain_logits = model(retain_data)
            retain_loss = criterion(retain_logits, retain_targets)

            forget_logits = model(forget_data)
            forget_loss = criterion(forget_logits, forget_targets)

            adv_forget_data = generate_gaussian_perturbation(
                forget_data,
                tau=tau,
            )

            adv_forget_logits = model(adv_forget_data)
            adv_forget_loss = criterion(
                adv_forget_logits,
                forget_targets,
            )

            total_loss = (
                retain_loss
                - lambda_f * forget_loss
                - lambda_a * adv_forget_loss
            )

            total_loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=gradient_clip_norm,
            )

            optimizer.step()
            scheduler.step()

            total_iterations += 1

        if total_iterations >= max_total_iterations:
            break

    return model
```

---

## 10. Requisitos técnicos

La implementación debe cumplir:

```text
1. original_model no debe modificarse in-place.
   Se debe trabajar sobre una copia profunda.

2. El modelo debe ponerse en train() durante RURK.

3. No se deben congelar capas.

4. El ruido Gaussian debe aplicarse a imágenes en rango [0, 1].

5. Después de añadir ruido, las imágenes deben mantenerse en [0, 1] mediante clamp.

6. La loss debe ser exactamente:
       RLoss - lambda_f * FLoss - lambda_a * AdvFLoss

7. El optimizador debe ser SGD.

8. Se debe aplicar gradient clipping con norma máxima 1.0.

9. El proceso debe detenerse al alcanzar max_total_iterations = 200.

10. La salida debe ser el modelo actualizado.
```

---

## 11. Tests mínimos de implementación

### Test 1: no modificar el modelo original

```text
Guardar copia de los parámetros de original_model.
Ejecutar RURK.
Comprobar que original_model conserva exactamente sus parámetros.
```

Resultado esperado:

```text
original_model no cambia.
unlearned_model sí cambia.
```

### Test 2: todas las capas entrenables

```text
Comprobar que no se ha llamado a requires_grad = False sobre ninguna capa.
```

Resultado esperado:

```text
Todos los parámetros que eran entrenables antes de RURK siguen siendo entrenables.
```

### Test 3: rango de perturbaciones

```text
Tomar un batch de forget_data.
Generar adv_forget_data.
Comprobar adv_forget_data.min() >= 0 y adv_forget_data.max() <= 1.
```

Resultado esperado:

```text
Las perturbaciones están en rango [0, 1].
```

### Test 4: loss con signos correctos

Comprobar que el código usa:

```text
retain_loss - lambda_f * forget_loss - lambda_a * adv_forget_loss
```

y no:

```text
retain_loss + lambda_f * forget_loss + lambda_a * adv_forget_loss
```

Este test es crítico, porque el objetivo de Algorithm 1 usa términos negativos para forget y adversarial forget.

### Test 5: parada por número máximo de iteraciones

```text
Ejecutar con retain_loader suficientemente largo.
Comprobar que total_iterations <= 200.
```

Resultado esperado:

```text
El método no supera max_total_iterations.
```

---

## 12. Aspectos no especificados completamente por el paper

Aunque la implementación anterior es suficiente para RURK-Gaussian en esta configuración, hay algunos puntos que el paper no deja totalmente cerrados:

```text
1. Reinicio del forget_loader:
   Algorithm 1 usa nextIter(FLoader), pero no especifica qué hacer cuando FLoader se agota.
   En la implementación se reinicia el iterador.

2. Argumentos exactos de CosineAnnealingLR:
   El paper indica CosineAnnealingLR y máximo de 200 iteraciones, pero no detalla todos los argumentos.
   Se usa T_max = 200 como elección técnica coherente con ese límite.

3. Implementación exacta de Gaussian noise:
   El paper menciona TorchAttacks GN con std=tau.
   Aquí se implementa explícitamente noise ~ Normal(0, tau^2) y clamp a [0, 1], consistente con el DataLoader en rango [0, 1].

4. Seeds:
   El paper menciona seeds [131, 42, 7], pero en un pipeline experimental normalmente se usan seeds propias por run.
   La función no fija la seed internamente para no romper la gestión global de reproducibilidad.
```

Estos puntos no cambian la definición del método, pero deben quedar documentados para que el programador no los interprete como decisiones matemáticas nuevas.

---

## 13. Resumen de implementación

La clase final podría llamarse:

```text
RURKGaussianUnlearner
```

Debe implementar:

```text
fit(original_model, retain_loader, forget_loader) -> unlearned_model
```

con esta configuración fija para CIFAR-10 + ResNet18:

```text
num_epochs = 2
tau = 0.03
v = 1
lambda_f = 0.03
lambda_a = 0.00045
optimizer = SGD
lr = 0.01
momentum = 0.90
weight_decay = 5e-4
scheduler = CosineAnnealingLR
max_total_iterations = 200
gradient_clip_norm = 1.0
loss = CrossEntropyLoss
```

La implementación debe seguir **Algorithm 1** y no debe añadir ningún mecanismo externo al paper.
