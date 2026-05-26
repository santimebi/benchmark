"""
utils/residual_knowledge.py
───────────────────────────────────────────────
Implementación estricta de la métrica Residual Knowledge (RK) según la especificación
"residual_knowledge_metric_spec_paper_only.md" (sin smoothing ni objetivos de Optuna).
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

def compute_residual_knowledge(
    model_unlearned: nn.Module,
    model_retrained: nn.Module,
    forget_loader: DataLoader,
    tau: float,
    c: int = 100,
    perturbation: str = "gaussian",
    device: str = "cuda",
    normalize_fn = None,
    denormalize_fn = None,
    seed: int = 0,
    mc_chunk_size: int = None,
) -> dict:
    """
    Calcula Residual Knowledge (RK) exactamente como el estimador Monte Carlo descrito en el paper.

    Parámetros:
    -----------
    model_unlearned:
        Modelo 'm' obtenido después de aplicar desaprendizaje.
    model_retrained:
        Modelo 'a' reentrenado desde cero sobre S_r.
    forget_loader:
        DataLoader determinista sobre S_f.
    tau:
        Escala/radio de la perturbación.
    c:
        Número de perturbaciones Monte Carlo por muestra.
    perturbation:
        Tipo de perturbación ("gaussian").
    device:
        Dispositivo de ejecución ("cpu" o "cuda").
    normalize_fn:
        Función opcional para normalizar imágenes en [0, 1] al espacio del modelo.
    denormalize_fn:
        Función opcional para desnormalizar imágenes al espacio [0, 1].
    seed:
        Semilla para reproducibilidad de las perturbaciones.
    mc_chunk_size:
        Tamaño de lote para procesar las 'c' perturbaciones de forma vectorizada.
        Si es None, se procesa 'c' de golpe.

    Retorna:
    --------
    dict
        Métricas de RK y conteos de soporte.
    """
    model_unlearned.eval()
    model_retrained.eval()

    rng = torch.Generator(device=device)
    rng.manual_seed(seed)

    rk_values = []
    u_counts = []
    r_counts = []

    if mc_chunk_size is None:
        mc_chunk_size = c

    # Particionar 'c' en bloques del tamaño indicado por mc_chunk_size
    chunks = []
    remaining = c
    while remaining > 0:
        current_chunk = min(remaining, mc_chunk_size)
        chunks.append(current_chunk)
        remaining -= current_chunk

    for batch in forget_loader:
        if len(batch) >= 2:
            x, y = batch[0], batch[1]
        else:
            raise ValueError("forget_loader debe devolver al menos (x, y).")

        x = x.to(device)
        y = y.to(device)

        batch_size = x.shape[0]
        feature_shape = x.shape[1:]

        if denormalize_fn is not None:
            x_base = denormalize_fn(x)
        else:
            x_base = x

        correct_unlearned = torch.zeros(batch_size, device=device, dtype=torch.long)
        correct_retrained = torch.zeros(batch_size, device=device, dtype=torch.long)

        for chunk_c in chunks:
            # Expansión vectorizada: shape [B, chunk_c, *feature_shape]
            repeat_dims = [1] * (len(feature_shape) + 2)
            repeat_dims[1] = chunk_c
            x_expanded = x_base.unsqueeze(1).repeat(*repeat_dims)

            if perturbation == "gaussian":
                noise = torch.randn(
                    x_expanded.shape,
                    generator=rng,
                    device=device,
                    dtype=x_base.dtype,
                ) * tau
                x_pert = x_expanded + noise

                # Solo aplicar clipping en espacio [0, 1] si es un dataset de imágenes (ej. CIFAR-10)
                if len(feature_shape) > 1:
                    x_pert = torch.clamp(x_pert, 0.0, 1.0)
            else:
                raise NotImplementedError(
                    f"Esta implementación sólo incluye Gaussian RK. Perturbación '{perturbation}' no soportada."
                )

            # Aplanar para pasar al modelo: shape [B * chunk_c, *feature_shape]
            x_pert_flat = x_pert.view(batch_size * chunk_c, *feature_shape)

            if normalize_fn is not None:
                x_pert_model = normalize_fn(x_pert_flat)
            else:
                x_pert_model = x_pert_flat

            with torch.no_grad():
                logits_m = model_unlearned(x_pert_model)
                logits_a = model_retrained(x_pert_model)

                pred_m = logits_m.argmax(dim=1)
                pred_a = logits_a.argmax(dim=1)

            # Devolver a forma [B, chunk_c]
            pred_m = pred_m.view(batch_size, chunk_c)
            pred_a = pred_a.view(batch_size, chunk_c)

            # Sumar aciertos comparando con y expandido a [B, 1]
            y_expanded = y.unsqueeze(1)
            correct_unlearned += (pred_m == y_expanded).long().sum(dim=1)
            correct_retrained += (pred_a == y_expanded).long().sum(dim=1)

        # Calcular cociente RK por cada muestra en el lote
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
