"""
utils/protocols.py
───────────────────────────────────────────────
Define los protocolos de entrenamiento y unlearning (olvido) del benchmark.
Permite registrar diferentes estrategias y llamarlas por su identificador.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def run_standard_training(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    device: torch.device,
    verbose: bool = True
) -> nn.Module:
    """
    Protocolo estándar de entrenamiento supervisado completo.

    Args:
        model: El modelo de PyTorch a entrenar.
        train_loader: DataLoader para el conjunto de entrenamiento.
        val_loader: DataLoader para el conjunto de validación.
        criterion: Función de pérdida (loss function).
        optimizer: Optimizador.
        epochs: Número de épocas de entrenamiento.
        device: Dispositivo de ejecución (cpu o cuda).
        verbose: Si es True, imprime las métricas periódicamente.

    Returns:
        nn.Module: El modelo entrenado.
    """
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
        train_loss = train_loss / total
        train_acc = 100. * correct / total
        
        # Validación
        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for val_inputs, val_targets in val_loader:
                    val_inputs, val_targets = val_inputs.to(device), val_targets.to(device)
                    val_outputs = model(val_inputs)
                    loss = criterion(val_outputs, val_targets)
                    val_loss += loss.item() * val_inputs.size(0)
                    _, val_predicted = val_outputs.max(1)
                    val_total += val_targets.size(0)
                    val_correct += val_predicted.eq(val_targets).sum().item()
                
            val_loss = val_loss / val_total if val_total > 0 else 0.0
            val_acc = 100. * val_correct / val_total if val_total > 0 else 0.0
                
            if verbose:
                print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

    return model


def run_cfk_unlearning(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    device: torch.device,
    verbose: bool = True,
    **kwargs
) -> nn.Module:
    """
    Protocolo CFK (Congelar K capas): Congela todas las capas con parámetros
    excepto las últimas k, y realiza ajuste fino (fine-tuning).

    Args:
        model: El modelo pre-entrenado.
        train_loader: DataLoader del conjunto a entrenar (normalmente solo retain).
        val_loader: DataLoader de validación.
        criterion: Función de pérdida.
        optimizer: Optimizador original (será reconstruido).
        epochs: Épocas de fine-tuning.
        device: Dispositivo (cpu/cuda).
        verbose: Logs de progreso.

    Returns:
        nn.Module: Modelo ajustado (desentrenado).
    """
    hp = kwargs.get("hp", {})
    k = hp.get("k", 1)
    lr = hp.get("lr", 1e-3)

    # 1. Identificar módulos/capas hojas con parámetros (ej. Linear)
    layers = []
    for module in model.modules():
        if len(list(module.children())) == 0 and len(list(module.parameters())) > 0:
            layers.append(module)

    if k > len(layers):
        raise ValueError(f"k={k} es mayor que el número total de capas con parámetros ({len(layers)})")

    # 2. Congelar todos los parámetros del modelo
    for param in model.parameters():
        param.requires_grad = False

    # 3. Descongelar únicamente las últimas k capas
    unfrozen_layers = layers[-k:]
    for layer in unfrozen_layers:
        for param in layer.parameters():
            param.requires_grad = True

    if verbose:
        print(f"[CFK] Congeladas {len(layers) - k} capas. Descongeladas las últimas {k} capas:")
        for i, layer in enumerate(unfrozen_layers):
            print(f"  - Capa descongelada {i+1}: {layer}")

    # 4. Re-crear el optimizador solo para los parámetros activos
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    # 5. Ajuste fino (Fine-tuning)
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        train_loss = train_loss / total
        train_acc = 100. * correct / total

        # Validación
        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for val_inputs, val_targets in val_loader:
                    val_inputs, val_targets = val_inputs.to(device), val_targets.to(device)
                    val_outputs = model(val_inputs)
                    loss = criterion(val_outputs, val_targets)
                    val_loss += loss.item() * val_inputs.size(0)
                    _, val_predicted = val_outputs.max(1)
                    val_total += val_targets.size(0)
                    val_correct += val_predicted.eq(val_targets).sum().item()

            val_loss = val_loss / val_total if val_total > 0 else 0.0
            val_acc = 100. * val_correct / val_total if val_total > 0 else 0.0

            if verbose:
                print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

    return model


# Registro global de protocolos
PROTOCOLS = {
    "standard": run_standard_training,
    "cfk": run_cfk_unlearning
}


def get_protocol(name: str):
    """
    Obtiene la función de entrenamiento/unlearning registrada con el nombre indicado.
    """
    if name not in PROTOCOLS:
        raise ValueError(f"Protocolo '{name}' no reconocido. Opciones: {list(PROTOCOLS.keys())}")
    return PROTOCOLS[name]
