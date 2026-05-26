"""
utils/protocols.py
───────────────────────────────────────────────
Define los protocolos de entrenamiento y unlearning (olvido) del benchmark.
Permite registrar diferentes estrategias y llamarlas por su identificador.
"""

import copy
import torch
import torch.nn as nn
import torch.optim as optim
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
    excepto las indicadas, y realiza ajuste fino (fine-tuning).
    Soporta la Opción B (bloques completos) para arquitecturas tipo ResNet.

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

    # 1. Congelar todos los parámetros del modelo
    for param in model.parameters():
        param.requires_grad = False

    # 2. Identificar si es una arquitectura tipo ResNet para aplicar Opción B (bloques)
    resnet_obj = model
    if not hasattr(resnet_obj, "layer4") and hasattr(resnet_obj, "model"):
        resnet_obj = resnet_obj.model

    is_resnet = hasattr(resnet_obj, "layer1") and hasattr(resnet_obj, "layer2") and hasattr(resnet_obj, "layer3") and hasattr(resnet_obj, "layer4")

    if is_resnet:
        # Opción B: Descongelación por bloques para ResNet
        unfrozen_modules = []
        if k >= 1:
            if hasattr(resnet_obj, "fc"):
                unfrozen_modules.append(resnet_obj.fc)
        if k >= 5:
            unfrozen_modules.append(resnet_obj.layer4)
        if k >= 9:
            unfrozen_modules.append(resnet_obj.layer3)
        if k >= 13:
            unfrozen_modules.append(resnet_obj.layer2)
        if k >= 17:
            unfrozen_modules.append(resnet_obj.layer1)

        for m in unfrozen_modules:
            for param in m.parameters():
                param.requires_grad = True

        if k >= 18:
            # Descongelar todo el modelo
            for param in model.parameters():
                param.requires_grad = True

        if verbose:
            print(f"[CFK ResNet] Congeladas capas. Descongeladas capas hasta bloque k={k}:")
            for m in unfrozen_modules:
                print(f"  - Módulo descongelado: {m.__class__.__name__}")
    else:
        # Fallback estándar: Identificar módulos hojas y descongelar últimas k capas
        layers = []
        for module in model.modules():
            if len(list(module.children())) == 0 and len(list(module.parameters())) > 0:
                layers.append(module)

        if k > len(layers):
            raise ValueError(f"k={k} es mayor que el número total de capas con parámetros ({len(layers)})")

        unfrozen_layers = layers[-k:]
        for layer in unfrozen_layers:
            for param in layer.parameters():
                param.requires_grad = True

        if verbose:
            print(f"[CFK] Congeladas {len(layers) - k} capas. Descongeladas las últimas {k} capas:")
            for i, layer in enumerate(unfrozen_layers):
                print(f"  - Capa descongelada {i+1}: {layer}")

    # 3. Re-crear el optimizador solo para los parámetros activos
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    # 4. Ajuste fino (Fine-tuning)
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


def run_euk_unlearning(
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
    Protocolo EUK (Erase Unfrozen K-layers): Congela todas las capas excepto las k-últimas
    (o bloques en ResNet), inicializa a 0 los parámetros de las capas descongeladas,
    y realiza ajuste fino (fine-tuning).

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

    # 1. Congelar todos los parámetros del modelo
    for param in model.parameters():
        param.requires_grad = False

    # 2. Identificar si es una arquitectura tipo ResNet para aplicar Opción B (bloques)
    resnet_obj = model
    if not hasattr(resnet_obj, "layer4") and hasattr(resnet_obj, "model"):
        resnet_obj = resnet_obj.model

    is_resnet = hasattr(resnet_obj, "layer1") and hasattr(resnet_obj, "layer2") and hasattr(resnet_obj, "layer3") and hasattr(resnet_obj, "layer4")

    if is_resnet:
        # Opción B: Descongelación por bloques para ResNet
        unfrozen_modules = []
        if k >= 1:
            if hasattr(resnet_obj, "fc"):
                unfrozen_modules.append(resnet_obj.fc)
        if k >= 5:
            unfrozen_modules.append(resnet_obj.layer4)
        if k >= 9:
            unfrozen_modules.append(resnet_obj.layer3)
        if k >= 13:
            unfrozen_modules.append(resnet_obj.layer2)
        if k >= 17:
            unfrozen_modules.append(resnet_obj.layer1)

        for m in unfrozen_modules:
            for param in m.parameters():
                param.requires_grad = True

        if k >= 18:
            # Descongelar todo el modelo
            for param in model.parameters():
                param.requires_grad = True

        if verbose:
            print(f"[EUK ResNet] Congeladas capas. Descongeladas capas hasta bloque k={k}:")
            for m in unfrozen_modules:
                print(f"  - Módulo descongelado: {m.__class__.__name__}")
    else:
        # Fallback estándar: Identificar módulos hojas y descongelar últimas k capas
        layers = []
        for module in model.modules():
            if len(list(module.children())) == 0 and len(list(module.parameters())) > 0:
                layers.append(module)

        if k > len(layers):
            raise ValueError(f"k={k} es mayor que el número total de capas con parámetros ({len(layers)})")

        unfrozen_layers = layers[-k:]
        for layer in unfrozen_layers:
            for param in layer.parameters():
                param.requires_grad = True

        if verbose:
            print(f"[EUK] Congeladas {len(layers) - k} capas. Descongeladas las últimas {k} capas:")
            for i, layer in enumerate(unfrozen_layers):
                print(f"  - Capa descongelada {i+1}: {layer}")

    # 2.5 Reinicializar a 0 los pesos de las capas no congeladas (las que tienen requires_grad=True)
    with torch.no_grad():
        for param in model.parameters():
            if param.requires_grad:
                param.zero_()
        if verbose:
            print("[EUK] Reinicializados a 0 los pesos de las capas descongeladas.")

    # 3. Re-crear el optimizador solo para los parámetros activos
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    # 4. Ajuste fino (Fine-tuning)
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

        norm = torch.sqrt(squared_sum + 1e-8)
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


def run_cfgk_unlearning(
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
    Protocolo CFGK: Similar a CFK pero usa una regularización exponencial
    basada en la distancia L2 de los pesos descongelados respecto al modelo base.
    """
    hp = kwargs.get("hp", {})
    k = hp.get("k", 1)
    lr = hp.get("lr", 1e-3)
    c = hp.get("c", 1.0)
    gamma = hp.get("gamma", 1.0)
    logs_dict = kwargs.get("logs_dict", None)

    # 1. Congelar todos los parámetros
    for param in model.parameters():
        param.requires_grad = False

    # 2. Descongelar bloques/capas (Lógica idéntica a CFK)
    resnet_obj = model
    if not hasattr(resnet_obj, "layer4") and hasattr(resnet_obj, "model"):
        resnet_obj = resnet_obj.model

    is_resnet = hasattr(resnet_obj, "layer1") and hasattr(resnet_obj, "layer2") and hasattr(resnet_obj, "layer3") and hasattr(resnet_obj, "layer4")

    if is_resnet:
        unfrozen_modules = []
        if k >= 1:
            if hasattr(resnet_obj, "fc"):
                unfrozen_modules.append(resnet_obj.fc)
        if k >= 5:
            unfrozen_modules.append(resnet_obj.layer4)
        if k >= 9:
            unfrozen_modules.append(resnet_obj.layer3)
        if k >= 13:
            unfrozen_modules.append(resnet_obj.layer2)
        if k >= 17:
            unfrozen_modules.append(resnet_obj.layer1)

        for m in unfrozen_modules:
            for param in m.parameters():
                param.requires_grad = True

        if k >= 18:
            for param in model.parameters():
                param.requires_grad = True

        if verbose:
            print(f"[CFGK ResNet] Congeladas capas. Descongeladas capas hasta bloque k={k}:")
            for m in unfrozen_modules:
                print(f"  - Módulo descongelado: {m.__class__.__name__}")
    else:
        layers = []
        for module in model.modules():
            if len(list(module.children())) == 0 and len(list(module.parameters())) > 0:
                layers.append(module)

        if k > len(layers):
            raise ValueError(f"k={k} es mayor que el número total de capas con parámetros ({len(layers)})")

        unfrozen_layers = layers[-k:]
        for layer in unfrozen_layers:
            for param in layer.parameters():
                param.requires_grad = True

        if verbose:
            print(f"[CFGK] Congeladas {len(layers) - k} capas. Descongeladas las últimas {k} capas:")
            for i, layer in enumerate(unfrozen_layers):
                print(f"  - Capa descongelada {i+1}: {layer}")

    # 3. Guardar estado de referencia y configurar Loss
    reference_state_dict = {
        name: tensor.detach().clone()
        for name, tensor in model.state_dict().items()
    }
    cfgk_loss_fn = CFGKRegularizedLoss(
        reference_state_dict=reference_state_dict,
        c=c,
        gamma=gamma,
        beta=1e-12
    )

    # 4. Re-crear el optimizador solo para parámetros activos
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    first_batch_logs = None
    last_batch_logs = None

    # 5. Ajuste fino
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            
            # Usar CFGKRegularizedLoss
            loss, batch_logs = cfgk_loss_fn(model, outputs, targets)
            
            if first_batch_logs is None:
                first_batch_logs = batch_logs
            last_batch_logs = batch_logs
                
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
                    # En validación usamos CrossEntropy pura
                    v_loss = criterion(val_outputs, val_targets)
                    val_loss += v_loss.item() * val_inputs.size(0)
                    _, val_predicted = val_outputs.max(1)
                    val_total += val_targets.size(0)
                    val_correct += val_predicted.eq(val_targets).sum().item()

            val_loss = val_loss / val_total if val_total > 0 else 0.0
            val_acc = 100. * val_correct / val_total if val_total > 0 else 0.0

            if verbose:
                print(f"Epoch [{epoch+1}/{epochs}] | Train CFGK Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

    if logs_dict is not None and first_batch_logs is not None and last_batch_logs is not None:
        logs_dict["c"] = c
        logs_dict["gamma"] = gamma
        logs_dict["sign_c"] = 1 if c > 0 else (-1 if c < 0 else 0)
        logs_dict["sign_gamma"] = 1 if gamma > 0 else (-1 if gamma < 0 else 0)
        logs_dict["sign_c_gamma"] = 1 if (c*gamma) > 0 else (-1 if (c*gamma) < 0 else 0)
        logs_dict["cfgk_norm_initial"] = first_batch_logs["cfgk_norm"].item()
        logs_dict["cfgk_norm_final"] = last_batch_logs["cfgk_norm"].item()
        logs_dict["loss_cfgk_penalty_initial"] = first_batch_logs["loss_cfgk_penalty"].item()
        logs_dict["loss_cfgk_penalty_final"] = last_batch_logs["loss_cfgk_penalty"].item()

    return model


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


def run_rurk_gaussian_unlearning(
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
    Implements RURK-Gaussian for CIFAR-10 + ResNet18 following Algorithm 1.
    """
    hp = kwargs.get("hp", {})
    tau = hp.get("tau", 0.03)
    v = hp.get("v", 1)
    lambda_f = hp.get("lambda_f", 0.03)
    lambda_a = hp.get("lambda_a", 0.00045)
    learning_rate = hp.get("lr", 0.01)
    momentum = hp.get("momentum", 0.90)
    weight_decay = hp.get("weight_decay", 5e-4)
    max_total_iterations = hp.get("max_total_iterations", 200)
    gradient_clip_norm = hp.get("gradient_clip_norm", 1.0)

    if v != 1:
        raise ValueError("This paper-only implementation is restricted to v = 1.")

    forget_loader = kwargs.get("forget_loader", None)
    if forget_loader is None:
        raise ValueError("forget_loader is required for RURK-Gaussian unlearning.")

    model_copy = copy.deepcopy(model)
    model_copy = model_copy.to(device)
    model_copy.train()

    # RURK does not freeze any layers
    for param in model_copy.parameters():
        param.requires_grad = True

    sgd_optimizer = optim.SGD(
        model_copy.parameters(),
        lr=learning_rate,
        momentum=momentum,
        weight_decay=weight_decay,
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        sgd_optimizer,
        T_max=max_total_iterations,
    )

    forget_iterator = iter(forget_loader)
    total_iterations = 0

    for epoch in range(epochs):
        train_loss = 0.0
        correct = 0
        total = 0

        for retain_data, retain_targets in train_loader:
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

            sgd_optimizer.zero_grad(set_to_none=True)

            retain_logits = model_copy(retain_data)
            retain_loss = criterion(retain_logits, retain_targets)

            forget_logits = model_copy(forget_data)
            forget_loss = criterion(forget_logits, forget_targets)

            adv_forget_data = generate_gaussian_perturbation(
                forget_data,
                tau=tau,
            )

            adv_forget_logits = model_copy(adv_forget_data)
            adv_forget_loss = criterion(
                adv_forget_logits,
                forget_targets,
            )

            # Loss = RLoss - lambda_f * FLoss - lambda_a * AdvFLoss
            total_loss = (
                retain_loss
                - lambda_f * forget_loss
                - lambda_a * adv_forget_loss
            )

            total_loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model_copy.parameters(),
                max_norm=gradient_clip_norm,
            )

            sgd_optimizer.step()
            scheduler.step()

            train_loss += total_loss.item() * retain_data.size(0)
            _, predicted = retain_logits.max(1)
            total += retain_targets.size(0)
            correct += predicted.eq(retain_targets).sum().item()

            total_iterations += 1

        if total_iterations >= max_total_iterations:
            break

        train_loss = train_loss / total if total > 0 else 0.0
        train_acc = 100. * correct / total if total > 0 else 0.0

        if verbose:
            model_copy.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for val_inputs, val_targets in val_loader:
                    val_inputs, val_targets = val_inputs.to(device), val_targets.to(device)
                    val_outputs = model_copy(val_inputs)
                    v_loss = criterion(val_outputs, val_targets)
                    val_loss += v_loss.item() * val_inputs.size(0)
                    _, val_predicted = val_outputs.max(1)
                    val_total += val_targets.size(0)
                    val_correct += val_predicted.eq(val_targets).sum().item()

            val_loss = val_loss / val_total if val_total > 0 else 0.0
            val_acc = 100. * val_correct / val_total if val_total > 0 else 0.0

            print(f"Epoch [{epoch+1}/{epochs}] | RURK-Gaussian Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            model_copy.train()

    return model_copy


# Registro global de protocolos
PROTOCOLS = {
    "standard": run_standard_training,
    "cfk": run_cfk_unlearning,
    "euk": run_euk_unlearning,
    "cfgk": run_cfgk_unlearning,
    "rurk": run_rurk_gaussian_unlearning
}


def get_protocol(name: str):
    """
    Obtiene la función de entrenamiento/unlearning registrada con el nombre indicado.
    """
    if name not in PROTOCOLS:
        raise ValueError(f"Protocolo '{name}' no reconocido. Opciones: {list(PROTOCOLS.keys())}")
    return PROTOCOLS[name]
