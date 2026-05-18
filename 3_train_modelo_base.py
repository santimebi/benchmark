import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from pathlib import Path
from utils.config import DATASETS_PATH
from models.base_nn import BaseMLP

def train_base_model(seed: int = 0, epochs: int = 100, batch_size: int = 32, lr: float = 1e-3, verbose: bool = True):
    # 1. Cargar datos del split específico
    data_path = DATASETS_PATH / f"spiral_splits_seed_{seed}.npz"
    if not data_path.exists():
        raise FileNotFoundError(f"No se encontró el dataset para la seed {seed}. Corre 2_split_dataset.py primero.")
    
    data = np.load(data_path)
    
    # Para el modelo base "original", entrenamos con TODO el set de entrenamiento (Retain + Forget)
    X_train = np.vstack([data['X_retain'], data['X_forget']])
    y_train = np.concatenate([data['y_retain'], data['y_forget']])
    
    X_val = data['X_val']
    y_val = data['y_val']
    
    # Convertir a tensores
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.long)
    
    # Crear DataLoaders
    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # 2. Instanciar el modelo
    # El dataset espiral tiene 2 características (x1, x2) y 3 clases (0, 1, 2)
    model = BaseMLP(input_dim=2, hidden_dim=16, output_dim=3)
    
    # 3. Configurar optimizador y función de pérdida
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    if verbose:
        print(f"--- Iniciando entrenamiento del modelo original (Seed: {seed}) ---")
    
    # 4. Bucle de Entrenamiento
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, targets in train_loader:
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
            with torch.no_grad():
                val_outputs = model(X_val_t)
                val_loss = criterion(val_outputs, y_val_t).item()
                _, val_predicted = val_outputs.max(1)
                val_acc = 100. * val_predicted.eq(y_val_t).sum().item() / len(y_val_t)
                
            if verbose:
                print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
                
    # 5. Guardar el modelo
    weights_dir = Path("models/weights")
    weights_dir.mkdir(parents=True, exist_ok=True)
    model_path = weights_dir / f"base_model_seed_{seed}.pth"
    torch.save(model.state_dict(), model_path)
    
    if verbose:
        print(f"Modelo guardado exitosamente en: {model_path}")
        
    return model

if __name__ == "__main__":
    # Podemos entrenar los modelos base para todas las seeds generadas
    seeds_to_train = [0, 1, 2]
    
    for s in seeds_to_train:
        train_base_model(seed=s, epochs=150, batch_size=32, lr=1e-3, verbose=True)
        print("-" * 50)
