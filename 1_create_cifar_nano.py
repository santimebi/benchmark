"""
1_create_cifar_nano.py
───────────────────────────────────────────────
Genera el dataset 'cifar_nano' extrayendo exactamente 10 imágenes por clase (100 en total)
del dataset oficial CIFAR-10 usando torchvision.

Divide el dataset en:
  - Train: 7 imágenes por clase (70 total)
  - Val: 1 imagen por clase (10 total)
  - Test: 2 imágenes por clase (20 total)

Para la clase 0, el conjunto Train se divide en:
  - Forget: 2 imágenes
  - Retain: 5 imágenes
Para las clases 1-9, las 7 imágenes de Train van a Retain.

Guarda los ficheros en datasets/cifar_nano_splits_seed_{seed}.npz.
"""

import os
from pathlib import Path
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from utils.config import DATASETS_PATH, DATA_PATH


def create_cifar_nano(output_dir=DATASETS_PATH, seeds=[0, 1, 2], download=True):
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Cargar CIFAR-10 completo (train + test) para tener suficiente variedad
    transform = transforms.Compose([
        transforms.ToTensor(), # Convierte a [0, 1] y shape (C, H, W)
    ])
    
    cifar10_dir = DATA_PATH / 'cifar-10-batches-py'
    if download:
        if cifar10_dir.exists() and any(cifar10_dir.iterdir()):
            print("CIFAR-10 ya está descargado localmente. Omitiendo descarga...")
            download = False
        else:
            print("Iniciando descarga...")

    print("Cargando CIFAR-10 desde torchvision...")
    cifar_train = torchvision.datasets.CIFAR10(root=str(DATA_PATH), train=True, download=download, transform=transform)
    cifar_test = torchvision.datasets.CIFAR10(root=str(DATA_PATH), train=False, download=download, transform=transform)
    
    # Combinar todas las muestras disponibles
    all_images = []
    all_targets = []
    
    for img, target in cifar_train:
        all_images.append(img.numpy())
        all_targets.append(target)
    for img, target in cifar_test:
        all_images.append(img.numpy())
        all_targets.append(target)
        
    X_all = np.array(all_images, dtype=np.float32)  # (60000, 3, 32, 32)
    y_all = np.array(all_targets, dtype=np.int64)   # (60000,)
    
    # Agrupar índices por clase
    class_indices = {c: np.where(y_all == c)[0] for c in range(10)}
    
    for seed in seeds:
        print(f"\n--- Generando splits cifar_nano para seed {seed} ---")
        rng = np.random.default_rng(seed)
        
        selected_indices = {}
        for c in range(10):
            # Mezclar índices de la clase c de forma reproducible con la semilla
            shuffled = rng.permutation(class_indices[c])
            # Tomar exactamente 10 imágenes
            selected_indices[c] = shuffled[:10]
            
        # Repartir los 10 elementos de cada clase:
        # Indices 0-6: Train (7)
        # Indice 7: Val (1)
        # Indices 8-9: Test (2)
        
        train_X_list, train_y_list = [], []
        val_X_list, val_y_list = [], []
        test_X_list, test_y_list = [], []
        
        forget_X_list, forget_y_list = [], []
        retain_X_list, retain_y_list = [], []
        
        for c in range(10):
            idxs = selected_indices[c]
            
            # Imágenes y etiquetas de esta clase
            imgs_c = X_all[idxs]
            targets_c = y_all[idxs]
            
            # Separar train, val, test
            train_imgs = imgs_c[0:7]
            train_targets = targets_c[0:7]
            
            val_imgs = imgs_c[7:8]
            val_targets = targets_c[7:8]
            
            test_imgs = imgs_c[8:10]
            test_targets = targets_c[8:10]
            
            val_X_list.append(val_imgs)
            val_y_list.append(val_targets)
            
            test_X_list.append(test_imgs)
            test_y_list.append(test_targets)
            
            # Si es la clase 0, dividir train_imgs en forget (2) y retain (5)
            if c == 0:
                forget_imgs = train_imgs[0:2]
                forget_targets = train_targets[0:2]
                
                retain_imgs = train_imgs[2:7]
                retain_targets = train_targets[2:7]
                
                forget_X_list.append(forget_imgs)
                forget_y_list.append(forget_targets)
                retain_X_list.append(retain_imgs)
                retain_y_list.append(retain_targets)
            else:
                retain_X_list.append(train_imgs)
                retain_y_list.append(train_targets)
                
        # Concatenar y construir arrays finales
        X_retain = np.vstack(retain_X_list)
        y_retain = np.concatenate(y_retain_list) if 'y_retain_list' in locals() else np.concatenate([y for y in retain_y_list])
        
        X_forget = np.vstack(forget_X_list)
        y_forget = np.concatenate(forget_y_list)
        
        X_val = np.vstack(val_X_list)
        y_val = np.concatenate(val_y_list)
        
        X_test = np.vstack(test_X_list)
        y_test = np.concatenate(test_y_list)
        
        # Mezclar el retain set final para que no esté ordenado por clases en el DataLoader
        retain_perm = rng.permutation(len(X_retain))
        X_retain = X_retain[retain_perm]
        y_retain = y_retain[retain_perm]
        
        # Mezclar val y test por la misma razón
        val_perm = rng.permutation(len(X_val))
        X_val = X_val[val_perm]
        y_val = y_val[val_perm]
        
        test_perm = rng.permutation(len(X_test))
        X_test = X_test[test_perm]
        y_test = y_test[test_perm]
        
        # Guardar en archivo npz
        output_file = output_dir / f"cifar_nano_splits_seed_{seed}.npz"
        np.savez(
            output_file,
            X_retain=X_retain, y_retain=y_retain,
            X_forget=X_forget, y_forget=y_forget,
            X_val=X_val, y_val=y_val,
            X_test=X_test, y_test=y_test
        )
        
        print(f"Generados splits cifar_nano para seed {seed}:")
        print(f"  Retain size: {len(X_retain)} | Shape: {X_retain.shape}")
        print(f"  Forget size: {len(X_forget)} | Shape: {X_forget.shape}")
        print(f"  Val size:    {len(X_val)} | Shape: {X_val.shape}")
        print(f"  Test size:   {len(X_test)} | Shape: {X_test.shape}")
        print(f"  Guardado en: {output_file}")


if __name__ == "__main__":
    create_cifar_nano(seeds=[0, 1, 2])
