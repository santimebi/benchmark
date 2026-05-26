"""
1_create_cifar10.py
───────────────────────────────────────────────
Genera los splits para el dataset CIFAR-10 completo.

Para la clase elegida para el forget (por defecto clase 7), 
un porcentaje (por defecto 40%) del conjunto de entrenamiento
pasa al conjunto 'forget', y el resto a 'retain'. Para las otras
clases, todo su conjunto de entrenamiento va a 'retain'.

Guarda los ficheros en datasets/cifar10_splits_seed_{seed}.npz.
"""

import os
from pathlib import Path
import numpy as np
import torchvision
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from utils.config import DATASETS_PATH, DATA_PATH

def create_cifar10(output_dir=DATASETS_PATH, seeds=[0, 1, 2], download=True, forget_class=7, forget_ratio=0.4, val_ratio=0.1):
    output_dir.mkdir(parents=True, exist_ok=True)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    cifar10_dir = DATA_PATH / 'cifar-10-batches-py'
    if download:
        if cifar10_dir.exists() and any(cifar10_dir.iterdir()):
            print("CIFAR-10 ya está descargado localmente. Omitiendo descarga...")
            download = False
        else:
            print("Iniciando descarga de CIFAR-10...")

    print("Cargando CIFAR-10 desde torchvision...")
    cifar_train = torchvision.datasets.CIFAR10(root=str(DATA_PATH), train=True, download=download, transform=transform)
    cifar_test = torchvision.datasets.CIFAR10(root=str(DATA_PATH), train=False, download=download, transform=transform)
    
    # Conjunto Test
    test_images = []
    test_targets = []
    for img, target in cifar_test:
        test_images.append(img.numpy())
        test_targets.append(target)
        
    X_test = np.array(test_images, dtype=np.float32)
    y_test = np.array(test_targets, dtype=np.int64)
    
    # Conjunto Train Completo
    train_images = []
    train_targets = []
    for img, target in cifar_train:
        train_images.append(img.numpy())
        train_targets.append(target)
        
    X_train_full = np.array(train_images, dtype=np.float32)
    y_train_full = np.array(train_targets, dtype=np.int64)
    
    for seed in seeds:
        print(f"\n--- Generando splits cifar10 para seed {seed} ---")
        rng = np.random.default_rng(seed)
        
        # Split train into train and val (estratificado para mantener balance de clases)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_full, y_train_full, test_size=val_ratio, random_state=seed, stratify=y_train_full
        )
        
        forget_X_list, forget_y_list = [], []
        retain_X_list, retain_y_list = [], []
        
        for c in range(10):
            class_idx = np.where(y_train == c)[0]
            X_c = X_train[class_idx]
            y_c = y_train[class_idx]
            
            # Mezclar índices de la clase para extracción aleatoria
            shuffled = rng.permutation(len(X_c))
            X_c_shuffled = X_c[shuffled]
            y_c_shuffled = y_c[shuffled]
            
            if c == forget_class:
                n_forget = int(len(X_c) * forget_ratio)
                forget_X_list.append(X_c_shuffled[:n_forget])
                forget_y_list.append(y_c_shuffled[:n_forget])
                
                retain_X_list.append(X_c_shuffled[n_forget:])
                retain_y_list.append(y_c_shuffled[n_forget:])
            else:
                retain_X_list.append(X_c_shuffled)
                retain_y_list.append(y_c_shuffled)
                
        X_retain = np.vstack(retain_X_list)
        y_retain = np.concatenate(retain_y_list)
        
        X_forget = np.vstack(forget_X_list)
        y_forget = np.concatenate(forget_y_list)
        
        # Mezclar el retain para evitar orden por clases en el DataLoader
        retain_perm = rng.permutation(len(X_retain))
        X_retain = X_retain[retain_perm]
        y_retain = y_retain[retain_perm]
        
        output_file = output_dir / f"cifar10_splits_seed_{seed}.npz"
        np.savez(
            output_file,
            X_retain=X_retain, y_retain=y_retain,
            X_forget=X_forget, y_forget=y_forget,
            X_val=X_val, y_val=y_val,
            X_test=X_test, y_test=y_test
        )
        
        print(f"Generados splits cifar10 para seed {seed}:")
        print(f"  Retain size: {len(X_retain)} | Shape: {X_retain.shape}")
        print(f"  Forget size: {len(X_forget)} | Shape: {X_forget.shape}")
        print(f"  Val size:    {len(X_val)} | Shape: {X_val.shape}")
        print(f"  Test size:   {len(X_test)} | Shape: {X_test.shape}")
        print(f"  Guardado en: {output_file}")

if __name__ == "__main__":
    create_cifar10(seeds=[0, 1, 2], forget_class=7, forget_ratio=0.4)
