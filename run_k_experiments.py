import subprocess
import sys
from pathlib import Path

def main():
    protocols = ["cfk", "euk", "cfgk"]
    k_values = [17, 13, 9, 5]
    epochs = 20
    seeds = "0,1,2"
    dataset = "cifar_nano"
    model_arch = "models.resnet.ResNet18"
    
    python_exec = sys.executable

    for protocol in protocols:
        for k in k_values:
            model_name = f"{protocol}_k{k}"
            print(f"\n========================================================")
            
            # Buscar el archivo de mejores hiperparámetros específico del protocolo
            hp_file = f"models/best_{protocol}_hp.json"
            if not Path(hp_file).exists():
                hp_file = "models/best_hp.json"
                
            cmd_train = [
                python_exec, "3_train_model.py",
                "--model_arch", model_arch,
                "--protocol", protocol,
                "--train_splits", "retain",
                "--model_name", model_name,
                "--dataset", dataset,
                "--pretrained_weights", "models/weights/base_model_seed_{seed}.pth",
                "--hp_file", hp_file,
                "--epochs", str(epochs),
                "--k", str(k),
                "--seeds", seeds
            ]
            print(f"EJECUTANDO ENTRENAMIENTO: {protocol} con k={k}")
            print(f"Comando: {' '.join(cmd_train)}")
            subprocess.run(cmd_train, check=True)
            
            # Ejecutar métricas automáticamente para este experimento
            cmd_metrics = [
                python_exec, "4_metricas.py",
                "--unlearned_name", model_name,
                "--model_arch", model_arch,
                "--dataset", dataset,
                "--seeds", seeds,
                "--hp_file", "models/best_hp.json"
            ]
            print(f"EJECUTANDO EVALUACIÓN DE MÉTRICAS: {model_name}")
            print(f"Comando: {' '.join(cmd_metrics)}")
            subprocess.run(cmd_metrics, check=True)

    print("\n========================================================")
    print(" ¡TODAS LAS PRUEBAS DE K HAN SIDO COMPLETADAS CON ÉXITO! ")
    print("========================================================\n")

if __name__ == "__main__":
    main()
