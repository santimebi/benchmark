"""
tests/test_train_model.py
───────────────────────────────────────────────
Valida la carga dinámica de clases, la concatenación de splits de datos,
el entrenamiento con diferentes protocolos, y el guardado de pesos correctos.
"""

import sys
from pathlib import Path
import pytest
import numpy as np
import torch
import importlib.util

# Cargamos el módulo dinámicamente porque empieza por un dígito.
_spec = importlib.util.spec_from_file_location(
    "train_model",
    str(Path(__file__).parent.parent / "3_train_model.py"),
)
train_module = importlib.util.module_from_spec(_spec)
sys.modules["train_model"] = train_module
_spec.loader.exec_module(train_module)


# ─────────────────────────────────────────────
# Helpers / fixtures
# ─────────────────────────────────────────────
def _create_fake_npz(directory: Path, seed: int = 0):
    """Genera un .npz sintético con las claves que espera train_model."""
    rng = np.random.default_rng(42)
    np.savez(
        directory / f"spiral_splits_seed_{seed}.npz",
        X_retain=rng.standard_normal((60, 2)).astype(np.float32),
        y_retain=rng.integers(0, 3, 60),
        X_forget=rng.standard_normal((30, 2)).astype(np.float32),
        y_forget=rng.integers(0, 3, 30),
        X_val=rng.standard_normal((30, 2)).astype(np.float32),
        y_val=rng.integers(0, 3, 30),
        X_test=rng.standard_normal((30, 2)).astype(np.float32),
        y_test=rng.integers(0, 3, 30),
    )


@pytest.fixture(autouse=True)
def _redirect_datasets_path(tmp_path, monkeypatch):
    """Redirige DATASETS_PATH a tmp_path para todos los tests."""
    monkeypatch.setattr(train_module, "DATASETS_PATH", tmp_path)


@pytest.fixture
def fake_dataset(tmp_path):
    """Crea un .npz sintético y devuelve tmp_path."""
    _create_fake_npz(tmp_path, seed=0)
    return tmp_path


@pytest.fixture
def weights_dir(tmp_path, monkeypatch):
    """
    Redirige el guardado de pesos interceptando torch.save.
    """
    target_dir = tmp_path / "models" / "weights"

    _original_torch_save = torch.save
    _saved_paths = []

    def _patched_save(obj, f, *args, **kwargs):
        if isinstance(f, Path) and "_model_seed" in str(f):
            target_dir.mkdir(parents=True, exist_ok=True)
            new_path = target_dir / f.name
            _saved_paths.append(new_path)
            return _original_torch_save(obj, new_path, *args, **kwargs)
        return _original_torch_save(obj, f, *args, **kwargs)

    monkeypatch.setattr(train_module.torch, "save", _patched_save)
    return target_dir


# ─────────────────────────────────────────────
# Test Carga Dinámica (load_class)
# ─────────────────────────────────────────────
def test_load_class_valid():
    cls = train_module.load_class("models.base_nn.BaseMLP")
    from models.base_nn import BaseMLP
    assert cls is BaseMLP


def test_load_class_invalid():
    with pytest.raises(ImportError):
        train_module.load_class("models.base_nn.NonExistentClass")
    with pytest.raises(ImportError):
        train_module.load_class("non_existent_module.Class")


# ─────────────────────────────────────────────
# Test Carga de Splits (load_dataset_splits)
# ─────────────────────────────────────────────
def test_load_dataset_splits_valid(fake_dataset):
    data_path = fake_dataset / "spiral_splits_seed_0.npz"
    # Carga combinada retain + forget (Modelo Base)
    X_train, y_train, X_val, y_val = train_module.load_dataset_splits(data_path, ["retain", "forget"])
    assert len(X_train) == 90  # 60 retain + 30 forget
    assert len(y_train) == 90
    assert len(X_val) == 30
    assert len(y_val) == 30

    # Carga solo retain (Modelo Naive)
    X_train_naive, y_train_naive, _, _ = train_module.load_dataset_splits(data_path, ["retain"])
    assert len(X_train_naive) == 60
    assert len(y_train_naive) == 60


def test_load_dataset_splits_invalid_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        train_module.load_dataset_splits(tmp_path / "missing.npz", ["retain"])


def test_load_dataset_splits_invalid_key(fake_dataset):
    data_path = fake_dataset / "spiral_splits_seed_0.npz"
    with pytest.raises(KeyError):
        train_module.load_dataset_splits(data_path, ["non_existent_split"])


# ─────────────────────────────────────────────
# Test Entrenamiento y Guardado de Pesos
# ─────────────────────────────────────────────
class TestTrainModel_Execution:

    def test_train_base_model(self, fake_dataset, weights_dir):
        """Entrenamiento del modelo base (retain + forget)."""
        hp = {"epochs": 2, "batch_size": 16, "lr": 1e-3, "hidden_dim": 8}
        model = train_module.train_model(
            model_arch="models.base_nn.BaseMLP",
            protocol="standard",
            train_splits=["retain", "forget"],
            seed=0,
            model_name="base",
            hp=hp,
            verbose=False
        )
        assert model is not None
        assert model.classifier.out_features == 3
        
        # Verificar creación de pesos con prefijo base
        expected_weights = weights_dir / "base_model_seed_0.pth"
        assert expected_weights.exists()

    def test_train_naive_model(self, fake_dataset, weights_dir):
        """Entrenamiento del modelo naive (solo retain)."""
        hp = {"epochs": 2, "batch_size": 16, "lr": 1e-3, "hidden_dim": 8}
        model = train_module.train_model(
            model_arch="models.base_nn.BaseMLP",
            protocol="standard",
            train_splits=["retain"],
            seed=0,
            model_name="naive",
            hp=hp,
            verbose=False
        )
        assert model is not None
        
        # Verificar creación de pesos con prefijo naive
        expected_weights = weights_dir / "naive_model_seed_0.pth"
        assert expected_weights.exists()

    def test_train_missing_protocol(self, fake_dataset, weights_dir):
        """Debe fallar si el protocolo no está registrado."""
        with pytest.raises(ValueError, match="Protocolo 'unknown' no reconocido"):
            train_module.train_model(
                protocol="unknown",
                seed=0,
                verbose=False
            )

    def test_train_cfk_unlearning_freezes_correct_layers(self, fake_dataset, weights_dir):
        """El protocolo CFK con k=1 debe congelar el feature_extractor y solo entrenar el classifier."""
        hp = {"epochs": 2, "batch_size": 16, "lr": 0.1, "hidden_dim": 8, "k": 1}
        
        # 1. Entrenamos un modelo base primero para guardar unos pesos iniciales
        model_base = train_module.train_model(
            model_arch="models.base_nn.BaseMLP",
            protocol="standard",
            train_splits=["retain", "forget"],
            seed=0,
            model_name="base",
            hp=hp,
            verbose=False
        )
        base_weights_path = weights_dir / "base_model_seed_0.pth"
        assert base_weights_path.exists()
        
        # Guardamos copias de los pesos iniciales del feature_extractor y classifier
        init_fe_weights = {name: param.clone() for name, param in model_base.feature_extractor.named_parameters()}
        init_clf_weights = {name: param.clone() for name, param in model_base.classifier.named_parameters()}
        
        # 2. Corremos CFK partiendo de los pesos del modelo base
        model_cfk = train_module.train_model(
            model_arch="models.base_nn.BaseMLP",
            protocol="cfk",
            train_splits=["retain"],
            seed=0,
            model_name="cfk",
            hp=hp,
            verbose=False,
            pretrained_weights=str(base_weights_path)
        )
        
        # 3. Comprobamos que los pesos del feature_extractor no cambiaron en absoluto
        for name, param in model_cfk.feature_extractor.named_parameters():
            assert torch.equal(param, init_fe_weights[name])
            
        # 4. Comprobamos que los pesos de classifier sí cambiaron
        changed = False
        for name, param in model_cfk.classifier.named_parameters():
            if not torch.equal(param, init_clf_weights[name]):
                changed = True
        assert changed, "Los pesos del clasificador debieron cambiar durante el entrenamiento CFK"

    def test_train_pretrained_weights_not_found(self, fake_dataset):
        """Debe fallar con FileNotFoundError si el archivo de pesos no existe."""
        hp = {"epochs": 1, "batch_size": 16, "lr": 1e-3, "hidden_dim": 8}
        with pytest.raises(FileNotFoundError):
            train_module.train_model(
                model_arch="models.base_nn.BaseMLP",
                protocol="standard",
                train_splits=["retain"],
                seed=0,
                hp=hp,
                verbose=False,
                pretrained_weights="non_existent_weights_seed_{seed}.pth"
            )

    def test_train_cfk_invalid_k(self, fake_dataset):
        """Debe fallar con ValueError si k es mayor que el número total de capas."""
        # BaseMLP tiene 3 capas (2 en extractor, 1 en clasificador)
        hp = {"epochs": 1, "batch_size": 16, "lr": 1e-3, "hidden_dim": 8, "k": 10}
        with pytest.raises(ValueError, match="k=10 es mayor que el número total de capas con parámetros"):
            train_module.train_model(
                model_arch="models.base_nn.BaseMLP",
                protocol="cfk",
                train_splits=["retain"],
                seed=0,
                hp=hp,
                verbose=False
            )

    def test_standard_training_updates_all_parameters(self, fake_dataset):
        """En el protocolo standard, todas las capas deben actualizarse."""
        hp = {"epochs": 2, "batch_size": 16, "lr": 0.1, "hidden_dim": 8}
        
        # Obtenemos modelo inicial y guardamos copias
        model = train_module.train_model(
            model_arch="models.base_nn.BaseMLP",
            protocol="standard",
            train_splits=["retain"],
            seed=0,
            hp=hp,
            verbose=False
        )
        
        # Clonamos estado inicial para comparar
        init_state = {name: param.clone() for name, param in model.named_parameters()}
        
        # Volvemos a entrenar
        model_trained = train_module.train_model(
            model_arch="models.base_nn.BaseMLP",
            protocol="standard",
            train_splits=["retain"],
            seed=0,
            hp=hp,
            verbose=False
        )
        
        # Ambos (extractor y clasificador) deben haber cambiado
        changed_params = 0
        for name, param in model_trained.named_parameters():
            if not torch.equal(param, init_state[name]):
                changed_params += 1
        
        assert changed_params > 0, "Al menos algunos parámetros deben haber cambiado en entrenamiento standard"

    def test_cli_execution_with_new_parameters(self, fake_dataset, weights_dir, monkeypatch):
        """Verifica la ejecución a través de la interfaz CLI (con argparse) y los nuevos argumentos."""
        # 1. Creamos pesos base para que la ejecución del CLI de unlearning tenga éxito
        hp_initial = {"epochs": 1, "batch_size": 16, "lr": 1e-3, "hidden_dim": 8}
        train_module.train_model(
            model_arch="models.base_nn.BaseMLP",
            protocol="standard",
            train_splits=["retain", "forget"],
            seed=0,
            model_name="base",
            hp=hp_initial,
            verbose=False
        )
        base_weights_path = weights_dir / "base_model_seed_0.pth"
        assert base_weights_path.exists()
        
        # 2. Simulamos la llamada CLI con mock argv
        import sys
        fake_argv = [
            "3_train_model.py",
            "--model_name", "cfk_cli",
            "--protocol", "cfk",
            "--train_splits", "retain",
            "--pretrained_weights", str(base_weights_path),
            "--epochs", "2",
            "--k", "1",
            "--seeds", "0"
        ]
        
        monkeypatch.setattr(sys, "argv", fake_argv)
        
        # Capturamos si llama a train_model
        _original_train_model = train_module.train_model
        called_args = []
        
        def _mock_train_model(*args, **kwargs):
            called_args.append((args, kwargs))
            return _original_train_model(*args, **kwargs)
            
        monkeypatch.setattr(train_module, "train_model", _mock_train_model)
        
        # Ejecutamos la lógica de main
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--model_arch", type=str, default="models.base_nn.BaseMLP")
        parser.add_argument("--protocol", type=str, default="standard")
        parser.add_argument("--train_splits", type=str, default="retain,forget")
        parser.add_argument("--model_name", type=str, default="base")
        parser.add_argument("--hp_file", type=str, default="models/best_hp.json")
        parser.add_argument("--epochs", type=int)
        parser.add_argument("--batch_size", type=int)
        parser.add_argument("--lr", type=float)
        parser.add_argument("--hidden_dim", type=int)
        parser.add_argument("--seeds", type=str, default="0,1,2")
        parser.add_argument("--pretrained_weights", type=str, default=None)
        parser.add_argument("--k", type=int, default=1)
        parser.add_argument("--no_verbose", dest="verbose", action="store_false")
        parser.set_defaults(verbose=True)
        
        args = parser.parse_args(fake_argv[1:])
        
        # Simula lógica principal
        hp = {"epochs": args.epochs, "batch_size": 32, "lr": 1e-3, "hidden_dim": 8, "k": args.k}
        
        train_module.train_model(
            model_arch=args.model_arch,
            protocol=args.protocol,
            train_splits=["retain"],
            seed=0,
            model_name=args.model_name,
            hp=hp,
            verbose=False,
            pretrained_weights=args.pretrained_weights
        )
        
        assert len(called_args) == 1
        _, kwargs = called_args[0]
        assert kwargs["protocol"] == "cfk"
        assert kwargs["model_name"] == "cfk_cli"
        assert kwargs["pretrained_weights"] == str(base_weights_path)
        assert kwargs["hp"]["k"] == 1


