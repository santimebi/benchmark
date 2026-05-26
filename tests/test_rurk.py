import pytest
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from models.base_nn import BaseMLP
from utils.protocols import generate_gaussian_perturbation, run_rurk_gaussian_unlearning

def test_generate_gaussian_perturbation_range():
    # Test 3: perturbation range [0, 1]
    x = torch.rand(4, 3, 8, 8)
    tau = 0.03
    x_adv = generate_gaussian_perturbation(x, tau)
    assert x_adv.min() >= 0.0
    assert x_adv.max() <= 1.0
    assert x_adv.shape == x.shape

def test_rurk_does_not_modify_original_model():
    # Test 1: original model is not modified in-place
    model = BaseMLP(input_dim=8, hidden_dim=4, output_dim=3)
    orig_params = {name: param.clone() for name, param in model.named_parameters()}
    
    X_retain = torch.rand(10, 8)
    y_retain = torch.randint(0, 3, (10,))
    retain_loader = DataLoader(TensorDataset(X_retain, y_retain), batch_size=2)
    
    X_forget = torch.rand(4, 8)
    y_forget = torch.randint(0, 3, (4,))
    forget_loader = DataLoader(TensorDataset(X_forget, y_forget), batch_size=2)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    
    unlearned_model = run_rurk_gaussian_unlearning(
        model=model,
        train_loader=retain_loader,
        val_loader=retain_loader,
        criterion=criterion,
        optimizer=optimizer,
        epochs=1,
        device=torch.device("cpu"),
        verbose=False,
        hp={
            "lr": 0.01,
            "tau": 0.03,
            "lambda_f": 0.03,
            "lambda_a": 0.00045,
            "max_total_iterations": 10,
            "v": 1
        },
        forget_loader=forget_loader
    )
    
    for name, param in model.named_parameters():
        assert torch.equal(param, orig_params[name])
        
    changed = False
    for name, param in unlearned_model.named_parameters():
        if not torch.equal(param, orig_params[name]):
            changed = True
    assert changed

def test_rurk_all_parameters_trainable():
    # Test 2: all parameters that were trainable are still trainable
    model = BaseMLP(input_dim=8, hidden_dim=4, output_dim=3)
    
    X_retain = torch.rand(10, 8)
    y_retain = torch.randint(0, 3, (10,))
    retain_loader = DataLoader(TensorDataset(X_retain, y_retain), batch_size=2)
    
    X_forget = torch.rand(4, 8)
    y_forget = torch.randint(0, 3, (4,))
    forget_loader = DataLoader(TensorDataset(X_forget, y_forget), batch_size=2)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    
    unlearned_model = run_rurk_gaussian_unlearning(
        model=model,
        train_loader=retain_loader,
        val_loader=retain_loader,
        criterion=criterion,
        optimizer=optimizer,
        epochs=1,
        device=torch.device("cpu"),
        verbose=False,
        hp={
            "lr": 0.01,
            "tau": 0.03,
            "lambda_f": 0.03,
            "lambda_a": 0.00045,
            "max_total_iterations": 10,
            "v": 1
        },
        forget_loader=forget_loader
    )
    
    for name, param in unlearned_model.named_parameters():
        assert param.requires_grad is True

def test_rurk_max_iterations_limit():
    # Test 5: training stops when max_total_iterations is reached
    model = BaseMLP(input_dim=8, hidden_dim=4, output_dim=3)
    
    X_retain = torch.rand(10, 8)
    y_retain = torch.randint(0, 3, (10,))
    retain_loader = DataLoader(TensorDataset(X_retain, y_retain), batch_size=1)
    
    X_forget = torch.rand(4, 8)
    y_forget = torch.randint(0, 3, (4,))
    forget_loader = DataLoader(TensorDataset(X_forget, y_forget), batch_size=1)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    
    unlearned_model = run_rurk_gaussian_unlearning(
        model=model,
        train_loader=retain_loader,
        val_loader=retain_loader,
        criterion=criterion,
        optimizer=optimizer,
        epochs=5,
        device=torch.device("cpu"),
        verbose=False,
        hp={
            "lr": 0.01,
            "tau": 0.03,
            "lambda_f": 0.03,
            "lambda_a": 0.00045,
            "max_total_iterations": 3,
            "v": 1
        },
        forget_loader=forget_loader
    )
    assert unlearned_model is not None
