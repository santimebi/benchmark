import sys
from unittest.mock import MagicMock, patch
import pytest

from utils.wandb_helper import init_wandb, log_wandb


def test_init_wandb_disabled():
    """Verify that disabled mode doesn't initialize wandb."""
    with patch("utils.wandb_helper.wandb") as mock_wandb:
        run = init_wandb(mode="disabled")
        assert run is None
        mock_wandb.init.assert_not_called()


def test_init_wandb_enabled():
    """Verify that online/offline mode initializes wandb with correct params."""
    with patch("utils.wandb_helper.wandb") as mock_wandb:
        mock_wandb.init.return_value = "mock_run"
        
        run = init_wandb(
            mode="offline",
            project="test_proj",
            name="test_run",
            group="test_group",
            job_type="test_job",
            config={"lr": 0.01}
        )
        
        assert run == "mock_run"
        mock_wandb.init.assert_called_once_with(
            project="test_proj",
            name="test_run",
            group="test_group",
            job_type="test_job",
            config={"lr": 0.01},
            mode="offline"
        )


def test_log_wandb_active_run():
    """Verify that log_wandb logs metrics when run is active."""
    with patch("utils.wandb_helper.wandb") as mock_wandb:
        # Mock active run
        mock_wandb.run = MagicMock()
        
        log_wandb({"loss": 0.5}, step=10)
        
        mock_wandb.log.assert_called_once_with({"loss": 0.5}, step=10)


def test_log_wandb_no_active_run():
    """Verify that log_wandb does nothing when no run is active."""
    with patch("utils.wandb_helper.wandb") as mock_wandb:
        # No active run
        mock_wandb.run = None
        
        log_wandb({"loss": 0.5})
        
        mock_wandb.log.assert_not_called()
