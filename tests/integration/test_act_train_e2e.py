import pytest
import os
from unittest.mock import patch
import ast


@pytest.mark.integration
def test_mouse_policy_e2e(tmp_path):
    """End-to-enable test of mouse policy training and inference"""
    from imitate_mouse.imitate_mouse import train_mouse_policy
    from imitate_mouse.run_mouse_policy import run_policy_eval

    # Train minimal policy
    args_dict = {
        'policy_class': 'ACT',
        'kl_weight': 1,
        'chunk_size': 1,
        'hidden_dim': 32,
        'batch_size': 2,
        'dim_feedforward': 64,
        'num_epochs': 1,
        'lr': 1e-4,
        'seed': 0,
        'use_dummy_images': True,
        'device': 'cpu',
        'latent_dim': 32,  # Match policy config
        'enc_layers': 1,  # Match reduced config
        'dec_layers': 1,
        'nheads': 2,
        'camera_names': ['mouse_cam'],
        'ckpt_dir': str(tmp_path)
    }

    with patch('wandb.init'), patch('wandb.log'):
        train_mouse_policy(args_dict, device='cpu')

    # Test inference
    class Args:
        ckpt = 'imitate_mouse/checkpoints/mouse_act_policy_initial_epoch0.ckpt'
        dummy = True
        cpu = True

    # Capture print output instead of mocking mouse movement
    with patch('builtins.print') as mock_print:
        run_policy_eval(Args(), num_steps=5)
        # Verify policy is producing normalized outputs
        outputs = [
            ast.literal_eval(call.args[0].split(": ")[1])
            for call in mock_print.call_args_list
            if 'Policy output' in call.args[0]
        ]
        assert len(outputs) > 3, f"Insufficient policy predictions, got {len(outputs)}"
        for out in outputs:
            assert 0 <= out[0] <= 1, f"X output {out[0]} not normalized"
            assert 0 <= out[1] <= 1, f"Y output {out[1]} not normalized"


@pytest.mark.integration
def test_mouse_policy_save_load_cycle(tmp_path):
    """Test full training->saving->loading cycle for mouse policy"""
    from imitate_mouse.imitate_mouse import train_mouse_policy, MouseACTDataset
    from imitate_mouse.run_mouse_policy import run_policy_eval

    # Create dummy args dictionary matching training config
    args_dict = {
        'policy_class': 'ACT',
        'kl_weight': 10,
        'chunk_size': 10,
        'hidden_dim': 64,
        'batch_size': 2,
        'dim_feedforward': 128,
        'num_epochs': 1,
        'lr': 1e-4,
        'seed': 0,
        'use_dummy_images': True,
        'device': 'cpu',
        'enc_layers': 1,
        'dec_layers': 1,
        'nheads': 1,
        'latent_dim': 16,
        'camera_names': ['mouse_cam'],
        'ckpt_dir': str(tmp_path)
    }

    # Train and save a model
    with patch('wandb.init'), patch('wandb.log'):
        train_mouse_policy(args_dict, device='cpu')

    # Verify checkpoint creation
    ckpt_path = os.path.join('imitate_mouse/checkpoints', 'mouse_act_policy_initial_epoch0.ckpt')
    assert os.path.exists(ckpt_path), "Checkpoint not created"

    # Test loading in evaluation script
    class Args:
        ckpt = ckpt_path
        dummy = True
        cpu = True

    # Test policy execution with dummy input
    with patch('pyautogui.moveTo') as mock_move:
        run_policy_eval(Args(), num_steps=10)
        assert mock_move.called, "Policy didn't generate mouse movements"
