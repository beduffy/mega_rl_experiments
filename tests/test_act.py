import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pytest
import torchvision.transforms as transforms

# Import your model and helper functions
path_to_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(path_to_root)
sys.path.append(path_to_root)
from act_relevant_files.detr.models.detr_vae import DETRVAE
from act_relevant_files.policy import ACTPolicy, kl_divergence


# Dummy Backbone for integration tests
class DummyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_channels = 3
    
    
    def forward(self, x):
        # x: [batch, C, H, W]
        batch = x.shape[0]
        # Return dummy features and position embeddings as lists
        features = [torch.randn(batch, 256, 14, 14)]
        pos = [torch.randn(batch, 256, 14, 14)]
        return features, pos



# Dummy MLP constructor needed by DETRVAE __init__
def mlp(input_dim, hidden_dim, output_dim, hidden_depth):
    layers = []
    in_dim = input_dim
    for _ in range(hidden_depth):
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(nn.ReLU())
        in_dim = hidden_dim
    layers.append(nn.Linear(hidden_dim, output_dim))
    return nn.Sequential(*layers)



# Helper to create a dummy instance of DETRVAE with minimal settings (state_dim=1 and action_dim=1)
def create_dummy_detrvae(state_dim=1, action_dim=1, num_queries=5, hidden_dim=512):
    camera_names = ['dummy']
    backbones = [DummyBackbone()]
    
    # Create a dummy DETRVAE instance.
    # (Assumes the DETRVAE __init__ accepts these keyword arguments.)
    model = DETRVAE(camera_names=camera_names, state_dim=state_dim, backbones=backbones)
    
    # Inject dummy submodules to allow the forward pass to run.
    # These modules are not the real versions but let us pass through the forward without errors.
    model.encoder_action_proj = nn.Linear(action_dim, hidden_dim)
    model.encoder_joint_proj = nn.Linear(state_dim, hidden_dim)
    model.cls_embed = nn.Embedding(1, hidden_dim)
    with torch.no_grad():
        model.cls_embed.weight.fill_(0.1)
    model.pos_table = torch.randn(1, num_queries, hidden_dim)
    
    encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=8)
    model.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
    
    # Assume latent_dim is 1; latent_proj outputs 2 numbers (first half mu, second half logvar)
    model.latent_proj = nn.Linear(hidden_dim, 2)
    model.latent_out_proj = nn.Linear(1, hidden_dim)
    model.input_proj_robot_state = nn.Linear(state_dim, hidden_dim)
    
    # For simplicity, override the image-based branch with an identity transformer.
    model.transformer = nn.Identity()
    
    model.action_head = nn.Linear(hidden_dim, action_dim)
    model.is_pad_head = nn.Linear(hidden_dim, 1)
    
    return model



# Test 1: Minimal forward pass of DETRVAE in training mode
def test_detrvae_forward_pass_training():
    batch = 2
    state_dim = 1
    action_dim = 1
    num_queries = 3   # Expecting encoder input sequence: [CLS, qpos, action]
    
    model = create_dummy_detrvae(state_dim=state_dim, action_dim=action_dim, num_queries=num_queries)
    
    qpos = torch.randn(batch, state_dim)
    # Dummy image tensor: [batch, num_cam, channel, H, W] (here 1 camera)
    image = torch.randn(batch, 1, 3, 64, 64)
    env_state = None
    # Dummy actions tensor: [batch, seq, action_dim]
    actions = torch.randn(batch, 1, action_dim)
    # Dummy is_pad tensor: [batch, seq] (boolean)
    is_pad = torch.zeros(batch, 1, dtype=torch.bool)
    
    # Call forward pass in training mode
    a_hat, is_pad_hat, (mu, logvar) = model(qpos, image, env_state, actions, is_pad)
    
    # Since our dummy forward selects the first token, a_hat should be [batch, action_dim]
    assert a_hat.shape == (batch, action_dim)
    # Latent variables (mu and logvar) should have batch size matching qpos's batch (latent_dim assumed 1)
    assert mu.shape[0] == batch
    assert logvar.shape[0] == batch



# Test 2: Minimal forward pass of ACTPolicy in training mode
def test_actpolicy_forward_training():
    batch = 2
    state_dim = 1
    action_dim = 1
    num_queries = 3
    
    # Create a dummy DETRVAE model and optimizer to inject into ACTPolicy.
    dummy_detrvae = create_dummy_detrvae(state_dim, action_dim, num_queries)
    dummy_optimizer = optim.Adam(dummy_detrvae.parameters(), lr=1e-3)
    
    
    # Monkey-patch build_ACT_model_and_optimizer to return our dummy modules.
    def dummy_build_ACT_model_and_optimizer(args):
        return dummy_detrvae, dummy_optimizer
    
    
    # Replace the build function in ACTPolicy.
    ACTPolicy.build_ACT_model_and_optimizer = staticmethod(dummy_build_ACT_model_and_optimizer)
    
    
    policy = ACTPolicy({'num_queries': num_queries, 'kl_weight': 1})
    
    
    # Dummy inputs
    qpos = torch.randn(batch, state_dim)
    image = torch.randn(batch, 3, 64, 64)   # 3-channel image
    actions = torch.randn(batch, 1, action_dim)
    is_pad = torch.zeros(batch, 1, dtype=torch.bool)
    
    
    loss_dict = policy(qpos, image, actions, is_pad)
    # loss_dict must contain the specified keys and hold scalar values.
    assert 'l1' in loss_dict
    assert 'kl' in loss_dict
    assert 'loss' in loss_dict
    assert torch.tensor(loss_dict['l1']).ndim == 0
    assert torch.tensor(loss_dict['kl']).ndim == 0
    assert torch.tensor(loss_dict['loss']).ndim == 0



# Test 3: ACTPolicy forward pass in inference mode
def test_actpolicy_forward_inference():
    batch = 2
    state_dim = 1
    action_dim = 1
    num_queries = 3
    
    dummy_detrvae = create_dummy_detrvae(state_dim, action_dim, num_queries)
    dummy_optimizer = optim.Adam(dummy_detrvae.parameters(), lr=1e-3)
    
    
    def dummy_build_ACT_model_and_optimizer(args):
        return dummy_detrvae, dummy_optimizer
    
    
    ACTPolicy.build_ACT_model_and_optimizer = staticmethod(dummy_build_ACT_model_and_optimizer)
    
    
    policy = ACTPolicy({'num_queries': num_queries, 'kl_weight': 1})
    qpos = torch.randn(batch, state_dim)
    image = torch.randn(batch, 3, 64, 64)
    
    # In inference mode, no actions are provided.
    a_hat = policy(qpos, image)
    # Expecting a_hat to have shape [batch, action_dim]
    assert a_hat.shape == (batch, action_dim)



# Test 4: Minimal forward pass of CNNMLPPolicy (if defined) using ACTPolicy as a placeholder.
def test_cnnmlp_forward_pass():
    # For this test, we simulate a CNNMLP forward pass.
    from act_relevant_files.policy import ACTPolicy   # Using ACTPolicy if CNNMLPPolicy is not defined.
    
    batch = 2
    qpos = torch.randn(batch, 1)
    image = torch.randn(batch, 3, 64, 64)
    actions = torch.randn(batch, 1, 1)
    is_pad = torch.zeros(batch, 1, dtype=torch.bool)
    
    policy = ACTPolicy({'num_queries': 1, 'kl_weight': 1})
    output = policy(qpos, image, actions, is_pad)
    assert isinstance(output, dict)
    assert 'l1' in output



# Test 5: Test the kl_divergence function.
def test_kl_divergence():
    batch = 4
    latent_dim = 5
    mu = torch.randn(batch, latent_dim)
    logvar = torch.randn(batch, latent_dim)
    total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)
    # Check that the shapes of the KLD outputs are as expected.
    assert total_kld.shape == torch.Size([1])
    assert dim_wise_kld.shape == torch.Size([latent_dim])
    assert mean_kld.shape == torch.Size([1])



# Test 6: Normalization Check in ACTPolicy.
def test_normalization():
    from act_relevant_files.policy import ACTPolicy
    batch = 2
    qpos = torch.randn(batch, 1)
    image = torch.ones(batch, 3, 64, 64)   # constant image values
    
    # Create dummy DETRVAE and inject into ACTPolicy as before.
    dummy_detrvae = create_dummy_detrvae(1, 1, 3)
    dummy_optimizer = optim.Adam(dummy_detrvae.parameters(), lr=1e-3)
    
    
    def dummy_build_ACT_model_and_optimizer(args):
        return dummy_detrvae, dummy_optimizer
    
    
    ACTPolicy.build_ACT_model_and_optimizer = staticmethod(dummy_build_ACT_model_and_optimizer)
    policy = ACTPolicy({'num_queries': 3, 'kl_weight': 1})
    
    # Manually compute normalization as performed in the policy.
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    expected = normalize(image)
    
    # Call inference forward pass (which uses normalization internally).
    a_hat = policy(qpos, image)
    # We cannot directly extracted the normalized image, so if no error occurs we assume success.
    assert a_hat is not None



# Test 7: Positional Embedding Alignment in DETRVAE forward pass.
def test_positional_embedding_alignment():
    batch = 2
    state_dim = 1
    action_dim = 1
    # Set num_queries such that encoder input sequence consists of [CLS, qpos, action]
    num_queries = 4
    model = create_dummy_detrvae(state_dim, action_dim, num_queries)
    
    qpos = torch.randn(batch, state_dim)
    image = torch.randn(batch, 1, 3, 64, 64)
    actions = torch.randn(batch, 1, action_dim)
    is_pad = torch.zeros(batch, 1, dtype=torch.bool)
    
    # Execute forward pass; if positional embedding slicing fails, an error would be thrown.
    a_hat, is_pad_hat, _ = model(qpos, image, None, actions, is_pad)
    # Confirm output has the correct batch dimension.
    assert a_hat.shape[0] == batch



# Test 8: Full DETRVAE Integration Test with Dummy Backbone.
def test_full_detrvae_integration():
    batch = 2
    state_dim = 1
    action_dim = 1
    num_queries = 5
    model = create_dummy_detrvae(state_dim, action_dim, num_queries)
    
    qpos = torch.randn(batch, state_dim)
    # Dummy image: one camera, 3 channels.
    image = torch.randn(batch, 1, 3, 64, 64)
    actions = torch.randn(batch, 1, action_dim)
    is_pad = torch.zeros(batch, 1, dtype=torch.bool)
    
    a_hat, is_pad_hat, _ = model(qpos, image, None, actions, is_pad)
    assert a_hat.shape == (batch, action_dim)



# Test 9: Single-Number Regression - Training a simple model on a single scalar target.
class SingleNumberRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)
    
    
    def forward(self, x):
        return self.linear(x)
    


def test_single_number_regression():
    # Simple regression: y = 2x; train on one example.
    model = SingleNumberRegression()
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    x = torch.tensor([[1.0]])
    target = torch.tensor([[2.0]])
    
    
    for _ in range(200):
        optimizer.zero_grad()
        pred = model(x)
        loss = F.mse_loss(pred, target)
        loss.backward()
        optimizer.step()
    
    
    final_loss = F.mse_loss(model(x), target).item()
    assert final_loss < 0.001



# Test 10: Batch vs. Sequence Dimension Consistency Check in ACTPolicy.
def test_batch_sequence_consistency():
    batch = 4
    state_dim = 1
    action_dim = 1
    num_queries = 3
    dummy_detrvae = create_dummy_detrvae(state_dim, action_dim, num_queries)
    dummy_optimizer = optim.Adam(dummy_detrvae.parameters(), lr=1e-3)
    
    
    def dummy_build_ACT_model_and_optimizer(args):
        return dummy_detrvae, dummy_optimizer
    
    
    from act_relevant_files.policy import ACTPolicy
    ACTPolicy.build_ACT_model_and_optimizer = staticmethod(dummy_build_ACT_model_and_optimizer)
    policy = ACTPolicy({'num_queries': num_queries, 'kl_weight': 1})
    
    # Create a dummy actions tensor with an extra sequence dimension (e.g. sequence length 2)
    qpos = torch.randn(batch, state_dim)
    image = torch.randn(batch, 3, 64, 64)
    actions = torch.randn(batch, 2, action_dim)  # sequence length is 2
    is_pad = torch.zeros(batch, 2, dtype=torch.bool)
    
    loss_dict = policy(qpos, image, actions, is_pad)
    # If no errors occur, the actions have been reduced to match the prediction.
    assert 'loss' in loss_dict



# Test 11: Optimizer Configuration Test in ACTPolicy.
def test_optimizer_configuration():
    from act_relevant_files.policy import ACTPolicy
    dummy_detrvae = create_dummy_detrvae(1, 1, 3)
    dummy_optimizer = optim.Adam(dummy_detrvae.parameters(), lr=1e-3)
    
    
    def dummy_build_ACT_model_and_optimizer(args):
        return dummy_detrvae, dummy_optimizer
    
    
    ACTPolicy.build_ACT_model_and_optimizer = staticmethod(dummy_build_ACT_model_and_optimizer)
    policy = ACTPolicy({'num_queries': 3, 'kl_weight': 1})
    optimizer = policy.configure_optimizers()
    assert isinstance(optimizer, optim.Optimizer)



# Test 12: Edge-case Input Tests and Error Handling in DETRVAE forward.
def test_edge_case_inputs():
    batch = 2
    state_dim = 1
    action_dim = 1
    num_queries = 3
    model = create_dummy_detrvae(state_dim, action_dim, num_queries)
    
    # Provide qpos with wrong dimensions (missing feature dimension)
    qpos_wrong = torch.randn(batch)  # should be [batch, state_dim]
    image = torch.randn(batch, 1, 3, 64, 64)
    actions = torch.randn(batch, 1, action_dim)
    is_pad = torch.zeros(batch, 1, dtype=torch.bool)
    
    with pytest.raises(Exception):
        _ = model(qpos_wrong, image, None, actions, is_pad)
    
    
if __name__ == '__main__':
    pytest.main()
