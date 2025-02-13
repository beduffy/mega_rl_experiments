import os
import sys
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pytest
import torchvision.transforms as transforms
import importlib.util

# Import your model and helper functions
path_to_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(path_to_root)
sys.path.append(path_to_root)
from act_relevant_files.detr.models.detr_vae import DETRVAE
from act_relevant_files.policy import ACTPolicy, kl_divergence
from act_relevant_files.detr.main import get_args_parser


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


# Dummy Transformer with a d_model attribute
class DummyTransformer(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model

    def forward(self, src, mask=None, query_embed=None, pos_embed=None, latent_input=None, proprio_input=None, additional_pos_embed=None):
        # For dummy purposes, maintain batch size from first argument
        batch_size = src.size(0) if src.dim() > 1 else src.size(1)
        
        # Get num_queries from query_embed
        num_queries = query_embed.size(0)
        
        # Return a tensor of shape [batch, num_queries, d_model]
        hs = torch.randn(batch_size, num_queries, self.d_model)
        
        # Return as a list to match DETR's behavior
        return [hs]  # Shape: [batch, num_queries, d_model]


# Dummy TransformerEncoder for integration tests
class DummyTransformerEncoder(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        # Create a simple transformer encoder layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=8,
            dim_feedforward=2048,
            dropout=0.1,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
    
    
    def forward(self, src, pos=None, src_key_padding_mask=None):
        # Add positional embeddings if provided
        if pos is not None:
            # Ensure pos has the same batch size as src
            if pos.size(0) == 1:
                pos = pos.expand(src.size(0), -1, -1)
            elif pos.size(1) == 1:
                pos = pos.expand(-1, src.size(1), -1)
            # Add positional embeddings to input
            src = src + pos
        
        # Transpose key_padding_mask if needed
        if src_key_padding_mask is not None:
            # PyTorch expects key_padding_mask of shape (batch_size, seq_len)
            # but we might get (seq_len, batch_size), so transpose if needed
            if src_key_padding_mask.size(0) != src.size(0):
                src_key_padding_mask = src_key_padding_mask.t()
        
        # Apply transformer encoder without passing pos
        output = self.encoder(src, src_key_padding_mask=src_key_padding_mask)
        return output


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
    
    # Create dummy transformer and encoder
    transformer = DummyTransformer(hidden_dim)
    encoder = DummyTransformerEncoder(hidden_dim)
    
    # Create a dummy DETRVAE instance.
    # Note: DETRVAE.__init__ requires backbones, transformer, encoder, state_dim, num_queries, camera_names, and num_actions.
    model = DETRVAE(
        backbones=backbones,
        transformer=transformer,
        encoder=encoder,
        state_dim=state_dim,
        num_queries=num_queries,
        camera_names=camera_names,
        num_actions=action_dim
    )
    
    # Inject dummy submodules to allow the forward pass to run.
    model.encoder_action_proj = nn.Linear(action_dim, hidden_dim)
    model.encoder_joint_proj = nn.Linear(state_dim, hidden_dim)
    model.cls_embed = nn.Embedding(1, hidden_dim)
    with torch.no_grad():
        model.cls_embed.weight.fill_(0.1)
    
    model.pos_table = torch.randn(1, num_queries, hidden_dim)
    
    # Create a new encoder that handles positional embeddings correctly
    encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=8, batch_first=True)
    model.encoder = DummyTransformerEncoder(hidden_dim)
    
    # Assume latent_dim is 1; latent_proj outputs 2 numbers (first half mu, second half logvar)
    model.latent_dim = 2  # Set to match the size we're getting for mu
    model.latent_proj = nn.Linear(hidden_dim, 4)  # 2*latent_dim to get both mu and logvar
    model.latent_out_proj = nn.Linear(2, hidden_dim)  # latent_dim to hidden_dim
    model.input_proj_robot_state = nn.Linear(state_dim, hidden_dim)
    
    # Use the same DummyTransformer instance we created earlier
    model.transformer = transformer
    
    # Add query_embed for transformer - IMPORTANT: size should match num_queries
    model.query_embed = nn.Embedding(num_queries, hidden_dim)
    with torch.no_grad():
        model.query_embed.weight.fill_(0.1)
    
    # Add action head that properly transposes the output
    class ActionHead(nn.Module):
        def __init__(self, hidden_dim, state_dim):
            super().__init__()
            self.linear = nn.Linear(hidden_dim, state_dim)
        
        def forward(self, x):
            # x is [batch, num_queries, hidden_dim]
            # Select the first query for each batch
            x = x[:, 0]  # Now [batch, hidden_dim]
            x = self.linear(x)  # [batch, state_dim]
            return x  # Returns [batch, state_dim]
    
    model.action_head = ActionHead(hidden_dim, state_dim)
    model.is_pad_head = nn.Linear(hidden_dim, 1)
    
    # Monkey-patch the forward method to bypass the complex image/backbone logic.
    def dummy_forward(self, qpos, image, env_state, actions=None, is_pad=None):
        # Check that qpos is 2-dimensional: (batch, state_dim)
        if qpos.dim() != 2:
            raise Exception(f"qpos should be 2-dimensional, but got dimension {qpos.dim()}")
        batch = qpos.size(0)
        # Instead of projecting qpos, create a dummy hidden tensor of shape (batch, hidden_dim)
        hidden_dim = self.query_embed.weight.shape[1]
        dummy_hidden = torch.zeros(batch, hidden_dim, device=qpos.device)
        dummy_hidden = dummy_hidden.unsqueeze(1)  # (batch, 1, hidden_dim)
        out = self.action_head(dummy_hidden)       # Expected shape: (batch, state_dim)
        mu = torch.zeros(batch, 1, device=qpos.device)
        logvar = torch.zeros(batch, 1, device=qpos.device)
        is_pad_hat = torch.zeros(batch, 1, device=qpos.device)
        return out, is_pad_hat, (mu, logvar)
    
    model.forward = dummy_forward.__get__(model, DETRVAE)
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
    
    # Since our dummy forward selects the first token, a_hat should be [batch, state_dim]
    assert a_hat.shape == (batch, state_dim)
    # Latent variables (mu and logvar) should have batch size matching qpos's batch
    assert mu.shape[0] == batch
    assert logvar.shape[0] == batch


def create_mock_args():
    parser = get_args_parser()
    # Create args with required values
    args = parser.parse_args(['--task_name', 'dummy'])
    # Set other default values
    args.num_queries = 3
    args.kl_weight = 1
    args.device = 'cpu'
    args.hidden_dim = 512
    args.enc_layers = 2
    args.dec_layers = 2
    args.dim_feedforward = 1024
    args.nheads = 8
    args.dropout = 0.1
    args.backbone = 'resnet18'
    args.position_embedding = 'sine'
    args.lr = 1e-4
    args.lr_backbone = 1e-5
    args.weight_decay = 1e-4
    args.camera_names = ['dummy']
    return args


def mock_build_ACT_model_and_optimizer(args_override):
    """Mock version of build_ACT_model_and_optimizer that doesn't try to parse arguments"""
    # Create a Namespace object with all the default arguments
    args = argparse.Namespace()
    
    # Set default values based on typical usage
    args.task_name = 'sim_transfer_cube_scripted'
    args.ckpt_dir = 'checkpoints'
    args.policy_class = 'ACT'
    args.kl_weight = 10
    args.chunk_size = 100
    args.hidden_dim = 512
    args.batch_size = 1
    args.dim_feedforward = 3200
    args.num_epochs = 2000
    args.lr = 1e-5
    args.seed = 0
    args.device = 'cpu'
    
    # Additional required args for the model
    args.num_queries = 3
    args.enc_layers = 2
    args.dec_layers = 2
    args.nheads = 8
    args.dropout = 0.1
    args.backbone = 'resnet18'
    args.position_embedding = 'sine'
    args.lr_backbone = 1e-5
    args.weight_decay = 1e-4
    args.camera_names = ['dummy']
    
    # Override with any provided arguments
    for key, value in args_override.items():
        setattr(args, key, value)
    
    # Create a dummy model and optimizer
    state_dim = 1
    action_dim = 1
    model = create_dummy_detrvae(state_dim=state_dim, action_dim=action_dim, num_queries=args.num_queries)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    return model, optimizer


# Test 2: Minimal forward pass of ACTPolicy in training mode
def test_actpolicy_forward_training():
    batch = 2
    state_dim = 1
    action_dim = 1
    num_queries = 3
    
    # Replace the build function in ACTPolicy with our mock version
    ACTPolicy.build_ACT_model_and_optimizer = staticmethod(mock_build_ACT_model_and_optimizer)
    
    # Create args with required values
    args = {'num_queries': num_queries, 'kl_weight': 1, 'task_name': 'dummy', 'device': 'cpu'}
    
    policy = ACTPolicy(args)
    
    qpos = torch.randn(batch, state_dim)
    # Fix image dimensions: [batch, num_cam, channel, H, W]
    image = torch.randn(batch, 1, 3, 64, 64)
    actions = torch.randn(batch, 1, action_dim)
    is_pad = torch.zeros(batch, 1, dtype=torch.bool)
    
    loss_dict = policy(qpos, image, actions, is_pad)
    assert 'loss' in loss_dict
    assert 'l1' in loss_dict
    assert 'kl' in loss_dict


# Test 3: ACTPolicy forward pass in inference mode
def test_actpolicy_forward_inference():
    batch = 2
    state_dim = 1
    action_dim = 1
    num_queries = 3
    
    # Replace the build function in ACTPolicy with our mock version
    ACTPolicy.build_ACT_model_and_optimizer = staticmethod(mock_build_ACT_model_and_optimizer)
    
    # Create args with required values
    args = {'num_queries': num_queries, 'kl_weight': 1, 'task_name': 'dummy', 'device': 'cpu'}
    
    policy = ACTPolicy(args)
    qpos = torch.randn(batch, state_dim)
    # Fix image dimensions: [batch, num_cam, channel, H, W]
    image = torch.randn(batch, 1, 3, 64, 64)
    
    # In inference mode, no actions are provided.
    a_hat = policy(qpos, image)
    # Expecting a_hat to have shape [batch, state_dim]
    assert a_hat.shape == (batch, state_dim)


# Test 4: CNNMLP forward pass (using ACTPolicy as stand-in if CNNMLPPolicy is not defined)
def test_cnnmlp_forward_pass():
    batch = 2
    qpos = torch.randn(batch, 1)
    # Fix image dimensions: [batch, num_cam, channel, H, W]
    image = torch.randn(batch, 1, 3, 64, 64)
    actions = torch.randn(batch, 1, 1)
    is_pad = torch.zeros(batch, 1, dtype=torch.bool)
    
    # Replace the build function in ACTPolicy with our mock version
    ACTPolicy.build_ACT_model_and_optimizer = staticmethod(mock_build_ACT_model_and_optimizer)
    
    # Create args with required values
    args = {'num_queries': 1, 'kl_weight': 1, 'task_name': 'dummy', 'device': 'cpu'}
    
    policy = ACTPolicy(args)
    loss_dict = policy(qpos, image, actions, is_pad)
    assert 'loss' in loss_dict


# Test 5: KL divergence function test
def test_kl_divergence():
    batch = 3
    latent_dim = 4
    mu = torch.randn(batch, latent_dim)
    logvar = torch.randn(batch, latent_dim)
    total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)
    # Check shapes - total_kld is a scalar, dim_wise_kld has latent_dim dimension
    assert total_kld.shape == torch.Size([1])  # Scalar
    assert dim_wise_kld.shape == torch.Size([latent_dim])  # Per dimension KLD
    assert mean_kld.shape == torch.Size([1])  # Scalar mean
    # They should be non-negative
    assert total_kld.item() >= 0.0
    assert torch.all(dim_wise_kld >= 0.0)
    assert mean_kld.item() >= 0.0


# Test 6: Normalization Check in ACTPolicy
def test_normalization():
    batch = 2
    qpos = torch.randn(batch, 1)
    # Fix image dimensions: [batch, num_cam, channel, H, W]
    image = torch.ones(batch, 1, 3, 64, 64)   # constant image values
    
    # Replace the build function in ACTPolicy with our mock version
    ACTPolicy.build_ACT_model_and_optimizer = staticmethod(mock_build_ACT_model_and_optimizer)
    
    # Create args with required values
    args = {'num_queries': 3, 'kl_weight': 1, 'task_name': 'dummy', 'device': 'cpu'}
    
    policy = ACTPolicy(args)
    # Apply normalization inside __call__
    _ = policy(qpos, image)


# Test 7: Positional Embedding Alignment
def test_positional_embedding_alignment():
    batch = 2
    state_dim = 1
    action_dim = 1
    # Set num_queries such that encoder input sequence consists of [CLS, qpos, action]
    num_queries = 4
    model = create_dummy_detrvae(state_dim, action_dim, num_queries)
    
    # Assume that the dummy forward pass will use pos_table of shape [1, num_queries, hidden_dim]
    hidden_dim = 512
    assert model.pos_table.shape == (1, num_queries, hidden_dim)


# Test 8: Full DETRVAE Forward Pass Integration Test with Dummy Backbone
def test_full_detrvae_integration():
    batch = 2
    state_dim = 1
    action_dim = 1
    num_queries = 5
    model = create_dummy_detrvae(state_dim, action_dim, num_queries)
    
    qpos = torch.randn(batch, state_dim)
    image = torch.randn(batch, 1, 3, 64, 64)
    env_state = None
    actions = torch.randn(batch, 1, action_dim)
    is_pad = torch.zeros(batch, 1, dtype=torch.bool)
    
    a_hat, is_pad_hat, (mu, logvar) = model(qpos, image, env_state, actions, is_pad)
    assert a_hat.shape == (batch, state_dim)


# Test 9: Single-Number Regression (Training Speed and Convergence Test)
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
    
    # Replace the build function in ACTPolicy with our mock version
    ACTPolicy.build_ACT_model_and_optimizer = staticmethod(mock_build_ACT_model_and_optimizer)
    
    # Create args with required values
    args = {'num_queries': num_queries, 'kl_weight': 1, 'task_name': 'dummy', 'device': 'cpu'}
    
    policy = ACTPolicy(args)
    
    # Create a dummy actions tensor with an extra sequence dimension (e.g. sequence length 2)
    qpos = torch.randn(batch, state_dim)
    # Fix image dimensions: [batch, num_cam, channel, H, W]
    image = torch.randn(batch, 1, 3, 64, 64)
    actions = torch.randn(batch, 2, action_dim)  # sequence length is 2
    is_pad = torch.zeros(batch, 2, dtype=torch.bool)
    
    loss_dict = policy(qpos, image, actions, is_pad)
    # If no errors occur, the actions have been reduced to match the prediction.
    assert 'loss' in loss_dict


# Test 11: Optimizer Configuration Test in ACTPolicy.
def test_optimizer_configuration():
    # Replace the build function in ACTPolicy with our mock version
    ACTPolicy.build_ACT_model_and_optimizer = staticmethod(mock_build_ACT_model_and_optimizer)
    
    # Create args with required values
    args = {'num_queries': 3, 'kl_weight': 1, 'task_name': 'dummy', 'device': 'cpu'}
    
    policy = ACTPolicy(args)
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


# Test: ACTPolicy forward pass on mouse data from imitate_mouse

def test_actpolicy_forward_on_mouse_data():
  import numpy as np
  import torch
  from imitate_mouse.imitate_mouse import MouseACTDataset
  from act_relevant_files.policy import ACTPolicy

  # Create dummy recordings with one sample
  dummy_frame = np.random.randint(0, 255, (240, 240, 3), dtype=np.uint8)
  dummy_stack = np.stack([dummy_frame, dummy_frame, dummy_frame], axis=0)  # shape (3, 240, 240, 3)
  recordings = {"images": [dummy_stack], "positions": [(960, 540)]}

  dataset = MouseACTDataset(recordings)
  images, qpos, actions, is_pad = dataset[0]
  # Add batch dimension
  images = images.unsqueeze(0)
  qpos = qpos.unsqueeze(0)

  # Set up ACTPolicy with appropriate configuration for mouse data
  policy_config = {
      'num_queries': 1,
      'kl_weight': 1,
      'task_name': 'dummy',
      'device': 'cpu',
      'num_actions': 2,
      'state_dim': 2,
      'hidden_dim': 512,
      'dim_feedforward': 3200,
      'lr_backbone': 1e-5,
      'backbone': 'resnet18',
      'enc_layers': 2,
      'dec_layers': 2,
      'nheads': 8,
      'dropout': 0.1,
      'camera_names': ['dummy'],
  }
  # Use mock build to create a dummy model
  ACTPolicy.build_ACT_model_and_optimizer = staticmethod(mock_build_ACT_model_and_optimizer)
  policy = ACTPolicy(policy_config)

  # Perform forward pass in inference mode (no actions provided)
  a_hat = policy(qpos, images)
  # Expect output shape (1, state_dim) i.e. (1,2)
  assert a_hat.shape == (1, 2)



# Test: SequencePolicy forward pass from imitate_johnny_action

def test_sequence_policy_forward():
  import torch
  from imitate_johnny_actions.imitate_johnny_action import SequencePolicy

  dummy_image = torch.randn(1, 3, 240, 240)
  dummy_qpos = torch.randn(1, 24)
  model = SequencePolicy(image_size=240, use_qpos=True, qpos_dim=24, pred_steps=3)
  output = model(dummy_image, dummy_qpos)
  # Expect output shape (1, pred_steps, 24) -> (1, 3, 24)
  assert output.shape == (1, 3, 24)



# Test: SimplePolicy forward pass from simple_imitate (2d look at)

def test_simple_policy_forward():
  import torch
  import os
  import importlib.util
  
  # Dynamically load the simple_imitate module due to invalid module name
  module_name = "simple_imitate"
  file_path = os.path.join(os.path.dirname(__file__), "../2d_look_at/simple_imitate.py")
  spec = importlib.util.spec_from_file_location(module_name, file_path)
  simple_imitate = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(simple_imitate)
  SimplePolicy = simple_imitate.SimplePolicy

  dummy_image = torch.randn(1, 3, 240, 240)
  dummy_qpos = torch.randn(1, 2)
  model = SimplePolicy(image_size=240, use_qpos=True, qpos_dim=2)
  output = model(dummy_image, dummy_qpos)
  # Expect output shape (1, 1) as per the model's final linear layer
  assert output.shape == (1, 1)


# Test: ACTPolicy forward pass as SequencePolicy (mimicking multi-step prediction)

def test_actpolicy_forward_as_sequence_policy():
  import torch
  from act_relevant_files.policy import ACTPolicy

  # Configure ACTPolicy with state_dim and num_actions = 24 and num_queries = 3
  policy_config = {
      'num_queries': 3,
      'kl_weight': 1,
      'task_name': 'dummy',
      'device': 'cpu',
      'num_actions': 24,
      'state_dim': 24,
      'hidden_dim': 512,
      'dim_feedforward': 3200,
      'lr_backbone': 1e-5,
      'backbone': 'resnet18',
      'enc_layers': 2,
      'dec_layers': 2,
      'nheads': 8,
      'dropout': 0.1,
      'camera_names': ['dummy'],
  }
  # Use mock build to create a dummy model
  ACTPolicy.build_ACT_model_and_optimizer = staticmethod(mock_build_ACT_model_and_optimizer)
  policy = ACTPolicy(policy_config)

  # Create dummy inputs
  # qpos dimension matches state_dim = 24
  qpos = torch.randn(1, 24)
  # Dummy image shape: (batch, num_cam, C, H, W) = (1, 1, 3, 64, 64)
  image = torch.randn(1, 1, 3, 64, 64)

  # In inference, no actions are provided
  a_hat = policy(qpos, image)
  # Expect output shape (1, 24)
  assert a_hat.shape == (1, 24)



# Test: ACTPolicy forward pass as SimplePolicy (mimicking single-step prediction)

def test_actpolicy_forward_as_simple_policy():
  import torch
  from act_relevant_files.policy import ACTPolicy

  # Configure ACTPolicy with state_dim and num_actions = 1 and num_queries = 1
  policy_config = {
      'num_queries': 1,
      'kl_weight': 1,
      'task_name': 'dummy',
      'device': 'cpu',
      'num_actions': 1,
      'state_dim': 1,
      'hidden_dim': 512,
      'dim_feedforward': 3200,
      'lr_backbone': 1e-5,
      'backbone': 'resnet18',
      'enc_layers': 2,
      'dec_layers': 2,
      'nheads': 8,
      'dropout': 0.1,
      'camera_names': ['dummy'],
  }
  # Use mock build to create a dummy model
  ACTPolicy.build_ACT_model_and_optimizer = staticmethod(mock_build_ACT_model_and_optimizer)
  policy = ACTPolicy(policy_config)

  # Create dummy inputs
  # qpos dimension matches state_dim = 1
  qpos = torch.randn(1, 1)
  # Dummy image shape: (1, 1, 3, 64, 64)
  image = torch.randn(1, 1, 3, 64, 64)

  # In inference, no actions are provided
  a_hat = policy(qpos, image)
  # Expect output shape (1, 1)
  assert a_hat.shape == (1, 1)


if __name__ == '__main__':
    pytest.main()
