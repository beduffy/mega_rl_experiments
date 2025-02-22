import torch
import torch.nn as nn

from act_relevant_files.detr.models.detr_vae import DETRVAE


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
        num_actions=action_dim,
        hidden_dim=hidden_dim,
        latent_dim=2
    )

    # Inject dummy submodules to allow the forward pass to run.
    model.encoder_action_proj = nn.Linear(action_dim, hidden_dim)
    model.encoder_joint_proj = nn.Linear(state_dim, hidden_dim)
    model.cls_embed = nn.Embedding(1, hidden_dim)
    with torch.no_grad():
        model.cls_embed.weight.fill_(0.1)

    # Instead of assigning pos_table directly, directly assign the new tensor and update the _buffers dictionary
    new_tensor = torch.randn(1, num_queries, hidden_dim)
    model.pos_table = new_tensor
    model._buffers["pos_table"] = new_tensor

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
