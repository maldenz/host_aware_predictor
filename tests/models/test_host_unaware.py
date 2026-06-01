import torch

from host_aware_predictor.models.host_unaware import HostUnawareConfig, HostUnawarePredictor


def test_host_unaware_forward_shape():
    model = HostUnawarePredictor(
        HostUnawareConfig(input_dim=16, output_dim=3, hidden_dims=(8,), dropout=0.0)
    )
    x = torch.randn(5, 16)
    y = model(x)
    assert y.shape == (5, 3)
