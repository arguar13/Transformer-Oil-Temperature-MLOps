import torch

from src.model_architecture import DLinear


def test_dlinear_forward_pass_returns_expected_shape():
    batch_size, seq_len, n_features = 4, 48, 10
    model = DLinear(seq_len=seq_len, n_features=n_features)
    x = torch.randn(batch_size, seq_len, n_features)

    output = model(x)

    assert output.shape == (batch_size, 1)  # pred_len=1 (default): one step ahead


def test_dlinear_forward_pass_supports_multi_step_horizon():
    batch_size, seq_len, n_features, pred_len = 4, 48, 10, 48
    model = DLinear(seq_len=seq_len, n_features=n_features, pred_len=pred_len)
    x = torch.randn(batch_size, seq_len, n_features)

    output = model(x)

    assert output.shape == (batch_size, pred_len)


def test_dlinear_is_deterministic_in_eval_mode():
    model = DLinear(seq_len=48, n_features=10)
    model.eval()
    x = torch.randn(2, 48, 10)

    with torch.no_grad():
        first_pass = model(x)
        second_pass = model(x)

    torch.testing.assert_close(first_pass, second_pass)
