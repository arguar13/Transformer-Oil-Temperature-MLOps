import pytest


@pytest.fixture(autouse=True)
def _dummy_aws_credentials(monkeypatch):
    """Keeps boto3 from failing on a missing region or credentials in unit tests."""
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
