"""Tests for the scirag package structure."""


def test_package_imports():
    """Test that main package can be imported."""
    import scirag
    assert scirag.__version__ == "0.1.0"


def test_service_subpackage():
    """Test that service subpackage exists."""
    import scirag.service
    assert scirag.service is not None


def test_client_subpackage():
    """Test that client subpackage exists."""
    import scirag.client
    assert scirag.client is not None
