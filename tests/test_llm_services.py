"""Tests for the llm_services module."""

import os
from unittest.mock import MagicMock, patch

import pytest

from scirag.service.llm_services import OllamaService, get_llm_service


class TestOllamaService:
    """Tests for OllamaService class."""

    def test_init(self):
        """Test OllamaService initialization."""
        service = OllamaService(host="http://test:11434", model="test-model")

        assert service.host == "http://test:11434"
        assert service.model == "test-model"
        assert service.client is not None

    @pytest.mark.asyncio
    async def test_generate_response_success(self):
        """Test successful response generation."""
        service = OllamaService(host="http://test:11434", model="test-model")

        # Mock the client.chat method
        mock_response = {"message": {"content": "Hello, world!"}}
        service.client.chat = MagicMock(return_value=mock_response)

        messages = [{"role": "user", "content": "Say hello"}]
        response = await service.generate_response(messages)

        assert response == "Hello, world!"
        service.client.chat.assert_called_once_with(
            model="test-model", messages=messages
        )

    @pytest.mark.asyncio
    async def test_generate_response_with_multiple_messages(self):
        """Test response generation with conversation history."""
        service = OllamaService(host="http://test:11434", model="test-model")

        # Mock the client.chat method
        mock_response = {"message": {"content": "I'm doing well, thanks!"}}
        service.client.chat = MagicMock(return_value=mock_response)

        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"},
        ]
        response = await service.generate_response(messages)

        assert response == "I'm doing well, thanks!"
        service.client.chat.assert_called_once_with(
            model="test-model", messages=messages
        )

    @pytest.mark.asyncio
    async def test_generate_response_with_system_message(self):
        """Test response generation with system message."""
        service = OllamaService(host="http://test:11434", model="test-model")

        # Mock the client.chat method
        mock_response = {
            "message": {"content": "I am a helpful assistant focused on science."}
        }
        service.client.chat = MagicMock(return_value=mock_response)

        messages = [
            {"role": "system", "content": "You are a scientific assistant."},
            {"role": "user", "content": "Who are you?"},
        ]
        response = await service.generate_response(messages)

        assert response == "I am a helpful assistant focused on science."
        service.client.chat.assert_called_once_with(
            model="test-model", messages=messages
        )


class TestGetLLMService:
    """Tests for get_llm_service factory function."""

    @patch("scirag.service.llm_services.OllamaService")
    def test_creates_ollama_service_with_defaults(self, mock_ollama_class):
        """Test creating Ollama service with default configuration."""
        mock_service = MagicMock()
        mock_ollama_class.return_value = mock_service

        with patch.dict(
            os.environ,
            {"OLLAMA_HOST": "http://env-host:11434", "OLLAMA_MODEL": "env-model"},
        ):
            service = get_llm_service()

            mock_ollama_class.assert_called_once_with(
                host="http://env-host:11434", model="env-model"
            )
            assert service is mock_service

    @patch("scirag.service.llm_services.OllamaService")
    def test_creates_ollama_service_with_custom_config(self, mock_ollama_class):
        """Test creating Ollama service with custom configuration."""
        mock_service = MagicMock()
        mock_ollama_class.return_value = mock_service

        config = {
            "service": "ollama",
            "host": "http://custom:11434",
            "model": "custom-model",
        }
        service = get_llm_service(config)

        mock_ollama_class.assert_called_once_with(
            host="http://custom:11434", model="custom-model"
        )
        assert service is mock_service

    @patch("scirag.service.llm_services.OllamaService")
    def test_creates_ollama_service_with_partial_config(self, mock_ollama_class):
        """Test creating Ollama service with partial configuration."""
        mock_service = MagicMock()
        mock_ollama_class.return_value = mock_service

        with patch.dict(
            os.environ,
            {"OLLAMA_HOST": "http://env-host:11434", "OLLAMA_MODEL": "env-model"},
        ):
            # Only override host, model should come from env
            config = {"host": "http://custom:11434"}
            service = get_llm_service(config)

            mock_ollama_class.assert_called_once_with(
                host="http://custom:11434", model="env-model"
            )
            assert service is mock_service

    @patch("scirag.service.llm_services.OllamaService")
    def test_uses_hardcoded_defaults_when_no_env(self, mock_ollama_class):
        """Test that hardcoded defaults are used when env vars are missing."""
        mock_service = MagicMock()
        mock_ollama_class.return_value = mock_service

        with patch.dict(os.environ, {}, clear=True):
            service = get_llm_service()

            mock_ollama_class.assert_called_once_with(
                host="http://localhost:11434", model="llama3"
            )
            assert service is mock_service

    def test_raises_error_for_unsupported_service(self):
        """Test that ValueError is raised for unsupported service types."""
        config = {"service": "unsupported"}

        with pytest.raises(ValueError, match="Unsupported service type: unsupported"):
            get_llm_service(config)

    @patch("scirag.service.llm_services.OllamaService")
    def test_default_service_type_is_ollama(self, mock_ollama_class):
        """Test that 'ollama' is the default service type."""
        mock_service = MagicMock()
        mock_ollama_class.return_value = mock_service

        with patch.dict(os.environ, {}, clear=True):
            # No service type specified
            service = get_llm_service({})

            # Should still create OllamaService
            assert mock_ollama_class.called
            assert service is mock_service


class TestLLMServiceProtocol:
    """Tests for LLMService protocol compliance."""

    @pytest.mark.asyncio
    async def test_ollama_service_implements_protocol(self):
        """Test that OllamaService implements the LLMService protocol."""
        from scirag.service.llm_services import LLMService

        service = OllamaService(host="http://test:11434", model="test-model")

        # Check that OllamaService has the required method
        assert hasattr(service, "generate_response")
        assert callable(service.generate_response)

        # Verify it matches the protocol signature
        # This will be checked by type checkers like mypy
        service_instance: LLMService = service  # noqa: F841
