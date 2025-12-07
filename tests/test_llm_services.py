"""Tests for the llm_services module."""

import asyncio
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from scirag.llm.providers import (
    GeminiService,
    OllamaService,
    get_llm_service,
)


class TestOllamaService:
    """Tests for OllamaService class."""

    # Removed test_init - trivial test that only verifies constructor assignments work

    @pytest.mark.asyncio
    async def test_generate_response_success(self):
        """Test successful response generation."""
        service = OllamaService(host="http://test:11434", model="test-model")

        # Mock the client.chat method with ChatResponse-like object
        mock_message = MagicMock()
        mock_message.content = "Hello, world!"
        mock_message.tool_calls = None
        mock_response = MagicMock()
        mock_response.message = mock_message
        service.client.chat = MagicMock(return_value=mock_response)

        messages = [{"role": "user", "content": "Say hello"}]
        response = await service.generate_response(messages)

        assert response == "Hello, world!"
        service.client.chat.assert_called_once_with(model="test-model", messages=messages)

    @pytest.mark.asyncio
    async def test_generate_response_with_multiple_messages(self):
        """Test response generation with conversation history."""
        service = OllamaService(host="http://test:11434", model="test-model")

        # Mock the client.chat method with ChatResponse-like object
        mock_message = MagicMock()
        mock_message.content = "I'm doing well, thanks!"
        mock_message.tool_calls = None
        mock_response = MagicMock()
        mock_response.message = mock_message
        service.client.chat = MagicMock(return_value=mock_response)

        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"},
        ]
        response = await service.generate_response(messages)

        assert response == "I'm doing well, thanks!"
        service.client.chat.assert_called_once_with(model="test-model", messages=messages)

    @pytest.mark.asyncio
    async def test_generate_response_with_system_message(self):
        """Test response generation with system message."""
        service = OllamaService(host="http://test:11434", model="test-model")

        # Mock the client.chat method with ChatResponse-like object
        mock_message = MagicMock()
        mock_message.content = "I am a helpful assistant focused on science."
        mock_message.tool_calls = None
        mock_response = MagicMock()
        mock_response.message = mock_message
        service.client.chat = MagicMock(return_value=mock_response)

        messages = [
            {"role": "system", "content": "You are a scientific assistant."},
            {"role": "user", "content": "Who are you?"},
        ]
        response = await service.generate_response(messages)

        assert response == "I am a helpful assistant focused on science."
        service.client.chat.assert_called_once_with(model="test-model", messages=messages)

    @pytest.mark.integration
    @pytest.mark.requires_ollama
    def test_generate_embeddings_real_ollama(self, ollama_service):
        """Test generating embeddings with real Ollama service."""
        texts = ["Python programming", "Machine learning", "Data science"]
        embeddings = ollama_service.generate_embeddings(texts, "nomic-embed-text")
        
        # Verify we got embeddings for all texts
        assert len(embeddings) == 3
        
        # Verify embedding structure (nomic-embed-text produces 768-dim vectors)
        for embedding in embeddings:
            assert isinstance(embedding, list)
            assert len(embedding) == 768
            assert all(isinstance(x, (int, float)) for x in embedding)
        
        # Verify embeddings are different for different texts
        assert embeddings[0] != embeddings[1]
        assert embeddings[1] != embeddings[2]

    @pytest.mark.integration
    @pytest.mark.requires_ollama
    def test_generate_embeddings_similarity(self, ollama_service):
        """Test that similar texts have similar embeddings."""
        from scirag.service.database import cosine_similarity
        
        # Similar texts
        text1 = "Python programming language"
        text2 = "Programming in Python"
        # Dissimilar text
        text3 = "Cooking delicious recipes"
        
        embeddings = ollama_service.generate_embeddings(
            [text1, text2, text3], 
            "nomic-embed-text"
        )
        
        # Calculate similarities
        sim_1_2 = cosine_similarity(embeddings[0], embeddings[1])
        sim_1_3 = cosine_similarity(embeddings[0], embeddings[2])
        
        # Similar texts should have higher similarity than dissimilar texts
        assert sim_1_2 > sim_1_3
        assert sim_1_2 > 0.5  # Related texts should have decent similarity

    def test_generate_embeddings_multiple_texts(self):
        """Test generating embeddings for multiple texts with OllamaService."""
        service = OllamaService(host="http://test:11434", model="test-model")

        # Mock the client.embed method
        service.client.embed = MagicMock(
            side_effect=[
                {"embeddings": [[0.1, 0.2]]},
                {"embeddings": [[0.3, 0.4]]},
                {"embeddings": [[0.5, 0.6]]},
            ]
        )

        texts = ["text1", "text2", "text3"]
        embeddings = service.generate_embeddings(texts, "nomic-embed-text")

        assert len(embeddings) == 3
        assert embeddings[0] == [0.1, 0.2]
        assert embeddings[1] == [0.3, 0.4]
        assert embeddings[2] == [0.5, 0.6]
        assert service.client.embed.call_count == 3

    def test_generate_embeddings_default_model(self):
        """Test generating embeddings with default model."""
        service = OllamaService(host="http://test:11434", model="test-model")

        # Mock the client.embed method
        mock_response = {"embeddings": [[0.1, 0.2, 0.3]]}
        service.client.embed = MagicMock(return_value=mock_response)

        # Clear EMBEDDING_MODEL env to test default
        with patch.dict(os.environ, {}, clear=True):
            embeddings = service.generate_embeddings(["test text"])

        assert len(embeddings) == 1
        # Should use default model "nomic-embed-text" when EMBEDDING_MODEL env is not set
        service.client.embed.assert_called_once_with(model="nomic-embed-text", input="test text")

    @pytest.mark.asyncio
    async def test_generate_response_with_mcp_servers(self):
        """Test response generation with MCP servers parameter."""
        service = OllamaService(host="http://test:11434", model="test-model")

        # Mock the client.chat method with ChatResponse-like object
        mock_message = MagicMock()
        mock_message.content = "Response with tools available"
        mock_message.tool_calls = None
        mock_response = MagicMock()
        mock_response.message = mock_message
        service.client.chat = MagicMock(return_value=mock_response)

        # Mock the discover_mcp_tools method to avoid actual MCP connections
        service.discover_mcp_tools = AsyncMock(return_value=([], {}))

        messages = [{"role": "user", "content": "Use a tool"}]
        mcp_servers = ["http://localhost:8001/sse", "http://localhost:8002/sse"]
        response = await service.generate_response(messages, mcp_servers=mcp_servers)

        assert response == "Response with tools available"
        service.client.chat.assert_called_once_with(model="test-model", messages=messages)

    @pytest.mark.asyncio
    async def test_generate_response_with_tool_calls(self):
        """Test response generation when the model calls tools."""
        service = OllamaService(host="http://test:11434", model="test-model")

        # Create mock tool call
        mock_tool_call = MagicMock()
        mock_tool_call.function.name = "retrieve_document_chunks"
        mock_tool_call.function.arguments = {"query": "test query", "top_k": 5}

        # First response has tool calls
        mock_message_with_tools = MagicMock()
        mock_message_with_tools.content = ""
        mock_message_with_tools.tool_calls = [mock_tool_call]
        mock_message_with_tools.model_dump.return_value = {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"function": {"name": "retrieve_document_chunks"}}],
        }
        mock_response_with_tools = MagicMock()
        mock_response_with_tools.message = mock_message_with_tools

        # Second response is the final answer
        mock_message_final = MagicMock()
        mock_message_final.content = "Based on the documents, here is your answer."
        mock_message_final.tool_calls = None
        mock_response_final = MagicMock()
        mock_response_final.message = mock_message_final

        service.client.chat = MagicMock(side_effect=[mock_response_with_tools, mock_response_final])

        # Mock tool discovery to return a tool (using generic format now)
        mock_tools = [
            {
                "name": "retrieve_document_chunks",
                "description": "Search documents",
                "parameters": {"type": "object", "properties": {}},
            }
        ]
        mock_registry = {"retrieve_document_chunks": ("http://localhost:8001/sse", MagicMock())}
        service.discover_mcp_tools = AsyncMock(return_value=(mock_tools, mock_registry))

        # Mock the MCP tool call
        service.call_mcp_tool = AsyncMock(return_value='[{"content": "test chunk"}]')

        messages = [{"role": "user", "content": "Search for information"}]
        response = await service.generate_response(messages, mcp_servers=["http://localhost:8001/sse"])

        assert response == "Based on the documents, here is your answer."
        # Should have called chat twice (initial + after tool results)
        assert service.client.chat.call_count == 2
        # Should have called the MCP tool
        service.call_mcp_tool.assert_called_once_with(
            "http://localhost:8001/sse",
            "retrieve_document_chunks",
            {"query": "test query", "top_k": 5},
        )


class TestGetLLMService:
    """Tests for get_llm_service factory function."""

    @patch("scirag.llm.providers.OllamaService")
    def test_creates_ollama_service_with_defaults(self, mock_ollama_class):
        """Test creating Ollama service with default configuration."""
        mock_service = MagicMock()
        mock_ollama_class.return_value = mock_service

        # Need to explicitly set LLM_SERVICE=ollama (or unset it) to get OllamaService
        with patch.dict(
            os.environ,
            {"OLLAMA_HOST": "http://env-host:11434", "LLM_MODEL": "env-model", "LLM_SERVICE": "ollama"},
        ):
            service = get_llm_service()

            mock_ollama_class.assert_called_once_with(
                host="http://env-host:11434", model="env-model"
            )
            assert service is mock_service

    @patch("scirag.llm.providers.OllamaService")
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

        mock_ollama_class.assert_called_once_with(host="http://custom:11434", model="custom-model")
        assert service is mock_service

    @patch("scirag.llm.providers.OllamaService")
    def test_creates_ollama_service_with_partial_config(self, mock_ollama_class):
        """Test creating Ollama service with partial configuration."""
        mock_service = MagicMock()
        mock_ollama_class.return_value = mock_service

        # Need to set LLM_SERVICE=ollama to ensure OllamaService is selected
        with patch.dict(
            os.environ,
            {"OLLAMA_HOST": "http://env-host:11434", "LLM_MODEL": "env-model", "LLM_SERVICE": "ollama"},
        ):
            # Only override host, model should come from env
            config = {"host": "http://custom:11434"}
            service = get_llm_service(config)

            mock_ollama_class.assert_called_once_with(host="http://custom:11434", model="env-model")
            assert service is mock_service

    @patch("scirag.llm.providers.OllamaService")
    def test_uses_hardcoded_defaults_when_no_env(self, mock_ollama_class):
        """Test that hardcoded defaults are used when env vars are missing."""
        mock_service = MagicMock()
        mock_ollama_class.return_value = mock_service

        with patch.dict(os.environ, {}, clear=True):
            service = get_llm_service()

            mock_ollama_class.assert_called_once_with(host="http://localhost:11434", model="llama3")
            assert service is mock_service

    def test_raises_error_for_unsupported_service(self):
        """Test that ValueError is raised for unsupported service types."""
        config = {"service": "unsupported"}

        with pytest.raises(ValueError, match="Unsupported service type: unsupported"):
            get_llm_service(config)

    @patch("scirag.llm.providers.OllamaService")
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
        from scirag.llm.providers import LLMService

        service = OllamaService(host="http://test:11434", model="test-model")

        # Check that OllamaService has the required method
        assert hasattr(service, "generate_response")
        assert callable(service.generate_response)

        # Verify it matches the protocol signature
        # This will be checked by type checkers like mypy
        service_instance: LLMService = service  # noqa: F841


class TestGeminiService:
    """Tests for GeminiService class."""

    # Removed test_init - trivial test that only verifies constructor assignments work

    @pytest.mark.asyncio
    @patch("scirag.llm.providers.genai.Client")
    async def test_generate_response_success(self, mock_client_class):
        """Test successful response generation."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Hello, world!"
        mock_response.candidates = []  # No tool calls
        mock_client.models.generate_content.return_value = mock_response
        mock_client_class.return_value = mock_client

        service = GeminiService(model="gemini-2.5-flash")
        messages = [{"role": "user", "content": "Say hello"}]
        response = await service.generate_response(messages)

        assert response == "Hello, world!"
        mock_client.models.generate_content.assert_called_once_with(
            model="gemini-2.5-flash", contents="Say hello"
        )

    @pytest.mark.asyncio
    @patch("scirag.llm.providers.genai.Client")
    async def test_generate_response_with_multiple_messages(self, mock_client_class):
        """Test response generation with conversation history."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "I'm doing well, thanks!"
        mock_response.candidates = []  # No tool calls
        mock_client.models.generate_content.return_value = mock_response
        mock_client_class.return_value = mock_client

        service = GeminiService(model="gemini-2.5-flash")
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"},
        ]
        response = await service.generate_response(messages)

        assert response == "I'm doing well, thanks!"
        # Messages are concatenated
        expected_contents = "Hello\nHi there!\nHow are you?"
        mock_client.models.generate_content.assert_called_once_with(
            model="gemini-2.5-flash", contents=expected_contents
        )

    @pytest.mark.asyncio
    @patch("scirag.llm.providers.genai.Client")
    async def test_generate_response_with_system_message(self, mock_client_class):
        """Test response generation with system message."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "I am a helpful assistant focused on science."
        mock_response.candidates = []  # No tool calls
        mock_client.models.generate_content.return_value = mock_response
        mock_client_class.return_value = mock_client

        service = GeminiService(model="gemini-2.5-flash")
        messages = [
            {"role": "system", "content": "You are a scientific assistant."},
            {"role": "user", "content": "Who are you?"},
        ]
        response = await service.generate_response(messages)

        assert response == "I am a helpful assistant focused on science."
        expected_contents = "You are a scientific assistant.\nWho are you?"
        mock_client.models.generate_content.assert_called_once_with(
            model="gemini-2.5-flash", contents=expected_contents
        )

    @pytest.mark.asyncio
    @patch("scirag.llm.providers.genai.Client")
    async def test_generate_response_error(self, mock_client_class):
        """Test error handling in response generation."""
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = Exception("API Error")
        mock_client_class.return_value = mock_client

        service = GeminiService(model="gemini-2.5-flash")
        messages = [{"role": "user", "content": "Say hello"}]

        with pytest.raises(Exception, match="API Error"):
            await service.generate_response(messages)

    @pytest.mark.asyncio
    @patch("scirag.llm.providers.genai.Client")
    async def test_generate_response_with_mcp_servers(self, mock_client_class):
        """Test response generation with MCP servers parameter."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Response with tools available"
        # Mock response without function calls
        mock_response.candidates = []
        mock_client.models.generate_content.return_value = mock_response
        mock_client_class.return_value = mock_client

        service = GeminiService(model="gemini-2.5-flash")

        # Mock the discover_mcp_tools method to avoid actual MCP connections
        service.discover_mcp_tools = AsyncMock(return_value=([], {}))

        messages = [{"role": "user", "content": "Use a tool"}]
        mcp_servers = ["http://localhost:8001/sse", "http://localhost:8002/sse"]
        response = await service.generate_response(messages, mcp_servers=mcp_servers)

        assert response == "Response with tools available"
        # Verify generate_content was called (no tools since discover returned empty)
        mock_client.models.generate_content.assert_called_once_with(
            model="gemini-2.5-flash", contents="Use a tool"
        )


    @patch("scirag.llm.providers.genai.Client")
    def test_generate_embeddings_multiple_texts(self, mock_client_class):
        """Test generating embeddings for multiple texts with GeminiService."""
        mock_client = MagicMock()

        # Create mock responses for each text
        def create_mock_response(values):
            mock_response = MagicMock()
            mock_embedding = MagicMock()
            mock_embedding.values = values
            mock_response.embeddings = [mock_embedding]
            return mock_response

        mock_client.models.embed_content.side_effect = [
            create_mock_response([0.1, 0.2]),
            create_mock_response([0.3, 0.4]),
            create_mock_response([0.5, 0.6]),
        ]
        mock_client_class.return_value = mock_client

        service = GeminiService(model="gemini-2.5-flash")
        texts = ["text1", "text2", "text3"]
        embeddings = service.generate_embeddings(texts, "text-embedding-004")

        assert len(embeddings) == 3
        assert embeddings[0] == [0.1, 0.2]
        assert embeddings[1] == [0.3, 0.4]
        assert embeddings[2] == [0.5, 0.6]
        assert mock_client.models.embed_content.call_count == 3


class TestGetLLMServiceExtended:
    """Extended tests for get_llm_service factory function with Gemini."""

    @patch("scirag.llm.providers.GeminiService")
    def test_creates_gemini_service_with_config(self, mock_gemini_class):
        """Test creating Gemini service with custom configuration."""
        mock_service = MagicMock()
        mock_gemini_class.return_value = mock_service

        config = {
            "service": "gemini",
            "model": "gemini-2.0-flash",
        }
        service = get_llm_service(config)

        mock_gemini_class.assert_called_once_with(model="gemini-2.0-flash")
        assert service is mock_service

    @patch("scirag.llm.providers.GeminiService")
    def test_creates_gemini_service_with_defaults(self, mock_gemini_class):
        """Test creating Gemini service with default configuration."""
        mock_service = MagicMock()
        mock_gemini_class.return_value = mock_service

        with patch.dict(os.environ, {"GEMINI_MODEL": "gemini-pro"}, clear=False):
            config = {"service": "gemini"}
            service = get_llm_service(config)

    @patch("scirag.llm.providers.GeminiService")
    def test_creates_gemini_service_with_hardcoded_default(self, mock_gemini_class):
        """Test Gemini service uses hardcoded default when env var missing."""
        mock_service = MagicMock()
        mock_gemini_class.return_value = mock_service

        # Clear GEMINI_MODEL env var if it exists
        with patch.dict(os.environ, {}, clear=True):
            config = {"service": "gemini"}
            service = get_llm_service(config)

            mock_gemini_class.assert_called_once_with(model="gemini-2.5-flash")
            assert service is mock_service
