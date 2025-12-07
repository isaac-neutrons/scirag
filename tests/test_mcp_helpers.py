"""Tests for MCP helper functions."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from scirag.service.mcp_helpers import (
    call_mcp_tool,
    check_mcp_server,
    extract_mcp_result,
    run_async,
)


class TestExtractMcpResult:
    """Tests for extract_mcp_result function."""

    def test_extracts_json_from_text_content_list(self):
        """Test extracting JSON from a TextContent list."""
        mock_result = MagicMock()
        mock_text_content = MagicMock()
        mock_text_content.text = '{"key": "value", "number": 42}'
        mock_result.content = [mock_text_content]

        result = extract_mcp_result(mock_result)

        assert result == {"key": "value", "number": 42}

    def test_extracts_list_from_text_content(self):
        """Test extracting a JSON list from TextContent."""
        mock_result = MagicMock()
        mock_text_content = MagicMock()
        mock_text_content.text = '["item1", "item2", "item3"]'
        mock_result.content = [mock_text_content]

        result = extract_mcp_result(mock_result)

        assert result == ["item1", "item2", "item3"]

    def test_returns_text_if_not_valid_json(self):
        """Test that non-JSON text is returned as-is."""
        mock_result = MagicMock()
        mock_text_content = MagicMock()
        mock_text_content.text = "plain text response"
        mock_result.content = [mock_text_content]

        result = extract_mcp_result(mock_result)

        assert result == "plain text response"

    def test_returns_content_if_not_list(self):
        """Test returning content directly if not a list."""
        mock_result = MagicMock()
        mock_result.content = "direct content"

        result = extract_mcp_result(mock_result)

        assert result == "direct content"

    def test_returns_result_if_no_content_attribute(self):
        """Test returning result as-is if no content attribute."""
        result = extract_mcp_result({"raw": "data"})

        assert result == {"raw": "data"}

    def test_returns_result_if_content_is_empty(self):
        """Test returning result if content is empty."""
        mock_result = MagicMock()
        mock_result.content = []

        result = extract_mcp_result(mock_result)

        assert result == mock_result

    def test_returns_result_if_content_is_none(self):
        """Test returning result if content is None."""
        mock_result = MagicMock()
        mock_result.content = None

        result = extract_mcp_result(mock_result)

        assert result == mock_result


class TestCallMcpTool:
    """Tests for call_mcp_tool async function."""

    @pytest.mark.asyncio
    async def test_calls_tool_with_params(self):
        """Test calling an MCP tool with parameters."""
        mock_text_content = MagicMock()
        mock_text_content.text = '{"success": true}'
        mock_result = MagicMock()
        mock_result.content = [mock_text_content]

        mock_client = AsyncMock()
        mock_client.call_tool = AsyncMock(return_value=mock_result)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch("scirag.service.mcp_helpers.MCPClient", return_value=mock_client):
            result = await call_mcp_tool(
                "http://localhost:8001/sse",
                "test_tool",
                {"param1": "value1"},
            )

        assert result == {"success": True}
        mock_client.call_tool.assert_called_once_with("test_tool", {"param1": "value1"})

    @pytest.mark.asyncio
    async def test_calls_tool_with_empty_params(self):
        """Test calling an MCP tool with no parameters."""
        mock_text_content = MagicMock()
        mock_text_content.text = '[]'
        mock_result = MagicMock()
        mock_result.content = [mock_text_content]

        mock_client = AsyncMock()
        mock_client.call_tool = AsyncMock(return_value=mock_result)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch("scirag.service.mcp_helpers.MCPClient", return_value=mock_client):
            result = await call_mcp_tool("http://localhost:8001/sse", "list_items")

        assert result == []
        mock_client.call_tool.assert_called_once_with("list_items", {})

    @pytest.mark.asyncio
    async def test_propagates_exceptions(self):
        """Test that exceptions from MCP client are propagated."""
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(side_effect=ConnectionError("Server unavailable"))

        with patch("scirag.service.mcp_helpers.MCPClient", return_value=mock_client):
            with pytest.raises(ConnectionError, match="Server unavailable"):
                await call_mcp_tool("http://localhost:8001/sse", "test_tool")


class TestCheckMcpServer:
    """Tests for check_mcp_server async function."""

    @pytest.mark.asyncio
    async def test_returns_connected_status_with_tools(self):
        """Test checking a connected server with tools."""
        mock_tool1 = MagicMock()
        mock_tool1.name = "tool1"
        mock_tool2 = MagicMock()
        mock_tool2.name = "tool2"

        mock_server_info = MagicMock()
        mock_server_info.name = "Test Server"

        mock_init_result = MagicMock()
        mock_init_result.serverInfo = mock_server_info

        mock_client = AsyncMock()
        mock_client.list_tools = AsyncMock(return_value=[mock_tool1, mock_tool2])
        mock_client.initialize_result = mock_init_result
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch("scirag.service.mcp_helpers.MCPClient", return_value=mock_client):
            result = await check_mcp_server("http://localhost:8001/sse")

        assert result["url"] == "http://localhost:8001/sse"
        assert result["status"] == "connected"
        assert result["tools"] == ["tool1", "tool2"]
        assert result["server_name"] == "Test Server"

    @pytest.mark.asyncio
    async def test_returns_connected_without_server_name(self):
        """Test checking a connected server without server name info."""
        mock_client = AsyncMock()
        mock_client.list_tools = AsyncMock(return_value=[])
        mock_client.initialize_result = None
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch("scirag.service.mcp_helpers.MCPClient", return_value=mock_client):
            result = await check_mcp_server("http://localhost:8001/sse")

        assert result["status"] == "connected"
        assert result["tools"] == []
        assert result["server_name"] is None

    @pytest.mark.asyncio
    async def test_returns_failed_status_on_error(self):
        """Test checking a server that fails to connect."""
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(side_effect=ConnectionError("Connection refused"))

        with patch("scirag.service.mcp_helpers.MCPClient", return_value=mock_client):
            result = await check_mcp_server("http://localhost:9999/sse")

        assert result["url"] == "http://localhost:9999/sse"
        assert result["status"] == "failed"
        assert "Connection refused" in result["error"]
        assert "tools" not in result


class TestRunAsync:
    """Tests for run_async helper function."""

    def test_runs_simple_coroutine(self):
        """Test running a simple async function."""
        async def simple_coro():
            return 42

        result = run_async(simple_coro())

        assert result == 42

    def test_runs_coroutine_with_await(self):
        """Test running a coroutine that uses await."""
        import asyncio

        async def sleep_coro():
            await asyncio.sleep(0.01)
            return "completed"

        result = run_async(sleep_coro())

        assert result == "completed"

    def test_propagates_exceptions(self):
        """Test that exceptions from coroutines are propagated."""
        async def error_coro():
            raise ValueError("Test error")

        with pytest.raises(ValueError, match="Test error"):
            run_async(error_coro())

    def test_cleans_up_event_loop(self):
        """Test that the event loop is properly closed after execution."""
        async def simple_coro():
            return "done"

        # Run the coroutine
        run_async(simple_coro())

        # Verify we can run another coroutine (loop was properly cleaned up)
        result = run_async(simple_coro())
        assert result == "done"
