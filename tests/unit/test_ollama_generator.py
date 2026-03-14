"""Tests for the OllamaClient wrapper."""

from unittest.mock import MagicMock, patch

import requests

from codebase_rag.llm.ollama_client import OllamaClient


class TestOllamaClient:
    """Test cases for OllamaClient."""

    @patch("codebase_rag.llm.ollama_client.Config")
    def test_initialization(self, mock_config_cls: MagicMock) -> None:
        """Test OllamaClient initialization."""
        mock_config = MagicMock()
        mock_config.llm_model_name = "default-model"
        mock_config.ollama_base_url = "http://localhost:11434"
        mock_config_cls.get_instance.return_value = mock_config

        client = OllamaClient(
            model_name="test-model",
            base_url="http://test:11434",
            timeout=60,
        )

        assert client.model_name == "test-model"
        assert client.base_url == "http://test:11434"
        assert client.timeout == 60

    @patch("codebase_rag.llm.ollama_client.Config")
    def test_invoke(self, mock_config_cls: MagicMock) -> None:
        """Test text generation via invoke."""
        mock_config = MagicMock()
        mock_config.llm_model_name = "test-model"
        mock_config.ollama_base_url = "http://localhost:11434"
        mock_config_cls.get_instance.return_value = mock_config

        client = OllamaClient(model_name="test-model")

        # Mock the ChatOllama inner LLM
        mock_message = MagicMock()
        mock_message.content = "Generated text"
        client._llm = MagicMock()
        client._llm.invoke.return_value = mock_message

        result = client.invoke("Test prompt")
        assert result == "Generated text"
        client._llm.invoke.assert_called_once_with("Test prompt")

    @patch("codebase_rag.llm.ollama_client.requests.get")
    @patch("codebase_rag.llm.ollama_client.Config")
    def test_check_connection_success(self, mock_config_cls: MagicMock, mock_get: MagicMock) -> None:
        """Test successful connection check."""
        mock_config = MagicMock()
        mock_config.llm_model_name = "test-model"
        mock_config.ollama_base_url = "http://localhost:11434"
        mock_config_cls.get_instance.return_value = mock_config

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"version": "0.1.0"}
        mock_get.return_value = mock_response

        client = OllamaClient(model_name="test-model")
        result = client.check_connection()

        assert result["status"] == "connected"
        assert result["version"] == "0.1.0"

    @patch("codebase_rag.llm.ollama_client.requests.get")
    @patch("codebase_rag.llm.ollama_client.Config")
    def test_check_connection_failure(self, mock_config_cls: MagicMock, mock_get: MagicMock) -> None:
        """Test connection check when Ollama is not reachable."""

        mock_config = MagicMock()
        mock_config.llm_model_name = "test-model"
        mock_config.ollama_base_url = "http://localhost:11434"
        mock_config_cls.get_instance.return_value = mock_config

        mock_get.side_effect = requests.exceptions.ConnectionError("Connection refused")

        client = OllamaClient(model_name="test-model")
        result = client.check_connection()

        assert result["status"] == "error"
        assert "Cannot connect" in result["message"]

    @patch("codebase_rag.llm.ollama_client.requests.get")
    @patch("codebase_rag.llm.ollama_client.Config")
    def test_check_model_available(self, mock_config_cls: MagicMock, mock_get: MagicMock) -> None:
        """Test model availability check when model exists."""
        mock_config = MagicMock()
        mock_config.llm_model_name = "test-model"
        mock_config.ollama_base_url = "http://localhost:11434"
        mock_config_cls.get_instance.return_value = mock_config

        # First call: version check, second call: tags
        version_resp = MagicMock()
        version_resp.status_code = 200
        version_resp.json.return_value = {"version": "0.1.0"}

        tags_resp = MagicMock()
        tags_resp.status_code = 200
        tags_resp.json.return_value = {"models": [{"name": "test-model"}]}

        mock_get.side_effect = [version_resp, tags_resp]

        client = OllamaClient(model_name="test-model")
        result = client.check_model_availability()

        assert result["status"] == "available"

    @patch("codebase_rag.llm.ollama_client.requests.get")
    @patch("codebase_rag.llm.ollama_client.Config")
    def test_check_model_not_found(self, mock_config_cls: MagicMock, mock_get: MagicMock) -> None:
        """Test model availability check when model is missing."""
        mock_config = MagicMock()
        mock_config.llm_model_name = "missing-model"
        mock_config.ollama_base_url = "http://localhost:11434"
        mock_config_cls.get_instance.return_value = mock_config

        version_resp = MagicMock()
        version_resp.status_code = 200
        version_resp.json.return_value = {"version": "0.1.0"}

        tags_resp = MagicMock()
        tags_resp.status_code = 200
        tags_resp.json.return_value = {"models": [{"name": "other-model"}]}

        mock_get.side_effect = [version_resp, tags_resp]

        client = OllamaClient(model_name="missing-model")
        result = client.check_model_availability()

        assert result["status"] == "not_found"
