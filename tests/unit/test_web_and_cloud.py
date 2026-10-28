"""
Unit tests for Cloud LLM client and FastAPI web interface.
"""

from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
import pytest

from codelens.config import CodeLensConfig
from codelens.llm.cloud_client import CloudLLMClient
from codelens.llm.factory import create_llm_client
from codelens.web.app import app


client = TestClient(app)


def test_cloud_llm_client_initialization():
    cloud = CloudLLMClient(
        api_key="test-api-key",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai",
        default_model="gemini-2.5-flash",
    )
    assert cloud.api_key == "test-api-key"
    assert cloud.default_model == "gemini-2.5-flash"
    assert "generativelanguage" in cloud.base_url


def test_cloud_llm_client_mock_gemini_chat():
    cloud = CloudLLMClient(api_key="AIzaSyMockKey", default_model="gemini-1.5-flash")
    with patch("urllib.request.urlopen") as mock_urlopen:
        mock_response = MagicMock()
        mock_response.read.return_value = (
            b'{"candidates": [{"content": {"parts": [{"text": "{\\"thought\\": \\"gemini_ok\\"}"}]}}]}'
        )
        mock_urlopen.return_value.__enter__.return_value = mock_response

        res = cloud.chat(messages=[{"role": "user", "content": "hello"}])
        assert '{"thought": "gemini_ok"}' in res


def test_cloud_llm_client_mock_openai_chat():
    cloud = CloudLLMClient(
        api_key="gsk-mock-key",
        base_url="https://api.groq.com/openai/v1",
        default_model="llama-3.3-70b-versatile",
    )
    with patch("urllib.request.urlopen") as mock_urlopen:
        mock_response = MagicMock()
        mock_response.read.return_value = (
            b'{"choices": [{"message": {"content": "{\\"thought\\": \\"openai_ok\\"}"}}]}'
        )
        mock_urlopen.return_value.__enter__.return_value = mock_response

        res = cloud.chat(messages=[{"role": "user", "content": "hello"}])
        assert '{"thought": "openai_ok"}' in res


def test_factory_switching():
    # Force cloud provider
    cfg_cloud = CodeLensConfig(provider="cloud", cloud_api_key="test-key")
    c_cloud = create_llm_client(cfg_cloud)
    assert isinstance(c_cloud, CloudLLMClient)

    # Force ollama provider
    cfg_ollama = CodeLensConfig(provider="ollama")
    c_ollama = create_llm_client(cfg_ollama)
    from codelens.llm.ollama_client import OllamaClient
    assert isinstance(c_ollama, OllamaClient)


def test_web_index_endpoint():
    res = client.get("/")
    assert res.status_code == 200
    assert "CodeLens AI" in res.text


def test_web_status_endpoint():
    res = client.get("/api/status")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "online"
    assert "provider" in data


def test_web_files_endpoint():
    res = client.get("/api/files")
    assert res.status_code == 200
    data = res.json()
    assert "files" in data
    assert isinstance(data["files"], list)


def test_web_settings_update():
    res = client.post("/api/settings", json={"provider": "cloud", "cloud_api_key": "dummy-key"})
    assert res.status_code == 200
    assert res.json()["status"] == "updated"
