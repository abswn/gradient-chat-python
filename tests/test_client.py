import json
import tempfile
from pathlib import Path
import pytest
from unittest.mock import patch, MagicMock

from gradient_chat.client import GradientChatClient, GradientChatError
from gradient_chat.conversation import GradientConversation


@pytest.fixture
def temp_log_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def client(temp_log_dir):
    # Patch headers generation to avoid relying on fake_useragent/network
    with patch("gradient_chat.client.generate_headers", return_value={"user-agent": "pytest"}):
        return GradientChatClient(log_dir=temp_log_dir)


def test_get_model_info_success(client):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"data": {"availableModels": ["GPT OSS 120B", "Qwen3 235B"]}}

    with patch("gradient_chat.client.requests.get", return_value=mock_resp) as mock_get:
        models = client.get_model_info()
        assert models == ["GPT OSS 120B", "Qwen3 235B"]
        mock_get.assert_called_once()


def test_get_model_info_failure_timeout(client):
    from requests.exceptions import Timeout
    with patch("gradient_chat.client.requests.get", side_effect=Timeout()):
        models = client.get_model_info()
        assert models == []


def test_get_model_info_failure_request_exception(client):
    from requests.exceptions import RequestException
    with patch("gradient_chat.client.requests.get", side_effect=RequestException("network fail")):
        models = client.get_model_info()
        assert models == []


def test_generate_success(client):
    """Test normal generate() request without thinking"""
    # Mock conversation state (so context retrieval works)
    client._internal_conversation = GradientConversation()
    client._internal_conversation.add_user_message("Earlier Q?")
    client._internal_conversation.add_assistant_message("Earlier A.")

    # Simulated API streaming response
    lines = [
        json.dumps({"type": "jobInfo", "data": {"status": "processing"}}),
        json.dumps({"type": "clusterInfo", "data": {"model": "GPT OSS 120B"}}),
        json.dumps({"type": "reply", "data": {"role": "assistant", "content": "Hello there"}}),
        json.dumps({"type": "jobInfo", "data": {"status": "completed"}}),
    ]

    mock_post = MagicMock()
    mock_post.status_code = 200
    mock_post.text = "\n".join(lines)
    mock_post.raise_for_status = lambda: None

    with patch("gradient_chat.client.requests.post", return_value=mock_post) as mock_req:
        result = client.generate("Hello")
        assert result["reply"] == "Hello there"
        assert result["reasoning"] == ""
        assert result["model"] == "GPT OSS 120B"
        mock_req.assert_called_once()

        # Inspect payload
        sent_payload = json.loads(mock_req.call_args.kwargs["data"])
        assert sent_payload["model"] == client.model
        assert sent_payload["clusterMode"] == client.cluster_mode
        assert sent_payload["messages"][-1] == {"role": "user", "content": "Hello"}
        assert sent_payload["enableThinking"] is False


def test_generate_with_thinking(client):
    """Test when reasoningContent is present"""
    client._internal_conversation = GradientConversation()

    lines = [
        json.dumps({"type": "clusterInfo", "data": {"model": "GPT OSS 120B"}}),
        json.dumps({"type": "reply", "data": {"role": "assistant", "reasoningContent": "Step 1", "content": "Hi"}}),
        json.dumps({"type": "jobInfo", "data": {"status": "completed"}}),
    ]

    mock_post = MagicMock()
    mock_post.status_code = 200
    mock_post.text = "\n".join(lines)
    mock_post.raise_for_status = lambda: None

    with patch("gradient_chat.client.requests.post", return_value=mock_post):
        result = client.generate("Test", enableThinking=True)
        assert result["reasoning"] == "Step 1"
        assert result["reply"] == "Hi"
        assert result["model"] == "GPT OSS 120B"


def test_generate_timeout(client):
    from requests.exceptions import Timeout
    with patch("gradient_chat.client.requests.post", side_effect=Timeout()):
        with pytest.raises(GradientChatError) as exc:
            client.generate("Test")
        assert "timeout" in str(exc.value).lower()


def test_generate_http_error(client):
    from requests.exceptions import HTTPError
    mock_resp = MagicMock()
    mock_resp.status_code = 500
    mock_resp.text = "Internal error"
    err = HTTPError(response=mock_resp)
    with patch("gradient_chat.client.requests.post", side_effect=err):
        with pytest.raises(GradientChatError) as exc:
            client.generate("Test")
        assert "HTTP error 500: Internal error" in str(exc.value)


def test_generate_request_exception(client):
    from requests.exceptions import RequestException
    with patch("gradient_chat.client.requests.post", side_effect=RequestException("fail")):
        with pytest.raises(GradientChatError) as exc:
            client.generate("Test")
        # The client wraps this as "Network error: ..."
        assert "Network error:" in str(exc.value)


def test_generate_job_failed(client):
    """Job never completes"""
    lines = [
        json.dumps({"type": "jobInfo", "data": {"status": "processing"}})
    ]  # never "completed"
    mock_post = MagicMock()
    mock_post.status_code = 200
    mock_post.text = "\n".join(lines)
    mock_post.raise_for_status = lambda: None
    with patch("gradient_chat.client.requests.post", return_value=mock_post):
        with pytest.raises(GradientChatError) as exc:
            client.generate("Test")
        assert "Job failed" in str(exc.value)
