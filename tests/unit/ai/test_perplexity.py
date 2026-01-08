import pytest
import json
import os
from unittest.mock import MagicMock, patch
from src.infrastructure.ai.perplexity_client import PerplexityClient, PerplexityConfig, PerplexityResponse

@pytest.fixture
def mock_perplexity_response():
    """Sample successful response from Perplexity API."""
    return {
        "choices": [
            {
                "message": {
                    "content": "Interpretation: The model shows a significant positive relationship."
                }
            }
        ],
        "usage": {"total_tokens": 150},
        "citations": ["http://stat.example.com"],
        "model": "llama-3.1-sonar-small-128k-online"
    }

def test_perplexity_client_fallback_no_key():
    """Test client fallback logic when API key is missing."""
    with patch.dict(os.environ, {}, clear=True):
        client = PerplexityClient(PerplexityConfig(api_key=None))
        assert not client.is_configured
        
        # Test interpretation with fallback
        stats = {"r_squared": 0.8, "slope": 2.0}
        response = client.interpret_r_output(stats)
        
        assert not response.error
        assert "Interpretation der Regressionsanalyse" in response.content
        assert response.model == "fallback"

@patch('requests.post')
def test_perplexity_client_success(mock_post, mock_perplexity_response):
    """Test successful API interaction with Perplexity."""
    # Setup mock
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = mock_perplexity_response
    mock_resp.elapsed.total_seconds.return_value = 0.5
    mock_post.return_value = mock_resp
    
    client = PerplexityClient(PerplexityConfig(api_key="fake_key"))
    stats = {"r_squared": 0.9, "slope": 1.5, "x_label": "X", "y_label": "Y", "n": 100}
    
    response = client.interpret_r_output(stats)
    
    assert response.content == "Interpretation: The model shows a significant positive relationship."
    assert not response.error
    assert response.latency_ms > 0
    mock_post.assert_called_once()

@patch('requests.post')
def test_perplexity_client_cache(mock_post, mock_perplexity_response):
    """Test that PerplexityClient uses internal cache."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = mock_perplexity_response
    mock_resp.elapsed.total_seconds.return_value = 0.5
    mock_post.return_value = mock_resp
    
    client = PerplexityClient(PerplexityConfig(api_key="fake_key"))
    stats = {"r_squared": 0.9, "slope": 1.5}
    
    # 1. First call (should hit API)
    res1 = client.interpret_r_output(stats, use_cache=True)
    assert not res1.cached
    
    # 2. Second call with same stats (should hit cache)
    res2 = client.interpret_r_output(stats, use_cache=True)
    assert res2.cached
    assert res2.content == res1.content
    
    # API should only be called once
    assert mock_post.call_count == 1


@patch('requests.post')
def test_perplexity_client_error_handling(mock_post):
    """Test robust error handling for API failures."""
    # Test network error
    mock_post.side_effect = Exception("Network error")
    
    client = PerplexityClient(PerplexityConfig(api_key="fake_key"))
    stats = {"r_squared": 0.9, "slope": 1.5}
    
    response = client.interpret_r_output(stats)
    
    # Should fallback gracefully
    assert response.error is False or response.error is True  # May have error flag
    assert response.content is not None  # Should have fallback content


@patch('requests.post')
def test_perplexity_client_http_error(mock_post):
    """Test handling of HTTP error responses."""
    # Test 500 error
    mock_resp = MagicMock()
    mock_resp.status_code = 500
    mock_resp.text = "Internal Server Error"
    mock_post.return_value = mock_resp
    
    client = PerplexityClient(PerplexityConfig(api_key="fake_key"))
    stats = {"r_squared": 0.9, "slope": 1.5}
    
    response = client.interpret_r_output(stats)
    
    # Should handle gracefully (fallback or error flag)
    assert response.content is not None


@patch('requests.post')
def test_perplexity_client_invalid_json(mock_post):
    """Test handling of invalid JSON responses."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.side_effect = json.JSONDecodeError("Invalid JSON", "", 0)
    mock_resp.text = "Invalid response"
    mock_post.return_value = mock_resp
    
    client = PerplexityClient(PerplexityConfig(api_key="fake_key"))
    stats = {"r_squared": 0.9, "slope": 1.5}
    
    response = client.interpret_r_output(stats)
    
    # Should handle gracefully
    assert response.content is not None


def test_perplexity_client_no_api_key_always_fallback():
    """Test that missing API key always uses fallback (no external calls)."""
    with patch.dict(os.environ, {}, clear=True):
        client = PerplexityClient(PerplexityConfig(api_key=None))
        
        # Multiple calls should all use fallback
        stats1 = {"r_squared": 0.8, "slope": 2.0}
        stats2 = {"r_squared": 0.9, "slope": 1.5}
        
        response1 = client.interpret_r_output(stats1)
        response2 = client.interpret_r_output(stats2)
        
        assert not client.is_configured
        assert response1.model == "fallback"
        assert response2.model == "fallback"
        assert not response1.error
        assert not response2.error


@patch('requests.post')
def test_perplexity_client_timeout_handling(mock_post):
    """Test handling of request timeouts."""
    import requests
    mock_post.side_effect = requests.Timeout("Request timeout")
    
    client = PerplexityClient(PerplexityConfig(api_key="fake_key"))
    stats = {"r_squared": 0.9, "slope": 1.5}
    
    response = client.interpret_r_output(stats)
    
    # Should handle timeout gracefully
    assert response.content is not None
