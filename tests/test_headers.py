import pytest
from gradient_chat import headers

def test_generate_headers_returns_dict():
    """Test that generate_headers returns a dictionary"""
    result = headers.generate_headers()
    assert isinstance(result, dict), "Headers should be a dictionary"

def test_generate_headers_contains_required_keys():
    """Test that all expected keys are in the headers dictionary"""
    result = headers.generate_headers()
    expected_keys = [
        "accept",
        "accept-language",
        "priority",
        "origin",
        "referer",
        "sec-ch-ua",
        "sec-ch-ua-mobile",
        "sec-ch-ua-platform",
        "sec-fetch-dest",
        "sec-fetch-mode",
        "sec-fetch-site",
        "user-agent",
    ]
    for key in expected_keys:
        assert key in result, f"Missing header key: {key}"

def test_generate_headers_user_agent_platform_mobile():
    """Test that sec-ch-ua-platform and sec-ch-ua-mobile are set correctly"""
    result = headers.generate_headers()
    assert result["sec-ch-ua-platform"] in ['"Windows"', '"macOS"', '"iOS"', '"Android"']
    assert result["sec-ch-ua-mobile"] in ["?0", "?1"]

def test_generate_headers_chrome_version_format():
    """Test that sec-ch-ua contains Chrome version"""
    result = headers.generate_headers()
    import re
    match = re.search(r'Chrome/(\d+)', result["user-agent"])
    assert match is not None, "User-agent should contain Chrome version"
