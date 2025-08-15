import re
import pytest
from unittest.mock import patch

from gradient_chat import headers


# Stable, known UA strings for each platform
FAKE_WINDOWS_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/123.0.0.0 Safari/537.36"
)
FAKE_MACOS_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/123.0.0.0 Safari/537.36"
)
FAKE_ANDROID_UA = (
    "Mozilla/5.0 (Linux; Android 13; Pixel 6) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/123.0.0.0 Mobile Safari/537.36"
)
FAKE_IOS_UA = (
    "Mozilla/5.0 (iPhone; CPU iPhone OS 16_5 like Mac OS X) "
    "AppleWebKit/605.1.15 (KHTML, like Gecko) "
    "Chrome/123.0.0.0 Mobile/15E148 Safari/537.36"
)


def _patch_module_level_ua(ua_string):
    """
    Patch gradient_chat.headers.ua so that ua.chrome returns ua_string deterministically.
    This targets the module-level instance created at import time.
    """
    # Assign a dummy object with a 'chrome' attribute
    class DummyUA:
        def __init__(self, chrome):
            self.chrome = chrome

    return patch("gradient_chat.headers.ua", new=DummyUA(ua_string))


def test_generate_headers_returns_dict():
    with _patch_module_level_ua(FAKE_WINDOWS_UA):
        result = headers.generate_headers()
        assert isinstance(result, dict)


def test_generate_headers_contains_required_keys():
    with _patch_module_level_ua(FAKE_WINDOWS_UA):
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


def test_generate_headers_platform_and_mobile_flags_windows_desktop():
    # Force Windows desktop UA; expect platform Windows and mobile ?0
    with _patch_module_level_ua(FAKE_WINDOWS_UA):
        result = headers.generate_headers()
        assert result["sec-ch-ua-platform"] in ['"Windows"', '"macOS"', '"iOS"', '"Android"']
        assert result["sec-ch-ua-mobile"] in ["?0", "?1"]
        assert result["sec-ch-ua-platform"] == '"Windows"'
        assert result["sec-ch-ua-mobile"] == "?0"


def test_generate_headers_chrome_version_consistency():
    with _patch_module_level_ua(FAKE_WINDOWS_UA):
        result = headers.generate_headers()
        # user-agent must contain Chrome major version
        ua_match = re.search(r"Chrome/(\d+)\.", result["user-agent"])
        assert ua_match, "User-agent should contain Chrome major version"
        ua_major = ua_match.group(1)

        # sec-ch-ua must echo that major version for Google Chrome and Chromium
        scua = result["sec-ch-ua"]
        gchrome_match = re.search(r'"Google Chrome";v="(\d+)"', scua)
        chromium_match = re.search(r'"Chromium";v="(\d+)"', scua)
        assert gchrome_match and chromium_match, "sec-ch-ua should include versioned Google Chrome and Chromium"
        assert gchrome_match.group(1) == ua_major
        assert chromium_match.group(1) == ua_major


@pytest.mark.parametrize(
    "ua, expected_platform, expected_mobile",
    [
        (FAKE_WINDOWS_UA, '"Windows"', "?0"),
        (FAKE_MACOS_UA, '"macOS"', "?0"),
        (FAKE_ANDROID_UA, '"Android"', "?1"),
        (FAKE_IOS_UA, '"iOS"', "?1"),
    ],
)
def test_generate_headers_platform_detection_matrix(ua, expected_platform, expected_mobile):
    with _patch_module_level_ua(ua):
        result = headers.generate_headers()
        # Behavior checks: platform and mobile flags
        assert result["sec-ch-ua-platform"] == expected_platform
        assert result["sec-ch-ua-mobile"] == expected_mobile

        # Version consistency checks
        ua_match = re.search(r"Chrome/(\d+)\.", result["user-agent"])
        assert ua_match, "User-agent should contain Chrome major version"
        ua_major = ua_match.group(1)
        scua = result["sec-ch-ua"]
        gchrome_match = re.search(r'"Google Chrome";v="(\d+)"', scua)
        chromium_match = re.search(r'"Chromium";v="(\d+)"', scua)
        assert gchrome_match and chromium_match
        assert gchrome_match.group(1) == ua_major
        assert chromium_match.group(1) == ua_major
