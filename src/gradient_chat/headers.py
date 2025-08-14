
import re
from fake_useragent import UserAgent

ua = UserAgent()

def generate_headers():

    while True:
        # Generate useragent
        user_agent = ua.chrome

        # Find platform
        if "Windows" in user_agent:
            platform = "Windows"
        elif "Macintosh" in user_agent:
            platform = "macOS"
        elif "iPhone" in user_agent:
            platform = "iOS"
        elif "Android" in user_agent:
            platform = "Android"
        else:
            continue
        
        # Find if device is mobile
        if "Mobile" in user_agent:
            mobile = "?1"
        else:
            mobile = "?0"
        
        # Find chrome version
        match = re.search(r'Chrome/([\d\.]+)', user_agent)
        if match:
            chrome_version = match.group(1).split('.')[0]
        else:
            continue

        # Generate headers
        headers = {
            "accept": "*/*",
            "accept-language": "en-US,en;q=0.9",
            "priority": "u=1, i",
            "origin": "https://chat.gradient.network",
            "referer": "https://chat.gradient.network/",
            "sec-ch-ua": f'"Not;A=Brand";v="99", "Google Chrome";v="{chrome_version}", "Chromium";v="{chrome_version}"',
            "sec-ch-ua-mobile": mobile,
            "sec-ch-ua-platform": f'"{platform}"',
            "sec-fetch-dest": "empty",
            "sec-fetch-mode": "cors",
            "sec-fetch-site": "same-origin",
            "user-agent": user_agent
        }
        return headers
