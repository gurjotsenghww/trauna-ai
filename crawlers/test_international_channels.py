import yt_dlp
import os

TEST_CREATORS = [
    {"name": "MrBeast",         "handle": "MrBeast"},
    {"name": "Markiplier",      "handle": "markiplier"},
    {"name": "RyanTrahan",      "handle": "ryan"},
    {"name": "Jacksepticeye",   "handle": "jacksepticeye"},
    {"name": "IShowSpeed",      "handle": "IShowSpeed"},
    {"name": "JakePaul",        "handle": "jakepaul"},
    {"name": "Sidemen",         "handle": "Sidemen"},
    {"name": "KaiCenat",        "handle": "KaiCenat"},
    {"name": "PewDiePie",       "handle": "PewDiePie"},
    {"name": "Caylus",          "handle": "Infinite"},
    {"name": "Caylus2",         "handle": "Caylus"},
    {"name": "Gunscreen",       "handle": "Gunscreen1"},
    {"name": "JeeJYT",          "handle": "jeeJYT"},
    {"name": "22luke",          "handle": "22luke"},
    {"name": "Qzeq",            "handle": "qzeq"},
    {"name": "Javeus",          "handle": "jjaveus"},
    {"name": "CoryxKenshin",    "handle": "CoryxKenshin"},
    {"name": "SSundee",         "handle": "SSundee"},
    {"name": "Ninja",           "handle": "Ninja"},
    {"name": "Aphmau",          "handle": "Aphmau"},
    {"name": "AliA",            "handle": "AliA"},
]

cookie_file = os.path.abspath("crawlers/cookies.txt")
ydl_opts = {
    "extract_flat": True,
    "quiet": True,
    "no_warnings": True,
    "playlist_items": "1-3",
}
if os.path.exists(cookie_file):
    ydl_opts["cookiefile"] = cookie_file

print(f"Testing {len(TEST_CREATORS)} International Channels...")

with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    for c in TEST_CREATORS:
        url = f"https://www.youtube.com/@{c['handle']}/videos"
        try:
            info = ydl.extract_info(url, download=False)
            title = info.get("channel") or info.get("title") or "Unknown"
            count = len(info.get("entries", []))
            print(f"OK: {c['name']} (@{c['handle']}) -> '{title}' ({count} sample vids)")
        except Exception as e:
            print(f"FAIL: {c['name']} (@{c['handle']}) -> {e}")
