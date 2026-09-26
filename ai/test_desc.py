import http.cookiejar
import urllib.request
import os
import re

cookie_file = os.path.abspath('crawlers/cookies.txt')
cj = http.cookiejar.MozillaCookieJar(cookie_file)
cj.load(ignore_discard=True, ignore_expires=True)

opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cj))
opener.addheaders = [
    ('User-Agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'),
    ('Accept-Language', 'en-US,en;q=0.9'),
]

url = 'https://www.youtube.com/watch?v=v9QtM6qnG50'
try:
    resp = opener.open(url, timeout=5)
    content = resp.read().decode('utf-8', errors='ignore')
    m = re.search(r'"shortDescription":"(.*?)"', content)
    if m:
        print('SUCCESS with cookies! shortDescription:', m.group(1)[:120])
    else:
        m2 = re.search(r'<meta property="og:description" content="(.*?)">', content)
        print('SUCCESS with cookies! meta:', m2.group(1)[:120] if m2 else 'No meta')
except Exception as e:
    print('Failed with cookies:', e)
