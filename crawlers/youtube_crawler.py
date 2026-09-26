#!/usr/bin/env python3
"""
Tràuna AI — YouTube Thumbnail & Dataset Crawler
Automated crawler for AI training dataset collection.

Directory & File Naming Structure:
  <output_dir>/<creator>/<game>/
      ├── thumbnails/
      │   ├── <creator>_<game>_1.jpg
      │   ├── <creator>_<game>_2.jpg
      │   └── ...
      ├── captions/
      │   ├── <creator>_<game>_1.txt
      │   ├── <creator>_<game>_2.txt
      │   └── ...
      └── metadata/
          ├── <creator>_<game>_1.json
          ├── <creator>_<game>_2.json
          └── ...

Features:
  - Preserves full YouTube video description in metadata & caption
  - Naming convention: <Youtubername>_<Gamename>_<Index>.jpg
  - Skips already downloaded videos
  - High resolution thumbnail extraction (1280x720 maxresdefault)
  - Automatic game classification from video title & description
  - Cookie support for bypass of bot detection / rate limits
"""

import os
import sys
import re
import json
import argparse
import urllib.request
import urllib.error
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, List, Optional
import yt_dlp

# Ensure UTF-8 output encoding on Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Default base directory for collected datasets
DEFAULT_DATASET_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "dataset", "creators")
)

DEFAULT_COOKIE_FILE = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "cookies.txt")
)

# Known game keywords mapping for automated game detection
GAME_KEYWORDS = {
    "Minecraft": [r"\bminecraft\b", r"\bminecraft hardcore\b", r"\bnether\b", r"\bherobrine\b", r"\bminecraft but\b"],
    "GTA_V": [r"\bgta\b", r"\bgta 5\b", r"\bgta v\b", r"\bgrand theft auto\b", r"\blos santos\b"],
    "Resident_Evil": [r"\bresident evil\b", r"\bre4\b", r"\bre8\b", r"\bvillage\b", r"\bbiohazard\b"],
    "Granny": [r"\bgranny\b", r"\bgrandpa\b", r"\bslendrina\b"],
    "Horror_Games": [
        r"\bhorror\b", r"\bfears to fathom\b", r"\bpoppy playtime\b", r"\bfnaf\b",
        r"\bfive nights\b", r"\boutlast\b", r"\bphasmophobia\b", r"\bamnesia\b",
        r"\bchilla's art\b", r"\bthe mortuary assistant\b", r"\bdon't pick up\b",
        r"\brehaunted\b", r"\bscary\b", r"\bhaunted\b", r"\bhellmart\b", r"\bdread neighbor\b"
    ],
    "Elden_Ring": [r"\belden ring\b", r"\bshadow of the erdtree\b", r"\bmalenia\b"],
    "Sekiro": [r"\bsekiro\b", r"\bshadows die twice\b"],
    "God_of_War": [r"\bgod of war\b", r"\bragnarok\b", r"\bkratos\b"],
    "Spider_Man": [r"\bspider-man\b", r"\bmiles morales\b", r"\bspider man\b"],
    "Wolverine": [r"\bwolverine\b", r"\bmarvel wolverine\b"],
    "Red_Dead_Redemption": [r"\bred dead\b", r"\brdr2\b", r"\brdr\b"],
    "Valorant": [r"\bvalorant\b"],
    "Fortnite": [r"\bfortnite\b"],
    "PUBG_BGMI": [r"\bpubg\b", r"\bbgmi\b", r"\bbattlegrounds mobile\b"],
    "Cyberpunk_2077": [r"\bcyberpunk\b", r"\bcyberpunk 2077\b"],
    "Fall_Guys": [r"\bfall guys\b"],
    "Among_Us": [r"\bamong us\b", r"\bimpostor\b"],
}


def sanitize_folder_name(name: str) -> str:
    """Sanitizes directory/file names for Windows and Unix."""
    name = re.sub(r'[\\/*?:"<>|]', "", name)
    name = re.sub(r'\s+', "_", name.strip())
    return name or "General"


def detect_game_from_title(title: str, default: str = "General_Gaming") -> str:
    """Classifies video title into a game category based on keywords and patterns."""
    lower_title = title.lower()

    # 1. Match keyword dictionaries
    for game, patterns in GAME_KEYWORDS.items():
        for pat in patterns:
            if re.search(pat, lower_title):
                return game

    # 2. Look for square brackets or pipe like: [Minecraft - Part 15] or | Game Name
    brackets = re.findall(r'\[(.*?)\]', title)
    for b in brackets:
        clean_b = sanitize_folder_name(b)
        if len(clean_b) > 2 and len(clean_b) <= 25:
            # check if bracket mentions a game
            for game, patterns in GAME_KEYWORDS.items():
                for pat in patterns:
                    if re.search(pat, b.lower()):
                        return game
            return clean_b

    # 3. Look for pipe or hyphen patterns like: "Escaping The Monster | The Classrooms"
    parts = re.split(r"\||-|–|—", title)
    if len(parts) >= 2:
        candidate = sanitize_folder_name(parts[-1].strip())
        if 2 < len(candidate) <= 25:
            return candidate

    return default


def download_thumbnail(video_id: str, dest_path: str) -> bool:
    """Downloads the highest available resolution YouTube thumbnail."""
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    resolutions = ["maxresdefault.jpg", "sddefault.jpg", "hqdefault.jpg"]

    for res in resolutions:
        url = f"https://i.ytimg.com/vi/{video_id}/{res}"
        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req) as response:
                if response.status == 200:
                    data = response.read()
                    # YouTube returns a 1097 byte blank placeholder if maxres does not exist
                    if len(data) > 10000:
                        with open(dest_path, "wb") as f:
                            f.write(data)
                        return True
        except (urllib.error.HTTPError, urllib.error.URLError):
            continue

    return False


def get_video_description(video_id: str, cookie_file: Optional[str] = None) -> str:
    """Fetches video description reliably, silently, and rapidly without triggering bot checks."""
    try:
        url = f"https://www.youtube.com/watch?v={video_id}"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept-Language": "en-US,en;q=0.9",
        }
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=5) as resp:
            html_text = resp.read().decode("utf-8", errors="ignore")
            # 1. shortDescription in player response
            m = re.search(r'"shortDescription":"(.*?)"', html_text)
            if m:
                raw_desc = m.group(1).replace("\\n", "\n").replace('\\"', '"').replace("\\\\", "\\")
                if len(raw_desc.strip()) > 0:
                    return raw_desc.strip()
            # 2. meta name="description"
            m_meta = re.search(r'<meta\s+(?:name|property)="description"\s+content="([^"]*)"', html_text, re.IGNORECASE)
            if m_meta and len(m_meta.group(1).strip()) > 0:
                import html
                return html.unescape(m_meta.group(1).strip())
            # 3. meta property="og:description"
            m_og = re.search(r'<meta\s+property="og:description"\s+content="([^"]*)"', html_text, re.IGNORECASE)
            if m_og and len(m_og.group(1).strip()) > 0:
                import html
                return html.unescape(m_og.group(1).strip())
    except Exception:
        pass

    return ""


def get_existing_video_ids(meta_dir: str) -> set:
    """Finds all video IDs already scraped in this folder."""
    existing_ids = set()
    if not os.path.exists(meta_dir):
        return existing_ids

    for fname in os.listdir(meta_dir):
        if fname.endswith(".json"):
            fpath = os.path.join(meta_dir, fname)
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if "id" in data:
                        existing_ids.add(data["id"])
            except Exception:
                pass
    return existing_ids


def get_next_index(thumb_dir: str, prefix: str) -> int:
    """Finds the next numerical index for files starting with prefix."""
    if not os.path.exists(thumb_dir):
        return 1

    max_idx = 0
    pattern = re.compile(rf"^{re.escape(prefix)}_(\d+)\.(jpg|png|jpeg)$", re.IGNORECASE)
    for fname in os.listdir(thumb_dir):
        match = pattern.match(fname)
        if match:
            max_idx = max(max_idx, int(match.group(1)))
    return max_idx + 1


def generate_training_caption(title: str, description: str, creator: str, game: str) -> str:
    """
    Generates a structured prompt/caption describing the thumbnail
    for future LoRA training in Diffusers / Kohya.
    """
    clean_game = game.replace("_", " ")
    # Clean description snippet (first 150 chars, no urls/hashtags)
    clean_desc = re.sub(r'http\S+', '', description)
    clean_desc = re.sub(r'#\S+', '', clean_desc)
    clean_desc = ' '.join(clean_desc.split())[:150]

    caption = (
        f"YouTube gaming thumbnail in {creator} style, {clean_game} gameplay thumbnail, "
        f"reaction face with intense emotion, cinematic lighting, high saturation, "
        f"bold visual hierarchy, 16:9 aspect ratio, YouTube click-worthy composition. "
        f"Video title: {title.strip()}. "
    )
    if clean_desc:
        caption += f"Context: {clean_desc}"
    return caption.strip()


def crawl(
    target: str,
    creator_name: Optional[str] = None,
    game_override: Optional[str] = None,
    limit: int = 50,
    output_base_dir: str = DEFAULT_DATASET_DIR,
    cookie_file: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Crawls YouTube channel or playlist and downloads thumbnails into structured folders.
    Names files as: <YouTuber>_<Game>_<Index>.jpg
    """
    print(f"\n[Crawler] Target: {target}")
    print(f"[Crawler] Output Directory: {output_base_dir}")
    print(f"[Crawler] Item Limit: {limit}")

    active_cookie_file = cookie_file or (DEFAULT_COOKIE_FILE if os.path.exists(DEFAULT_COOKIE_FILE) else None)
    if active_cookie_file and os.path.exists(active_cookie_file):
        print(f"[Crawler] Authenticated via cookies: {active_cookie_file}")

    ydl_opts = {
        "extract_flat": "in_playlist",
        "quiet": True,
        "no_warnings": True,
        "playlist_items": f"1-{limit}" if limit > 0 else None,
    }
    if active_cookie_file and os.path.exists(active_cookie_file):
        ydl_opts["cookiefile"] = active_cookie_file

    results = {
        "creator": creator_name or "Unknown",
        "total_scraped": 0,
        "by_game": {},
        "errors": [],
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            info = ydl.extract_info(target, download=False)
        except Exception as e:
            err_msg = f"Failed to extract info from {target}: {str(e)}"
            print(f"[Crawler Error] {err_msg}")
            results["errors"].append(err_msg)
            return results

    # Determine creator folder name
    channel = creator_name or info.get("channel") or info.get("uploader") or "Creator"
    creator_folder = sanitize_folder_name(channel)
    results["creator"] = creator_folder

    entries = info.get("entries", [])
    if not entries:
        if "id" in info and "title" in info:
            entries = [info]

    print(f"[Crawler] Found {len(entries)} videos for creator: {creator_folder}\n")

    lock = threading.Lock()

    def process_single(item):
        idx, entry = item
        video_id = entry.get("id")
        title = entry.get("title") or f"video_{video_id}"
        if not video_id:
            return None

        # Determine game category
        game_name = game_override or detect_game_from_title(title)
        game_folder = sanitize_folder_name(game_name)

        # Folder structure: <output_base_dir>/<creator_folder>/<game_folder>/thumbnails/
        game_dir = os.path.join(output_base_dir, creator_folder, game_folder)
        thumb_dir = os.path.join(game_dir, "thumbnails")
        caption_dir = os.path.join(game_dir, "captions")
        meta_dir = os.path.join(game_dir, "metadata")

        with lock:
            os.makedirs(thumb_dir, exist_ok=True)
            os.makedirs(caption_dir, exist_ok=True)
            os.makedirs(meta_dir, exist_ok=True)

            existing_ids = get_existing_video_ids(meta_dir)
            if video_id in existing_ids:
                return ("skip", title, game_folder, idx)

            prefix = f"{creator_folder}_{game_folder}"
            current_index = get_next_index(thumb_dir, prefix)
            base_filename = f"{prefix}_{current_index}"

            # Pre-reserve placeholder file so other threads get current_index + 1
            thumb_file = os.path.join(thumb_dir, f"{base_filename}.jpg")
            open(thumb_file, "a").close()

        # Concurrent download & description scrape outside lock
        success = download_thumbnail(video_id, thumb_file)
        if not success:
            with lock:
                if os.path.exists(thumb_file) and os.path.getsize(thumb_file) == 0:
                    os.remove(thumb_file)
            return ("fail", title, game_folder, idx)

        description = get_video_description(video_id, cookie_file=active_cookie_file)
        caption_text = generate_training_caption(title, description, creator_folder, game_folder)

        caption_file = os.path.join(caption_dir, f"{base_filename}.txt")
        meta_file = os.path.join(meta_dir, f"{base_filename}.json")

        with open(caption_file, "w", encoding="utf-8") as f:
            f.write(caption_text)

        metadata = {
            "file_name": f"{base_filename}.jpg",
            "id": video_id,
            "title": title,
            "description": description,
            "creator": creator_folder,
            "game": game_folder,
            "index": current_index,
            "url": f"https://www.youtube.com/watch?v={video_id}",
            "caption": caption_text,
            "view_count": entry.get("view_count"),
            "duration": entry.get("duration"),
            "upload_date": entry.get("upload_date"),
        }
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        with lock:
            results["total_scraped"] += 1
            results["by_game"][game_folder] = results["by_game"].get(game_folder, 0) + 1

        return ("ok", base_filename, game_folder, title, idx)

    print(f"[Crawler] Downloading with 8 concurrent worker threads...")
    items = list(enumerate(entries, start=1))
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(process_single, item): item for item in items}
        for future in as_completed(futures):
            res = future.result()
            if not res:
                continue
            status = res[0]
            if status == "ok":
                _, base_filename, game_folder, title, idx = res
                print(f"[{idx}/{len(entries)}] [OK] Saved {base_filename}.jpg -> [{game_folder}] {title[:38]}...")
            elif status == "skip":
                _, title, game_folder, idx = res
                print(f"[{idx}/{len(entries)}] [SKIP (Already Scraped)] {title[:40]}...")
            elif status == "fail":
                _, title, game_folder, idx = res
                print(f"[{idx}/{len(entries)}] [SKIP] Thumbnail unavailable: {title[:40]}...")

    print(f"\n[Crawler Finished] Successfully scraped {results['total_scraped']} thumbnails.")
    print("Breakdown by Game:")
    for g, count in results["by_game"].items():
        print(f"  • {g}: {count} thumbnails")

    return results


def main():
    parser = argparse.ArgumentParser(description="Tràuna AI — YouTube Thumbnail & Dataset Crawler")
    parser.add_argument("--channel", type=str, help="YouTube channel URL (e.g. https://www.youtube.com/@BeastBoyShub/videos)")
    parser.add_argument("--playlist", type=str, help="YouTube playlist URL")
    parser.add_argument("--search", type=str, help="YouTube search query (e.g. 'BeastBoyShub Minecraft')")
    parser.add_argument("--creator", type=str, default=None, help="Creator / YouTuber name override")
    parser.add_argument("--game", type=str, default=None, help="Game name override (skips auto-detection)")
    parser.add_argument("--limit", type=int, default=30, help="Max videos to scrape (default: 30)")
    parser.add_argument("--output", type=str, default=DEFAULT_DATASET_DIR, help="Base dataset output folder")
    parser.add_argument("--cookies", type=str, default=None, help="Path to Netscape cookies.txt file")

    args = parser.parse_args()

    target = None
    if args.channel:
        target = args.channel
        if not target.endswith("/videos") and "@" in target:
            target = target.rstrip("/") + "/videos"
    elif args.playlist:
        target = args.playlist
    elif args.search:
        target = f"ytsearch{args.limit}:{args.search}"
    else:
        # Default convenience shortcut for BeastBoyShub
        print("No target specified. Using default: BeastBoyShub channel")
        target = "https://www.youtube.com/@BeastBoyShub/videos"
        if not args.creator:
            args.creator = "BeastBoyShub"

    crawl(
        target=target,
        creator_name=args.creator,
        game_override=args.game,
        limit=args.limit,
        output_base_dir=args.output,
        cookie_file=args.cookies,
    )


if __name__ == "__main__":
    main()
