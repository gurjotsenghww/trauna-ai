"""Generate orientation fix images 2 and 3 (horror + GTA). Image 1 already saved."""
import sys, os, time
sys.path.insert(0, os.path.dirname(__file__))

ARTIFACT_DIR = r"C:\Users\Gurjotpal Singh\.gemini\antigravity\brain\17c0e50d-0f61-4094-8fdc-7395fdfb8ea7"

TEST_CASES = [
    {
        "name": "horror_facing_fix",
        "label": "Horror (front-facing override)",
        "prompt": "traunathumb, youtube thumbnail, solo gaming streamer screaming terrified face, front-facing, looking at camera on the right, dark terrifying abandoned corridor with fog and eerie shadows on the left, horror flashlight rim lighting, cinematic lighting, 16:9",
        "seed": 44,
    },
    {
        "name": "gta_facing_fix",
        "label": "GTA Los Santos (single subject fix)",
        "prompt": "traunathumb, youtube thumbnail, solo screaming shocked gaming streamer with LED headset, front-facing, looking at camera on the right, Los Santos city street with neon lights and sports car on the left, neon night city rim lights, cinematic lighting, 16:9",
        "seed": 45,
    },
]

from models.sdxl_turbo import SDXLTurboModel
from PIL import Image

print("[Test] Loading model...", flush=True)
model = SDXLTurboModel()
print("[Test] Model ready.\n", flush=True)

for i, case in enumerate(TEST_CASES, 2):
    print(f"[Test {i}/3] {case['label']}", flush=True)
    print(f"  Prompt: {case['prompt'][:90]}...", flush=True)
    t0 = time.time()
    img = model.generate(
        prompt=case["prompt"],
        width=1024,
        height=576,
        num_inference_steps=4,
        guidance_scale=0.0,
        seed=case["seed"],
    )
    img_hd = img.resize((1280, 720), Image.LANCZOS)
    out_path = os.path.join(ARTIFACT_DIR, f"orient_fix_{case['name']}.png")
    img_hd.save(out_path)
    print(f"  [OK] Saved ({time.time()-t0:.1f}s): {out_path}\n", flush=True)

print("[Test] Done! All 3 orientation fix images are ready.", flush=True)
