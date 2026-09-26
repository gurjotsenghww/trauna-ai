#!/usr/bin/env python3
"""
Tràuna AI — test_generation.py

Runs one image generation to verify the complete AI pipeline.
Do not proceed to application development until this test passes.

Usage:
  python test_generation.py
  TRAUNA_MODEL=sdxl_turbo python test_generation.py
"""

import sys
import os
import time

TEST_PROMPT = (
    'YouTube gaming thumbnail, a gaming streamer with an excited expression, '
    'Minecraft blocky world background, dramatic lighting, vibrant colours, '
    'high contrast, professional YouTube thumbnail composition, 16:9 aspect ratio'
)

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), 'outputs', 'test_output.png')


def main():
    print('=' * 60)
    print('  Tràuna AI — Generation Test')
    print('=' * 60)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    model_name = os.environ.get('TRAUNA_MODEL', 'flux_schnell')
    print(f'\nModel backend  : {model_name}')
    print(f'Test prompt    : {TEST_PROMPT[:60]}...')
    print(f'Output         : {OUTPUT_PATH}')
    print()

    try:
        if model_name == 'flux_schnell':
            from models.flux_schnell import FluxSchnellModel
            model = FluxSchnellModel()
        elif model_name == 'sdxl_turbo':
            from models.sdxl_turbo import SDXLTurboModel
            model = SDXLTurboModel()
        else:
            print(f'[ERROR] Unknown model: {model_name}')
            sys.exit(1)

        print('Generating image...')
        t0 = time.time()

        image = model.generate(
            prompt=TEST_PROMPT,
            width=1024,
            height=576,
            num_inference_steps=4,
            seed=42,
        )

        image.save(OUTPUT_PATH)
        elapsed = time.time() - t0

        print(f'\n[SUCCESS] Image saved to: {OUTPUT_PATH}')
        print(f'[SUCCESS] Generation time: {elapsed:.1f}s')
        print(f'[SUCCESS] Image size: {image.width}x{image.height}')
        print('\nTest passed. You can proceed to application development.')

    except Exception as e:
        print(f'\n[FAILED] {e}')
        import traceback
        traceback.print_exc()
        print('\nIf this is an OOM error, try: TRAUNA_MODEL=sdxl_turbo python test_generation.py')
        sys.exit(1)


if __name__ == '__main__':
    main()
