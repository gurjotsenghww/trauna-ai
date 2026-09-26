"""
Tràuna AI — ai/tests/test_tier3_pairwise.py

Tier 3: Pairwise Combinatorial & Interface Contract Testing
Tests interactions between (Model Backend × Resolution Bucket × Guidance Scale × Negative Prompt × Steps)
and validates interface contracts specified in PROJECT.md.
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch
from PIL import Image

from ai.benchmark_e2e import (
    CURATED_SCENARIOS,
    BoundingBox,
    E2EBenchmarkRunner,
    ImageQualityAnalyzer,
    ScenarioResult,
)
from ai.generate import parse_args, resolve_resolution
from ai.tests.helpers import create_synthetic_thumbnail


class TestTier3Pairwise(unittest.TestCase):

    # =========================================================================
    # Pairwise Combinations
    # =========================================================================

    def test_pairwise_comb_1_sdxl_standard_bucket_cfg15_neg_steps4(self):
        """Pair 1 (Production Master): SDXL Turbo × 1344x768 × CFG 1.5 × Neg Prompt Active × Steps 4."""
        test_args = [
            "generate.py",
            "--prompt", "streamer reaction face",
            "--output", "test_p1.png",
            "--bucket", "standard",
            "--guidance_scale", "1.5",
            "--negative_prompt", "conjoined twins, duplicate heads",
            "--steps", "4",
        ]
        with patch.object(sys, "argv", test_args):
            args = parse_args()
            w, h, label = resolve_resolution(args)
            self.assertEqual(w, 1344)
            self.assertEqual(h, 768)
            self.assertEqual(args.steps, 4)
            self.assertEqual(args.guidance_scale, 1.5)
            self.assertIn("conjoined", args.negative_prompt)

    def test_pairwise_comb_2_sdxl_fast_bucket_cfg0_noneg_steps1(self):
        """Pair 2 (Fast Lightning): SDXL Turbo × 1024x576 × CFG 0.0 × No Neg Prompt × Step 1."""
        test_args = [
            "generate.py",
            "--prompt", "fast gaming thumb",
            "--output", "test_p2.png",
            "--bucket", "fast",
            "--guidance_scale", "0.0",
            "--steps", "1",
        ]
        with patch.object(sys, "argv", test_args):
            args = parse_args()
            w, h, label = resolve_resolution(args)
            self.assertEqual(w, 1024)
            self.assertEqual(h, 576)
            self.assertEqual(args.steps, 1)
            self.assertEqual(args.guidance_scale, 0.0)
            self.assertIsNone(args.negative_prompt)

    def test_pairwise_comb_3_sdxl_standard_bucket_cfg20_neg_steps6(self):
        """Pair 3 (Ultra Quality): SDXL Turbo × 1344x768 × CFG 2.0 × Neg Prompt Active × Steps 6."""
        test_args = [
            "generate.py",
            "--prompt", "ultra thumbnail",
            "--output", "test_p3.png",
            "--bucket", "standard",
            "--guidance_scale", "2.0",
            "--negative_prompt", "blurry, low quality",
            "--steps", "6",
        ]
        with patch.object(sys, "argv", test_args):
            args = parse_args()
            w, h, _ = resolve_resolution(args)
            self.assertEqual(w, 1344)
            self.assertEqual(h, 768)
            self.assertEqual(args.steps, 6)
            self.assertEqual(args.guidance_scale, 2.0)

    def test_pairwise_comb_4_flux_schnell_fast_bucket_cfg0_steps4(self):
        """Pair 4 (FLUX Schnell): FLUX × 1024x576 × CFG 0.0 × Step 4."""
        test_args = [
            "generate.py",
            "--prompt", "flux gaming thumbnail",
            "--output", "test_p4.png",
            "--model", "flux_schnell",
            "--bucket", "fast",
            "--steps", "4",
        ]
        with patch.object(sys, "argv", test_args):
            args = parse_args()
            self.assertEqual(args.model, "flux_schnell")
            w, h, _ = resolve_resolution(args)
            self.assertEqual(w, 1024)
            self.assertEqual(h, 576)
            self.assertEqual(args.steps, 4)

    def test_pairwise_comb_5_custom_resolution_cfg15_seed42(self):
        """Pair 5 (Custom Resolution with Seed): Custom 1280x720 × CFG 1.5 × Seed 42."""
        test_args = [
            "generate.py",
            "--prompt", "custom res",
            "--output", "test_p5.png",
            "--width", "1280",
            "--height", "720",
            "--guidance_scale", "1.5",
            "--seed", "42",
        ]
        with patch.object(sys, "argv", test_args):
            args = parse_args()
            w, h, _ = resolve_resolution(args)
            self.assertEqual(w, 1280)
            self.assertEqual(h, 720)
            self.assertEqual(args.seed, 42)

    # =========================================================================
    # Interface Contracts (PROJECT.md)
    # =========================================================================

    def test_contract_ai_service_to_python_cli(self):
        """
        Contract: backend/services/aiService.js -> ai/generate.py
        Requires: --prompt, --output, --width, --height, --steps, --negative_prompt, --guidance_scale, --seed.
        """
        cli_flags = [
            "--prompt", "test",
            "--output", "out.png",
            "--width", "1344",
            "--height", "768",
            "--steps", "4",
            "--negative_prompt", "bad anatomy",
            "--guidance_scale", "1.5",
            "--seed", "99",
        ]
        with patch.object(sys, "argv", ["generate.py"] + cli_flags):
            args = parse_args()
            self.assertEqual(args.prompt, "test")
            self.assertEqual(args.output, "out.png")
            self.assertEqual(args.width, 1344)
            self.assertEqual(args.height, 768)
            self.assertEqual(args.steps, 4)
            self.assertEqual(args.negative_prompt, "bad anatomy")
            self.assertEqual(args.guidance_scale, 1.5)
            self.assertEqual(args.seed, 99)

    def test_contract_generate_to_sdxl_turbo_signature(self):
        """
        Contract: ai/generate.py -> ai/models/sdxl_turbo.py
        Signature: generate(prompt, width, height, num_inference_steps, guidance_scale, negative_prompt, seed)
        """
        from ai.models.base_model import BaseModel
        # Verify abstract method parameter names
        import inspect
        sig = inspect.signature(BaseModel.generate)
        params = list(sig.parameters.keys())
        expected_params = ["self", "prompt", "width", "height", "num_inference_steps", "guidance_scale", "negative_prompt", "seed"]
        self.assertEqual(params, expected_params)

    def test_contract_composite_downsampling_pipeline(self):
        """
        Contract: Native bucket (1344x768) -> Sharp composite -> 1280x720.
        Verifies clean Lanczos downsampling without pixel distortion.
        """
        native = create_synthetic_thumbnail(1344, 768, face_count=1)
        downsampled = native.resize((1280, 720), Image.Resampling.LANCZOS)
        self.assertEqual(downsampled.size, (1280, 720))

        analysis = ImageQualityAnalyzer.analyze(downsampled, expected_width=1280, expected_height=720)
        self.assertTrue(analysis.dimensions_pass)
        self.assertTrue(analysis.subject_count_pass)


if __name__ == "__main__":
    unittest.main()
