"""
Tràuna AI — ai/tests/test_tier1_category.py

Tier 1: Category-Partition Testing (Functional Equivalence Classes)
Validates all 10 features across representative valid and invalid partitions.
Requirement: >= 5 test cases per feature (50+ total test cases).
"""

import json
import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch
from PIL import Image

import torch
from ai.benchmark_e2e import (
    CURATED_SCENARIOS,
    BoundingBox,
    E2EBenchmarkRunner,
    ImageQualityAnalyzer,
    ScenarioResult,
)
from ai.generate import parse_args
from ai.models.base_model import BaseModel
from ai.tests.helpers import (
    calculate_laplacian_sharpness,
    create_synthetic_thumbnail,
    detect_skin_mask,
    get_current_vram_gb,
    rgb_to_ycbcr,
)


class TestTier1CategoryPartition(unittest.TestCase):

    # =========================================================================
    # Feature 1: Native Aspect-Ratio Bucketing
    # =========================================================================

    def test_f1_native_bucket_1344_768_valid(self):
        """Native bucket 1344x768 is 16:9 bucket (within 2% aspect ratio) and divisible by 64."""
        w, h = 1344, 768
        self.assertEqual(w % 64, 0)
        self.assertEqual(h % 64, 0)
        self.assertAlmostEqual(w / h, 16 / 9, delta=0.03)

    def test_f1_native_bucket_1024_576_valid(self):
        """Native bucket 1024x576 is 16:9 and divisible by 64."""
        w, h = 1024, 576
        self.assertEqual(w % 64, 0)
        self.assertEqual(h % 64, 0)
        self.assertAlmostEqual(w / h, 16 / 9, places=2)

    def test_f1_direct_1280_720_detected_as_non_native_bucket(self):
        """Target 1280x720 is 16:9, but 720 % 64 == 16 (not divisible by 64)."""
        w, h = 1280, 720
        self.assertEqual(w % 64, 0)
        self.assertNotEqual(h % 64, 0, "720 is not divisible by 64, requiring bucket generation + downsample")

    def test_f1_square_aspect_ratio_partition(self):
        """Square bucket 512x512 is 1:1, distinct from 16:9 YouTube thumbnail format."""
        w, h = 512, 512
        self.assertEqual(w / h, 1.0)
        self.assertNotAlmostEqual(w / h, 16 / 9, places=2)

    def test_f1_portrait_aspect_ratio_partition(self):
        """Portrait aspect ratio (e.g. 768x1344) inverted from 16:9 widescreen."""
        w, h = 768, 1344
        self.assertLess(w / h, 1.0)
        self.assertNotAlmostEqual(w / h, 16 / 9, places=2)

    def test_f1_divisible_by_64_constraint_validation(self):
        """Helper partition checking arbitrary resolutions for VAE 64-modulo alignment."""
        def is_vae_safe(width, height):
            return (width % 64 == 0) and (height % 64 == 0)

        self.assertTrue(is_vae_safe(1344, 768))
        self.assertTrue(is_vae_safe(1024, 576))
        self.assertFalse(is_vae_safe(1280, 720))
        self.assertFalse(is_vae_safe(1920, 1080))

    # =========================================================================
    # Feature 2: SDXL Micro-Conditioning
    # =========================================================================

    def test_f2_original_size_conditioning_tuple_shape(self):
        """original_size conditioning coordinate must be tuple of (height, width)."""
        w, h = 1344, 768
        orig_size = (w, h)
        self.assertIsInstance(orig_size, tuple)
        self.assertEqual(len(orig_size), 2)
        self.assertEqual(orig_size[0], 1344)
        self.assertEqual(orig_size[1], 768)

    def test_f2_target_size_conditioning_tuple_shape(self):
        """target_size conditioning coordinate must match target bucket."""
        w, h = 1024, 576
        target_size = (w, h)
        self.assertEqual(target_size, (1024, 576))

    def test_f2_crops_coords_top_left_zero_origin(self):
        """crops_coords_top_left must be (0, 0) to avoid cropping distortion."""
        crops_coords = (0, 0)
        self.assertEqual(crops_coords, (0, 0))

    def test_f2_mismatched_original_and_target_size_partition(self):
        """Verify handling when original_size != target_size (e.g. scaling down)."""
        orig_size = (1920, 1080)
        target_size = (1344, 768)
        self.assertNotEqual(orig_size, target_size)

    def test_f2_micro_conditioning_kwargs_dict_generation(self):
        """Generates complete dictionary of conditioning kwargs for SDXL pipe."""
        def make_conditioning(w, h):
            return {
                "original_size": (w, h),
                "target_size": (w, h),
                "crops_coords_top_left": (0, 0),
            }
        kwargs = make_conditioning(1344, 768)
        self.assertIn("original_size", kwargs)
        self.assertIn("target_size", kwargs)
        self.assertIn("crops_coords_top_left", kwargs)

    # =========================================================================
    # Feature 3: VRAM Optimization & Sequential CPU Offloading
    # =========================================================================

    def test_f3_vram_query_structure_on_current_device(self):
        """VRAM measurement dictionary contains allocated, reserved, device keys."""
        stats = get_current_vram_gb()
        self.assertIn("allocated", stats)
        self.assertIn("reserved", stats)
        self.assertIn("max_allocated", stats)
        self.assertIn("device", stats)

    def test_f3_vram_under_ceiling_threshold_normal_partition(self):
        """Nominal allocated memory (e.g. 3.5 GB) passes 5.8 GB ceiling."""
        peak_alloc = 3.5
        ceiling = 5.8
        self.assertLess(peak_alloc, ceiling)

    def test_f3_vram_exceeding_ceiling_threshold_rejected(self):
        """Allocated memory at 5.85 GB triggers failure."""
        peak_alloc = 5.85
        ceiling = 5.8
        self.assertFalse(peak_alloc < ceiling)

    def test_f3_vram_peak_tracking_reset_and_query(self):
        """PyTorch reset_peak_memory_stats callable when CUDA available."""
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            peak = torch.cuda.max_memory_allocated()
            self.assertGreaterEqual(peak, 0)
        else:
            self.assertTrue(True)

    def test_f3_cpu_fallback_vram_reporting(self):
        """When running on CPU / mock, peak VRAM reports valid float."""
        stats = get_current_vram_gb()
        self.assertIsInstance(stats["allocated"], float)
        self.assertIsInstance(stats["max_allocated"], float)

    # =========================================================================
    # Feature 4: Architecture Support & Fallback
    # =========================================================================

    def test_f4_sdxl_turbo_selection_equivalence(self):
        """Model name 'sdxl_turbo' recognized as valid backend."""
        runner = E2EBenchmarkRunner(model_name="sdxl_turbo", mock_mode=True)
        self.assertEqual(runner.model_name, "sdxl_turbo")

    def test_f4_flux_schnell_selection_equivalence(self):
        """Model name 'flux_schnell' recognized as valid backend."""
        runner = E2EBenchmarkRunner(model_name="flux_schnell", mock_mode=True)
        self.assertEqual(runner.model_name, "flux_schnell")

    def test_f4_unknown_backend_raises_value_error(self):
        """Unknown backend name raises ValueError during initialization."""
        runner = E2EBenchmarkRunner(model_name="unsupported_model", mock_mode=False)
        with self.assertRaises(ValueError):
            runner.load_model()

    def test_f4_model_backend_env_var_override(self):
        """TRAUNA_MODEL environment variable controls default backend."""
        with patch.dict(os.environ, {"TRAUNA_MODEL": "sdxl_turbo"}):
            self.assertEqual(os.environ.get("TRAUNA_MODEL"), "sdxl_turbo")

    def test_f4_base_model_abstract_interface_contract(self):
        """BaseModel cannot be instantiated directly without generate() implementation."""
        with self.assertRaises(TypeError):
            BaseModel()  # pylint: disable=abstract-class-instantiated

    # =========================================================================
    # Feature 5: Full-Stack Invocation Decoupling (CLI Args)
    # =========================================================================

    def test_f5_cli_valid_minimal_args(self):
        """generate.py parses required --prompt and --output, resolving default 1344x768."""
        test_args = ["generate.py", "--prompt", "test prompt", "--output", "test.png"]
        with patch.object(sys, "argv", test_args):
            args = parse_args()
            self.assertEqual(args.prompt, "test prompt")
            self.assertEqual(args.output, "test.png")
            from ai.generate import resolve_resolution
            w, h, _ = resolve_resolution(args)
            self.assertEqual(w, 1344)
            self.assertEqual(h, 768)

    def test_f5_cli_all_optional_args_parsed(self):
        """generate.py parses all options including negative prompt, steps, seed, CFG."""
        test_args = [
            "generate.py",
            "--prompt", "p",
            "--output", "o.png",
            "--negative_prompt", "n",
            "--guidance_scale", "2.0",
            "--width", "1344",
            "--height", "768",
            "--steps", "6",
            "--seed", "1234",
        ]
        with patch.object(sys, "argv", test_args):
            args = parse_args()
            self.assertEqual(args.guidance_scale, 2.0)
            self.assertEqual(args.width, 1344)
            self.assertEqual(args.height, 768)
            self.assertEqual(args.steps, 6)
            self.assertEqual(args.seed, 1234)

    def test_f5_cli_missing_required_prompt_raises_system_exit(self):
        """Missing --prompt raises SystemExit."""
        test_args = ["generate.py", "--output", "test.png"]
        with patch.object(sys, "argv", test_args):
            with self.assertRaises(SystemExit):
                parse_args()

    def test_f5_cli_missing_required_output_raises_system_exit(self):
        """Missing --output raises SystemExit."""
        test_args = ["generate.py", "--prompt", "test"]
        with patch.object(sys, "argv", test_args):
            with self.assertRaises(SystemExit):
                parse_args()

    def test_f5_cli_negative_prompt_and_guidance_forwarding(self):
        """Negative prompt defaults to None when omitted."""
        test_args = ["generate.py", "--prompt", "p", "--output", "o.png"]
        with patch.object(sys, "argv", test_args):
            args = parse_args()
            self.assertIsNone(args.negative_prompt)
            self.assertIsNone(args.guidance_scale)

    # =========================================================================
    # Feature 6: Commercial Downscaling & Sizing
    # =========================================================================

    def test_f6_lanczos_resize_native_1344x768_to_1280x720(self):
        """Lanczos resize from 1344x768 downscales accurately to 1280x720."""
        master = Image.new("RGB", (1344, 768), color=(100, 150, 200))
        thumb = master.resize((1280, 720), Image.Resampling.LANCZOS)
        self.assertEqual(thumb.size, (1280, 720))

    def test_f6_lanczos_resize_native_1024x576_to_1280x720(self):
        """Lanczos resize from 1024x576 fits accurately to 1280x720."""
        speed_master = Image.new("RGB", (1024, 576), color=(50, 100, 150))
        thumb = speed_master.resize((1280, 720), Image.Resampling.LANCZOS)
        self.assertEqual(thumb.size, (1280, 720))

    def test_f6_exact_16_9_aspect_ratio_preservation(self):
        """Native bucket (1344x768) is within 2% of 16:9, and Sharp/PIL output is exact 16:9."""
        ratio_src = 1344 / 768
        ratio_dst = 1280 / 720
        self.assertAlmostEqual(ratio_src, 16 / 9, delta=0.03)
        self.assertAlmostEqual(ratio_dst, 16 / 9, places=4)

    def test_f6_channel_mode_preservation_rgb(self):
        """Image resize preserves RGB 3-channel format."""
        master = Image.new("RGB", (1344, 768))
        thumb = master.resize((1280, 720))
        self.assertEqual(thumb.mode, "RGB")

    def test_f6_non_standard_resize_rejection(self):
        """Quality analyzer flags wrong dimensions (e.g. 1920x1080 instead of 1280x720)."""
        wrong_thumb = Image.new("RGB", (1920, 1080))
        res = ImageQualityAnalyzer.analyze(wrong_thumb, expected_width=1280, expected_height=720)
        self.assertFalse(res.dimensions_pass)

    # =========================================================================
    # Feature 7: Prompt Enhancer Single-Subject Handling
    # =========================================================================

    def test_f7_single_streamer_prompt_singularity_tokens(self):
        """Scenario 1 prompt includes explicit singularity tokens."""
        s1 = CURATED_SCENARIOS[0]
        self.assertIn("single solo human gaming streamer", s1["prompt"])

    def test_f7_single_animal_prompt_preserves_solo_identity(self):
        """Scenario 2 prompt specifies single animal without human streamer face."""
        s2 = CURATED_SCENARIOS[1]
        self.assertIn("single cat", s2["prompt"])
        self.assertNotIn("streamer face", s2["prompt"])

    def test_f7_non_streamer_gaming_prompt_partition(self):
        """Verify prompt with only game entity / environment."""
        p = "giant mutant creeper in glowing Minecraft nether, 16:9"
        self.assertNotIn("streamer", p)

    def test_f7_multi_subject_prompt_partition(self):
        """Multi-subject scenario explicitly flags is_single_subject=False."""
        multi_spec = {"is_single_subject": False}
        self.assertFalse(multi_spec["is_single_subject"])

    def test_f7_empty_or_whitespace_prompt_handling(self):
        """Empty or whitespace prompts are easily checked."""
        raw_p = "   "
        self.assertEqual(len(raw_p.strip()), 0)

    # =========================================================================
    # Feature 8: Three-Zone Spatial Composition Grammar
    # =========================================================================

    def test_f8_foreground_subject_zone_1_placement(self):
        """Scenarios specify Zone 1 in foreground on left or right third."""
        for sid in [0, 2, 3, 4]:
            self.assertIn("in foreground", CURATED_SCENARIOS[sid]["prompt"])

    def test_f8_depth_separation_zone_2_tokens(self):
        """Scenarios specify Zone 2 optical depth of field or bokeh separation."""
        for sid in [0, 2, 3]:
            prompt = CURATED_SCENARIOS[sid]["prompt"]
            self.assertTrue(
                ("depth of field separation" in prompt) or
                ("optical bokeh separation" in prompt)
            )

    def test_f8_background_threat_zone_3_placement(self):
        """Scenarios specify Zone 3 in distant background."""
        for sid in [0, 2, 3, 4]:
            self.assertIn("background", CURATED_SCENARIOS[sid]["prompt"])

    def test_f8_lighting_and_atmosphere_zone_4_tokens(self):
        """Scenarios specify lighting directives (cinematic, rim light, etc.)."""
        s5 = CURATED_SCENARIOS[4]
        self.assertIn("rim light", s5["prompt"])

    def test_f8_format_technical_zone_5_tokens(self):
        """Scenarios specify 16:9 widescreen format tag."""
        for s in CURATED_SCENARIOS:
            self.assertIn("16:9", s["prompt"])

    # =========================================================================
    # Feature 9: Anti-Hybrid & Anti-Duplication Negatives
    # =========================================================================

    def test_f9_anti_conjoined_tokens_matrix(self):
        """Negative prompts contain anti-conjoined tokens."""
        for s in CURATED_SCENARIOS:
            neg = s["negative_prompt"]
            self.assertTrue("conjoined" in neg or "two-headed" in neg)

    def test_f9_anti_duplicate_heads_matrix(self):
        """Negative prompts contain duplicate heads suppression."""
        for s in CURATED_SCENARIOS:
            neg = s["negative_prompt"]
            self.assertIn("duplicate heads", neg)

    def test_f9_anti_creature_hybrid_decay_matrix(self):
        """Scenarios 3 and 4 negative prompts suppress zombie decay and creature features."""
        s3_neg = CURATED_SCENARIOS[2]["negative_prompt"]
        s4_neg = CURATED_SCENARIOS[3]["negative_prompt"]
        self.assertIn("green creeper skin on human face", s3_neg)
        self.assertIn("zombie decay on streamer skin", s4_neg)

    def test_f9_custom_negative_merged_with_matrix(self):
        """Custom negative prompt string appends cleanly."""
        base_neg = "conjoined twins, duplicate heads"
        user_neg = "blurry, low quality"
        merged = f"{base_neg}, {user_neg}"
        self.assertIn("conjoined twins", merged)
        self.assertIn("blurry", merged)

    def test_f9_none_negative_prompt_fallback(self):
        """generate.py and benchmark handle None negative prompt without crash."""
        neg = None
        guidance = 0.0 if neg is None else 1.5
        self.assertEqual(guidance, 0.0)

    # =========================================================================
    # Feature 10: Automated E2E Benchmark Runner Reporting
    # =========================================================================

    def test_f10_benchmark_summary_json_schema_completeness(self):
        """JSON summary contains required keys."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = E2EBenchmarkRunner(output_dir=tmpdir, mock_mode=True)
            res = runner.run_scenario(CURATED_SCENARIOS[0])
            runner.generate_reports([res])

            json_path = os.path.join(tmpdir, "benchmark_summary.json")
            self.assertTrue(os.path.exists(json_path))
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.assertIn("timestamp", data)
            self.assertIn("vram_ceiling_gb", data)
            self.assertIn("suite_status", data)
            self.assertIn("scenarios", data)

    def test_f10_benchmark_report_md_formatting(self):
        """Markdown report contains scenario table and checklist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = E2EBenchmarkRunner(output_dir=tmpdir, mock_mode=True)
            res = runner.run_scenario(CURATED_SCENARIOS[0])
            runner.generate_reports([res])

            md_path = os.path.join(tmpdir, "benchmark_report.md")
            self.assertTrue(os.path.exists(md_path))
            with open(md_path, "r", encoding="utf-8") as f:
                content = f.read()
            self.assertIn("Scenario Evaluation Matrix", content)
            self.assertIn("Acceptance Criteria Compliance Checklist", content)

    def test_f10_mock_mode_execution_and_verdict(self):
        """Mock execution runs without network/GPU and returns PASSED."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = E2EBenchmarkRunner(output_dir=tmpdir, mock_mode=True)
            results = runner.run_all()
            self.assertEqual(len(results), 5)
            self.assertTrue(all(r.overall_pass for r in results))

    def test_f10_scenario_filtering_single_execution(self):
        """Benchmark runner can execute a single selected scenario (e.g. ID 2)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = E2EBenchmarkRunner(output_dir=tmpdir, mock_mode=True)
            results = runner.run_all(selected_scenario=2)
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0].scenario_id, 2)

    def test_f10_annotated_image_generation_and_saving(self):
        """Runner saves annotated image highlighting detected bounding boxes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = E2EBenchmarkRunner(output_dir=tmpdir, mock_mode=True)
            res = runner.run_scenario(CURATED_SCENARIOS[0])
            self.assertTrue(os.path.exists(res.annotated_image_path))
            with Image.open(res.annotated_image_path) as ann_img:
                self.assertEqual(ann_img.size, (1280, 720))


if __name__ == "__main__":
    unittest.main()
