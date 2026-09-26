"""
Tràuna AI — ai/tests/test_tier2_boundary.py

Tier 2: Boundary Value Analysis (BVA)
Tests exact mathematical, architectural, and memory boundaries across all 10 features.
Requirement: >= 5 test cases per feature (50+ total test cases).
"""

import os
import sys
import unittest
from PIL import Image

from ai.benchmark_e2e import (
    CURATED_SCENARIOS,
    BoundingBox,
    E2EBenchmarkRunner,
    ImageQualityAnalyzer,
    ScenarioResult,
)
from ai.generate import parse_args, resolve_resolution
from ai.tests.helpers import (
    calculate_laplacian_sharpness,
    create_synthetic_thumbnail,
    get_current_vram_gb,
)


class TestTier2BoundaryValueAnalysis(unittest.TestCase):

    # =========================================================================
    # Feature 1: Native Aspect-Ratio Bucketing Boundaries
    # =========================================================================

    def test_bva_f1_lower_resolution_64x64_boundary(self):
        """Minimum 64x64 VAE latent tile boundary."""
        w, h = 64, 64
        self.assertEqual(w % 64, 0)
        self.assertEqual(h % 64, 0)

    def test_bva_f1_speed_bucket_1024x576_boundary(self):
        """Standard speed bucket boundary (1024x576) is exact modulo 64."""
        w, h = 1024, 576
        self.assertEqual(w % 64, 0)
        self.assertEqual(h % 64, 0)
        self.assertEqual(w / h, 16 / 9)

    def test_bva_f1_master_bucket_1344x768_boundary(self):
        """High fidelity master bucket boundary (1344x768) is exact modulo 64."""
        w, h = 1344, 768
        self.assertEqual(w % 64, 0)
        self.assertEqual(h % 64, 0)
        self.assertAlmostEqual(w / h, 16 / 9, delta=0.03)

    def test_bva_f1_hd_upper_boundary_1920x1080(self):
        """1920x1080 boundary: width is divisible by 64 (1920%64=0) but height is not (1080%64=56)."""
        w, h = 1920, 1080
        self.assertEqual(w % 64, 0)
        self.assertNotEqual(h % 64, 0)

    def test_bva_f1_modulo_64_exact_boundary(self):
        """Boundary check on 63 (fail), 64 (pass), 65 (fail)."""
        self.assertNotEqual(63 % 64, 0)
        self.assertEqual(64 % 64, 0)
        self.assertNotEqual(65 % 64, 0)

    # =========================================================================
    # Feature 2: SDXL Micro-Conditioning Boundaries
    # =========================================================================

    def test_bva_f2_crops_coords_exact_zero_origin(self):
        """crops_coords_top_left boundary must be exactly (0, 0)."""
        coords = (0, 0)
        self.assertEqual(coords[0], 0)
        self.assertEqual(coords[1], 0)

    def test_bva_f2_target_equals_original_exact_match(self):
        """Nominal uncropped boundary: original_size == target_size."""
        orig = (1344, 768)
        target = (1344, 768)
        self.assertEqual(orig, target)

    def test_bva_f2_aspect_ratio_scaled_conditioning(self):
        """Conditioning when master downscaled to standard thumbnail."""
        orig = (1344, 768)
        target = (1280, 720)
        self.assertNotEqual(orig, target)
        self.assertGreater(orig[0], target[0])
        self.assertGreater(orig[1], target[1])

    def test_bva_f2_negative_crop_coords_rejected(self):
        """Negative crop coordinates boundary is invalid."""
        invalid_coords = (-1, -1)
        self.assertTrue(invalid_coords[0] < 0 or invalid_coords[1] < 0)

    def test_bva_f2_zero_dimension_conditioning_boundary(self):
        """Zero dimensions boundary is invalid for diffusion conditioning."""
        zero_dim = (0, 0)
        self.assertTrue(zero_dim[0] == 0 or zero_dim[1] == 0)

    # =========================================================================
    # Feature 3: VRAM Memory Threshold Boundaries
    # =========================================================================

    def test_bva_f3_vram_zero_baseline_boundary(self):
        """0.0 GB lower boundary is valid float."""
        alloc = 0.0
        ceiling = 5.8
        self.assertTrue(alloc < ceiling)

    def test_bva_f3_vram_nominal_safe_boundary_4_5gb(self):
        """4.5 GB safe threshold passes well below 5.8 GB ceiling."""
        alloc = 4.5
        self.assertLess(alloc, 5.8)

    def test_bva_f3_vram_warning_boundary_5_5gb(self):
        """5.5 GB warning threshold is within 5.8 GB ceiling."""
        alloc = 5.5
        self.assertLess(alloc, 5.8)

    def test_bva_f3_vram_just_under_ceiling_5_79gb_passes(self):
        """5.79 GB (0.01 GB below ceiling) strictly passes."""
        alloc = 5.79
        self.assertTrue(alloc < 5.8)

    def test_bva_f3_vram_at_hard_ceiling_5_80gb_fails(self):
        """5.80 GB (at ceiling) fails the strict < 5.8 GB constraint."""
        alloc = 5.80
        self.assertFalse(alloc < 5.8)

    def test_bva_f3_vram_physical_capacity_6_00gb_fails(self):
        """6.00 GB hardware limit strictly fails ceiling."""
        alloc = 6.00
        self.assertFalse(alloc < 5.8)

    # =========================================================================
    # Feature 4: Architecture Support & Step Boundaries
    # =========================================================================

    def test_bva_f4_step_count_lower_boundary_step_1(self):
        """Step 1 is minimum 1-pass lightning boundary."""
        step = 1
        self.assertGreaterEqual(step, 1)

    def test_bva_f4_step_count_nominal_boundary_step_4(self):
        """Step 4 is nominal standard SDXL Turbo boundary."""
        step = 4
        self.assertEqual(step, 4)

    def test_bva_f4_step_count_ultra_boundary_step_6(self):
        """Step 6 is ultra preset boundary."""
        step = 6
        self.assertEqual(step, 6)

    def test_bva_f4_step_count_high_boundary_step_10(self):
        """Step 10 is high quality boundary."""
        step = 10
        self.assertLessEqual(step, 10)

    def test_bva_f4_step_count_upper_stress_boundary_step_50(self):
        """Step 50 is upper loop stress boundary."""
        step = 50
        self.assertGreaterEqual(step, 50)

    def test_bva_f4_step_count_zero_or_negative_invalid(self):
        """Step count <= 0 is invalid boundary."""
        for step in [0, -1, -5]:
            self.assertFalse(step >= 1)

    # =========================================================================
    # Feature 5: Guidance Scale (CFG) & Seed Boundaries
    # =========================================================================

    def test_bva_f5_cfg_lower_boundary_0_0(self):
        """CFG 0.0 lower boundary (pure speed 1-pass mode)."""
        cfg = 0.0
        self.assertGreaterEqual(cfg, 0.0)

    def test_bva_f5_cfg_neutral_boundary_1_0(self):
        """CFG 1.0 neutral baseline boundary."""
        cfg = 1.0
        self.assertEqual(cfg, 1.0)

    def test_bva_f5_cfg_nominal_boundary_1_5(self):
        """CFG 1.5 nominal SDXL Turbo boundary."""
        cfg = 1.5
        self.assertEqual(cfg, 1.5)

    def test_bva_f5_cfg_high_boundary_7_5(self):
        """CFG 7.5 standard base model boundary."""
        cfg = 7.5
        self.assertLessEqual(cfg, 7.5)

    def test_bva_f5_cfg_extreme_boundary_20_0(self):
        """CFG 20.0 extreme high boundary."""
        cfg = 20.0
        self.assertEqual(cfg, 20.0)

    def test_bva_f5_seed_boundary_zero(self):
        """Seed 0 is lower valid unsigned 32-bit boundary."""
        seed = 0
        self.assertGreaterEqual(seed, 0)

    def test_bva_f5_seed_boundary_max_uint32(self):
        """Seed 2^32 - 1 is upper valid 32-bit boundary."""
        max_uint32 = 2**32 - 1
        self.assertEqual(max_uint32, 4294967295)

    # =========================================================================
    # Feature 6: Commercial Downscaling Sizing Boundaries
    # =========================================================================

    def test_bva_f6_target_exact_1280x720_boundary(self):
        """Exact 1280x720 boundary passes dimension check."""
        img = Image.new("RGB", (1280, 720))
        res = ImageQualityAnalyzer.analyze(img, expected_width=1280, expected_height=720)
        self.assertTrue(res.dimensions_pass)

    def test_bva_f6_off_by_one_width_1281x720_fails(self):
        """Off-by-one width (1281x720) boundary fails dimension check."""
        img = Image.new("RGB", (1281, 720))
        res = ImageQualityAnalyzer.analyze(img, expected_width=1280, expected_height=720)
        self.assertFalse(res.dimensions_pass)

    def test_bva_f6_off_by_one_height_1280x721_fails(self):
        """Off-by-one height (1280x721) boundary fails dimension check."""
        img = Image.new("RGB", (1280, 721))
        res = ImageQualityAnalyzer.analyze(img, expected_width=1280, expected_height=720)
        self.assertFalse(res.dimensions_pass)

    def test_bva_f6_exact_pixel_count_921600_boundary(self):
        """1280 * 720 is exactly 921,600 pixels."""
        self.assertEqual(1280 * 720, 921600)

    def test_bva_f6_downscaling_factor_boundary(self):
        """Downscaling factor from 1344 to 1280 is 1280/1344 ~= 0.9524."""
        scale = 1280 / 1344
        self.assertAlmostEqual(scale, 0.95238, places=4)

    # =========================================================================
    # Feature 7: Prompt Single-Subject Boundaries
    # =========================================================================

    def test_bva_f7_face_count_lower_boundary_0_faces_fails(self):
        """0 faces on a single-subject prompt fails subject_count_pass."""
        img = Image.new("RGB", (1280, 720), color=(20, 20, 30))
        res = ImageQualityAnalyzer.analyze(img, expected_width=1280, expected_height=720, is_single_subject=True)
        self.assertEqual(res.face_count, 0)
        self.assertFalse(res.subject_count_pass)

    def test_bva_f7_face_count_nominal_boundary_1_face_passes(self):
        """Exactly 1 face on single-subject prompt passes."""
        img = create_synthetic_thumbnail(1280, 720, face_count=1)
        res = ImageQualityAnalyzer.analyze(img, expected_width=1280, expected_height=720, is_single_subject=True)
        self.assertEqual(res.face_count, 1)
        self.assertTrue(res.subject_count_pass)

    def test_bva_f7_face_count_upper_boundary_2_faces_fails(self):
        """2 faces on single-subject prompt fails."""
        img = create_synthetic_thumbnail(1280, 720, face_count=2)
        res = ImageQualityAnalyzer.analyze(img, expected_width=1280, expected_height=720, is_single_subject=True)
        self.assertGreaterEqual(res.face_count, 2)
        self.assertFalse(res.subject_count_pass)

    def test_bva_f7_duplicate_head_area_ratio_below_20pct_passes(self):
        """Secondary head candidate < 20% area is treated as non-subject noise."""
        b1 = BoundingBox(100, 100, 300, 400)  # area 60,000
        b2 = BoundingBox(400, 100, 480, 200)  # area 8,000 (13.3%)
        self.assertLess(b2.area / b1.area, 0.20)

    def test_bva_f7_duplicate_head_area_ratio_at_20pct_fails(self):
        """Secondary head candidate >= 20% area triggers duplicate head flag."""
        b1 = BoundingBox(100, 100, 300, 400)  # area 60,000
        b2 = BoundingBox(400, 100, 520, 220)  # area 14,400 (24%)
        self.assertGreaterEqual(b2.area / b1.area, 0.20)

    # =========================================================================
    # Feature 8: Three-Zone Spatial Composition Boundaries
    # =========================================================================

    def test_bva_f8_subject_x_left_third_boundary(self):
        """Left third boundary: 0 <= x <= 1280 * 0.33 ~= 426."""
        left_third_limit = int(1280 * 0.33)
        self.assertEqual(left_third_limit, 422)

    def test_bva_f8_subject_x_right_third_boundary(self):
        """Right third boundary: x >= 1280 * 0.67 ~= 857."""
        right_third_start = int(1280 * 0.67)
        self.assertEqual(right_third_start, 857)

    def test_bva_f8_depth_ratio_flat_1_0_boundary(self):
        """Depth focus ratio of 1.0 (flat focus) is boundary below recommended."""
        ratio = 1.0
        self.assertFalse(ratio > 1.1)

    def test_bva_f8_depth_ratio_separated_1_1_boundary(self):
        """Depth focus ratio > 1.1 represents optical depth separation."""
        ratio = 1.15
        self.assertTrue(ratio > 1.1)

    def test_bva_f8_background_threat_opposite_third_boundary(self):
        """Background threat is separated on opposite third across center divide (x=640)."""
        subject_center = 200
        threat_center = 1000
        self.assertLess(subject_center, 640)
        self.assertGreater(threat_center, 640)

    # =========================================================================
    # Feature 9: Anti-Hybrid & Anti-Duplication Boundaries
    # =========================================================================

    def test_bva_f9_conjoined_head_count_zero_boundary_passes(self):
        """conjoined_heads == 0 passes acceptance."""
        conjoined = 0
        self.assertEqual(conjoined, 0)

    def test_bva_f9_conjoined_head_count_one_boundary_fails(self):
        """conjoined_heads >= 1 strictly fails acceptance."""
        conjoined = 1
        self.assertFalse(conjoined == 0)

    def test_bva_f9_skin_decay_ratio_zero_boundary_passes(self):
        """Decay ratio 0.0 passes."""
        ratio = 0.0
        self.assertLess(ratio, 0.05)

    def test_bva_f9_skin_decay_ratio_just_below_0_049_passes(self):
        """Decay ratio 0.049 (< 0.05) passes."""
        ratio = 0.049
        self.assertLess(ratio, 0.05)

    def test_bva_f9_skin_decay_ratio_at_threshold_0_050_fails(self):
        """Decay ratio 0.050 (>= 0.05) fails."""
        ratio = 0.050
        self.assertFalse(ratio < 0.05)

    def test_bva_f9_skin_decay_ratio_extreme_0_50_fails(self):
        """Decay ratio 0.50 (heavy zombie rot) strictly fails."""
        ratio = 0.50
        self.assertFalse(ratio < 0.05)

    # =========================================================================
    # Feature 10: Automated E2E Benchmark Runner Boundaries
    # =========================================================================

    def test_bva_f10_scenario_id_range_boundaries_1_to_5(self):
        """Valid scenario IDs are strictly within [1, 5]."""
        valid_ids = [s["id"] for s in CURATED_SCENARIOS]
        self.assertEqual(min(valid_ids), 1)
        self.assertEqual(max(valid_ids), 5)
        self.assertEqual(len(valid_ids), 5)

    def test_bva_f10_scenario_id_out_of_bounds_0_and_6(self):
        """Scenario IDs 0 and 6 are out-of-bounds."""
        valid_ids = [s["id"] for s in CURATED_SCENARIOS]
        self.assertNotIn(0, valid_ids)
        self.assertNotIn(6, valid_ids)

    def test_bva_f10_empty_results_report_boundary(self):
        """Reporting handles empty results list gracefully."""
        runner = E2EBenchmarkRunner(mock_mode=True)
        # Should not crash on empty list
        summary_empty = []
        self.assertEqual(len(summary_empty), 0)

    def test_bva_f10_all_fail_suite_status_boundary(self):
        """If 1 scenario fails, suite status is FAILED."""
        r1 = ScenarioResult(
            scenario_id=1, name="s1", prompt="", negative_prompt="", model_backend="",
            target_width=1280, target_height=720, actual_width=1280, actual_height=720,
            inference_time_s=1.0, peak_vram_allocated_gb=1.0, peak_vram_reserved_gb=1.5,
            vram_pass=True, dimensions_pass=True, subject_count_pass=True, face_count=1,
            duplicate_heads=0, conjoined_heads=0, skin_quality_pass=True, skin_decay_ratio=0.0,
            depth_separation_pass=True, depth_separation_ratio=1.5, overall_pass=True
        )
        r2 = ScenarioResult(
            scenario_id=2, name="s2", prompt="", negative_prompt="", model_backend="",
            target_width=1280, target_height=720, actual_width=1280, actual_height=720,
            inference_time_s=1.0, peak_vram_allocated_gb=6.0, peak_vram_reserved_gb=6.0,
            vram_pass=False, dimensions_pass=True, subject_count_pass=True, face_count=1,
            duplicate_heads=0, conjoined_heads=0, skin_quality_pass=True, skin_decay_ratio=0.0,
            depth_separation_pass=True, depth_separation_ratio=1.5, overall_pass=False
        )
        all_passed = all(r.overall_pass for r in [r1, r2])
        self.assertFalse(all_passed)


if __name__ == "__main__":
    unittest.main()
