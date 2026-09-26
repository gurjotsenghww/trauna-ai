"""
Tràuna AI — ai/tests/test_tier4_acceptance.py

Tier 4: Real-World Acceptance Scenarios
Executes and validates all 5 commercial YouTube thumbnail production scenarios:
1. Single Streamer Subject (Dual-Head Elimination Test)
2. Single Animal Subject (Anatomy & Non-Human Test)
3. Streamer + Minecraft Threat (Spatial Separation Test)
4. Streamer + Horror Entity (Skin/Eye Quality Test)
5. High-Speed Action / GTA (Scene Stability Test)
"""

import os
import tempfile
import unittest
from PIL import Image

from ai.benchmark_e2e import CURATED_SCENARIOS, E2EBenchmarkRunner, ImageQualityAnalyzer


class TestTier4RealWorldAcceptance(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.output_dir = self.tmpdir.name
        self.runner = E2EBenchmarkRunner(output_dir=self.output_dir, mock_mode=True)

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_scenario_1_single_streamer_dual_head_elimination(self):
        """Scenario 1: Single streamer subject yields exactly 1 face, 0 conjoined heads, 1280x720."""
        spec = CURATED_SCENARIOS[0]
        self.assertEqual(spec["id"], 1)
        res = self.runner.run_scenario(spec)

        self.assertTrue(res.overall_pass)
        self.assertTrue(res.dimensions_pass)
        self.assertEqual(res.actual_width, 1280)
        self.assertEqual(res.actual_height, 720)
        self.assertEqual(res.face_count, 1)
        self.assertEqual(res.duplicate_heads, 0)
        self.assertEqual(res.conjoined_heads, 0)
        self.assertTrue(res.subject_count_pass)
        self.assertLess(res.peak_vram_allocated_gb, 5.8)

    def test_scenario_2_single_animal_anatomy_and_duplicate_glitch(self):
        """Scenario 2: Single animal yields 1 coherent subject, 0 conjoined mutations, 1280x720."""
        spec = CURATED_SCENARIOS[1]
        self.assertEqual(spec["id"], 2)
        res = self.runner.run_scenario(spec)

        self.assertTrue(res.overall_pass)
        self.assertTrue(res.dimensions_pass)
        self.assertEqual(res.actual_width, 1280)
        self.assertEqual(res.actual_height, 720)
        self.assertEqual(res.duplicate_heads, 0)
        self.assertEqual(res.conjoined_heads, 0)
        self.assertTrue(res.subject_count_pass)

    def test_scenario_3_streamer_minecraft_threat_spatial_separation(self):
        """Scenario 3: Streamer + Creeper has clean human skin (decay_ratio < 0.05) and 1 face."""
        spec = CURATED_SCENARIOS[2]
        self.assertEqual(spec["id"], 3)
        res = self.runner.run_scenario(spec)

        self.assertTrue(res.overall_pass)
        self.assertEqual(res.face_count, 1)
        self.assertEqual(res.conjoined_heads, 0)
        self.assertEqual(res.duplicate_heads, 0)
        self.assertLess(res.skin_decay_ratio, 0.05)
        self.assertTrue(res.skin_quality_pass)
        self.assertGreater(res.depth_separation_ratio, 1.0)

    def test_scenario_4_streamer_horror_entity_skin_quality_and_anti_fusion(self):
        """Scenario 4: Streamer + Horror Zombie has natural skin/eyes without zombie decay fusion."""
        spec = CURATED_SCENARIOS[3]
        self.assertEqual(spec["id"], 4)
        res = self.runner.run_scenario(spec)

        self.assertTrue(res.overall_pass)
        self.assertEqual(res.face_count, 1)
        self.assertEqual(res.conjoined_heads, 0)
        self.assertLess(res.skin_decay_ratio, 0.05)
        self.assertTrue(res.skin_quality_pass)
        self.assertTrue(res.vram_pass)

    def test_scenario_5_gta_high_speed_action_stability(self):
        """Scenario 5: High speed action scene renders cleanly with zero OOM and distinct subject."""
        spec = CURATED_SCENARIOS[4]
        self.assertEqual(spec["id"], 5)
        res = self.runner.run_scenario(spec)

        self.assertTrue(res.overall_pass)
        self.assertEqual(res.face_count, 1)
        self.assertIsNone(res.error_message)
        self.assertLess(res.peak_vram_allocated_gb, 5.8)

    def test_all_scenarios_comprehensive_benchmark_run(self):
        """Full suite execution verifies 100% pass across all 5 scenarios with reports generated."""
        results = self.runner.run_all()
        self.assertEqual(len(results), 5)
        self.assertTrue(all(r.overall_pass for r in results))

        json_path = os.path.join(self.output_dir, "benchmark_summary.json")
        md_path = os.path.join(self.output_dir, "benchmark_report.md")
        self.assertTrue(os.path.exists(json_path))
        self.assertTrue(os.path.exists(md_path))


if __name__ == "__main__":
    unittest.main()
