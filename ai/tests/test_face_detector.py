"""
Tràuna AI — ai/tests/test_face_detector.py

Unit tests for computer vision and quality analyzers in benchmark_e2e:
- Face count detection
- Duplicate head anomaly detection
- Conjoined head anomaly detection
- Skin decay ratio (creeper/zombie green bleeding)
- Depth-of-field focus sharpness ratio
- Dimension compliance
"""

import unittest
from PIL import Image
from ai.benchmark_e2e import ImageQualityAnalyzer
from ai.tests.helpers import create_synthetic_thumbnail


class TestImageQualityAnalyzer(unittest.TestCase):

    def test_single_face_detection_passes(self):
        """Verify normal single-face image returns face_count=1, 0 conjoined, 0 duplicates."""
        img = create_synthetic_thumbnail(
            width=1280,
            height=720,
            face_count=1,
            conjoined=False,
            green_decay=False,
        )
        res = ImageQualityAnalyzer.analyze(img, expected_width=1280, expected_height=720, is_single_subject=True)
        self.assertTrue(res.dimensions_pass)
        self.assertEqual(res.face_count, 1)
        self.assertEqual(res.duplicate_heads, 0)
        self.assertEqual(res.conjoined_heads, 0)
        self.assertTrue(res.subject_count_pass)
        self.assertTrue(res.skin_quality_pass)

    def test_duplicate_heads_detection_fails_single_subject(self):
        """Verify multiple separate heads are flagged as duplicate heads on single-subject prompt."""
        img = create_synthetic_thumbnail(
            width=1280,
            height=720,
            face_count=2,
            conjoined=False,
        )
        res = ImageQualityAnalyzer.analyze(img, expected_width=1280, expected_height=720, is_single_subject=True)
        self.assertGreaterEqual(res.face_count, 2)
        self.assertGreaterEqual(res.duplicate_heads, 1)
        self.assertFalse(res.subject_count_pass)

    def test_conjoined_heads_anomaly_detection(self):
        """Verify overlapping conjoined heads are flagged as conjoined_heads >= 1."""
        img = create_synthetic_thumbnail(
            width=1280,
            height=720,
            conjoined=True,
        )
        res = ImageQualityAnalyzer.analyze(img, expected_width=1280, expected_height=720, is_single_subject=True)
        self.assertGreaterEqual(res.conjoined_heads, 1)
        self.assertFalse(res.subject_count_pass)

    def test_green_decay_ratio_fails_hybrid_monster_fusion(self):
        """Verify monster/zombie green decay on face ROI triggers skin quality failure."""
        img = create_synthetic_thumbnail(
            width=1280,
            height=720,
            face_count=1,
            green_decay=True,
        )
        res = ImageQualityAnalyzer.analyze(img, expected_width=1280, expected_height=720, is_single_subject=True)
        self.assertGreater(res.skin_decay_ratio, 0.05)
        self.assertFalse(res.skin_quality_pass)

    def test_clean_human_skin_quality_passes(self):
        """Verify natural human skin tones have decay ratio near 0.0%."""
        img = create_synthetic_thumbnail(
            width=1280,
            height=720,
            face_count=1,
            green_decay=False,
        )
        res = ImageQualityAnalyzer.analyze(img, expected_width=1280, expected_height=720, is_single_subject=True)
        self.assertLess(res.skin_decay_ratio, 0.05)
        self.assertTrue(res.skin_quality_pass)

    def test_dimensions_check_passes_and_fails(self):
        """Verify exact resolution enforcement."""
        img_correct = Image.new("RGB", (1280, 720))
        img_wrong = Image.new("RGB", (1024, 768))

        res_correct = ImageQualityAnalyzer.analyze(img_correct, expected_width=1280, expected_height=720)
        res_wrong = ImageQualityAnalyzer.analyze(img_wrong, expected_width=1280, expected_height=720)

        self.assertTrue(res_correct.dimensions_pass)
        self.assertFalse(res_wrong.dimensions_pass)

    def test_depth_separation_ratio(self):
        """Verify sharp subject on smooth bokeh background yields depth ratio > 1.0."""
        img = create_synthetic_thumbnail(
            width=1280,
            height=720,
            face_count=1,
            background_bokeh=True,
        )
        res = ImageQualityAnalyzer.analyze(img, expected_width=1280, expected_height=720)
        self.assertGreater(res.depth_separation_ratio, 1.0)
        self.assertTrue(res.depth_separation_pass)

    def test_animal_subject_mode(self):
        """Verify animal mode evaluation bypasses human skin chrominance constraints."""
        img = create_synthetic_thumbnail(width=1280, height=720, face_count=1)
        res = ImageQualityAnalyzer.analyze(img, expected_width=1280, expected_height=720, is_animal=True)
        self.assertTrue(res.dimensions_pass)
        self.assertTrue(res.subject_count_pass)
        self.assertTrue(res.skin_quality_pass)


if __name__ == "__main__":
    unittest.main()
