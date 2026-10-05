import shutil
import sys
import tempfile
import unittest

from pathlib import Path

PROJECT_ROOT = (
    Path(__file__).resolve().parents[1]
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.pipeline import HearHisVoicePipeline


class ResolveEpisodeDirectoryTest(unittest.TestCase):

    def test_accepts_relative_path_and_bare_number(self):
        media_root = (
            HearHisVoicePipeline.project_root()
            / "media" / "output" / "shorts"
        )
        media_root.mkdir(parents=True, exist_ok=True)
        created = []
        try:
            for name in ("001", "995", "996"):
                d = media_root / name
                d.mkdir(parents=True, exist_ok=True)
                created.append(d)

            by_path = HearHisVoicePipeline.resolve_episode_directory(
                "media/output/shorts/995",
            )
            self.assertEqual(by_path.name, "995")

            by_unpadded_number = HearHisVoicePipeline.resolve_episode_directory(
                "1",
            )
            self.assertEqual(by_unpadded_number.name, "001")

            by_number = HearHisVoicePipeline.resolve_episode_directory(
                "996",
            )
            self.assertEqual(by_number.name, "996")
        finally:
            for d in created:
                shutil.rmtree(d, ignore_errors=True)

    def test_missing_episode_raises_value_error(self):
        with self.assertRaises(ValueError):
            HearHisVoicePipeline.resolve_episode_directory(
                "media/output/shorts/994",
            )


if __name__ == "__main__":
    unittest.main()
