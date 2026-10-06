import copy
import importlib
import sys
from pathlib import Path

from production.footage.base import VideoProvider, VideoProviderError


MONKI_LABS_ROOT = (
    Path(__file__).resolve().parents[3]
    / "monki-labs"
)


def _load_monki_provider():
    """Load the existing Monki Labs provider without copying its logic."""
    saved_modules = {
        name: module
        for name, module in sys.modules.items()
        if name == "ai" or name.startswith("ai.")
    }

    for name in list(saved_modules):
        sys.modules.pop(name, None)

    original_path = list(sys.path)
    sys.path.insert(0, str(MONKI_LABS_ROOT))

    try:
        return importlib.import_module(
            "ai.providers.snapgenai_provider"
        ).SnapGenAiProvider
    finally:
        sys.path[:] = original_path

        for name in list(sys.modules):
            if name == "ai" or name.startswith("ai."):
                sys.modules.pop(name, None)

        sys.modules.update(saved_modules)


class SnapGenAiVideoProvider(VideoProvider):
    name = "snapgenai"

    def __init__(self, config, notify=None):
        super().__init__(config, notify=notify)
        self.config = copy.deepcopy(config)
        video_model = (
            self.config
            .setdefault("ai_models", {})
            .setdefault("models", {})
            .setdefault("video_model", {})
        )
        settings = video_model.setdefault("snapgenai", {})
        settings["profile_directory"] = str(
            MONKI_LABS_ROOT / "media" / "browser_profile" / "snapgenai"
        )
        settings["download_directory"] = str(
            MONKI_LABS_ROOT / "media" / "downloads" / "snapgenai"
        )
        remover = settings.setdefault("watermark_remover", {})
        remover["executable"] = str(
            MONKI_LABS_ROOT / "tools" / "GeminiWatermarkTool-Video.exe"
        )
        provider_class = _load_monki_provider()
        self.provider = provider_class(
            self.config,
            progress_callback=self.notify,
        )

    def fetch(
        self,
        query,
        destination_dir,
        max_videos=1,
        downloaded_ids=None,
        downloaded_hashes=None,
    ):
        query = str(query).strip()
        if not query:
            raise VideoProviderError(
                "The SnapGenAI visual prompt is empty."
            )

        destination_dir = Path(destination_dir)
        destination_dir.mkdir(parents=True, exist_ok=True)
        output_path = destination_dir / "clip_001.mp4"

        if output_path.is_file() and output_path.stat().st_size > 0:
            self.notify(
                "Already have the SnapGenAI visual clip; skipping generation."
            )
            return [str(output_path)]

        try:
            self.notify(
                "Generating the episode visual with SnapGenAI..."
            )
            generated = self.provider.generate_clip(
                query,
                output_path,
            )
        except Exception as error:
            raise VideoProviderError(
                f"SnapGenAI generation failed: {error}"
            ) from error

        return [str(generated)]