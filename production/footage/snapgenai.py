from pathlib import Path

from ai.providers.snapgenai_provider import SnapGenAiProvider
from production.footage.base import VideoProvider, VideoProviderError


class SnapGenAiVideoProvider(VideoProvider):
    name = "snapgenai"

    def __init__(self, config, notify=None):
        super().__init__(config, notify=notify)
        self.provider = SnapGenAiProvider(
            config,
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

        generated_path = Path(generated)

        if (
            generated_path.resolve()
            != output_path.resolve()
        ):
            raise VideoProviderError(
                "SnapGenAI returned an unexpected output path: "
                f"{generated_path}"
            )

        if (
            not generated_path.is_file()
            or generated_path.stat().st_size <= 0
        ):
            raise VideoProviderError(
                "SnapGenAI returned a missing or empty visual clip: "
                f"{generated_path.name}"
            )

        self.notify(
            "SnapGenAI visual clip downloaded and processed successfully."
        )

        return [str(generated_path)]