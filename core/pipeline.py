from core.config_loader import ConfigLoader
from core.hardware_detector import HardwareDetector

from ai.video_generator import VideoGenerator


class HearHisVoicePipeline:

    def __init__(
        self
    ):

        self.hardware = (
            HardwareDetector().detect()
        )

        print(
            f"[SYSTEM] Running on device: "
            f"{self.hardware['device']}"
        )

        loader = ConfigLoader()

        self.config = (
            loader.load_all()
        )

        self.config["hardware"] = (
            self.hardware
        )

        self.video = (
            VideoGenerator(
                self.config
            )
        )

    def set_progress_callback(
        self,
        callback
    ):

        self.video.set_progress_callback(
            callback
        )

    def create_prompt(
        self,
        episode_id=None
    ):

        return (
            self.video.create_prompt(
                episode_id=episode_id
            )
        )

    def generate_video_from_prompt(
        self,
        prompt_item,
        episode_id
    ):

        return (
            self.video.generate_from_prompt(
                prompt_item,
                episode_id
            )
        )

    def create_episode(
        self,
        prompt_only=False
    ):

        if prompt_only:

            return (
                self.video.create_prompt()
            )

        video = (
            self.video.generate()
        )

        return video