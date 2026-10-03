import json

from pathlib import Path


class ConfigLoader:

    def __init__(
        self,
        config_directory="config"
    ):

        self.config_directory = Path(
            config_directory
        )

    def load(
        self,
        filename
    ):

        file_path = (
            self.config_directory
            /
            filename
        )

        if not file_path.exists():

            raise FileNotFoundError(
                f"Config file missing: {file_path}"
            )

        with open(
            file_path,
            "r",
            encoding="utf-8"
        ) as file:

            return json.load(
                file
            )

    def load_all(
        self
    ):

        return {

            "content":
                self.load(
                    "content.json"
                ),

            "ai_models":
                self.load(
                    "ai_models.json"
                ),

        }