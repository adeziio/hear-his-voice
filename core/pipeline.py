from pathlib import Path

from core.config_loader import ConfigLoader
from core.dependencies import (
    ensure_dependencies
)

from scripture.selector import PassageSelector

from ai.content_generator import (
    ContentGenerator,
    write_content_files,
    read_content_file
)

from production.video import ProductionPipeline


OUTPUT_ROOT = (
    Path(
        "media"
    )
    /
    "output"
    /
    "shorts"
)


class HearHisVoicePipeline:

    """
    The Hear His Voice production pipeline.

        Create episode
            -> Scripture stage: read the exact WEBC passage
            -> Narration stage: tell that passage as original
               storytelling, checked against it
            -> Prompt stage: the AI directs the visuals for the telling
            -> Generate Video stage: narration audio, stock footage,
               music, captions, render

    The Scripture comes from the WEBC source and stays the authority:
    the narration may only restate what the passage says, the exact
    passage is kept beside it as source_text, and the AI only decides
    what the viewer sees while those words are spoken. The pipeline
    only enforces the structured output and the technical production
    requirements.
    """

    def __init__(
        self
    ):

        # Fails fast, with the interpreter named, rather than part way
        # through a job with a bare ModuleNotFoundError.
        ensure_dependencies()

        loader = ConfigLoader()

        self.config = (
            loader.load_all()
        )

        self.scripture_selector = (
            PassageSelector(
                self.config
            )
        )

        self.content_generator = (
            ContentGenerator(
                self.config
            )
        )

        self.production = (
            ProductionPipeline(
                self.config
            )
        )

        self.progress_callback = None

    def set_progress_callback(
        self,
        callback
    ):

        """
        Registers a progress callback invoked as
        callback(percent, message, stage) where stage is
        "prompt" or "video".

        The callback is also propagated to the production pipeline
        so that video-stage sub-progress reaches the web UI.
        """

        self.progress_callback = (
            callback
        )

        if self.production is not None:

            self.production.set_progress_callback(
                callback
            )

    def _notify(
        self,
        percent,
        message,
        stage
    ):

        if self.progress_callback is None:

            print(
                f"[{stage.upper()} {percent}%] {message}"
            )

            return

        try:

            self.progress_callback(
                int(
                    percent
                ),
                str(
                    message
                ),
                stage
            )

        except Exception:

            pass

    @staticmethod
    def project_root():

        return (
            Path(
                __file__
            )
            .resolve()
            .parents[1]
        )

    def _next_episode_directory(
        self
    ):

        output_root = (
            self.project_root()
            /
            OUTPUT_ROOT
        )

        output_root.mkdir(
            parents=True,
            exist_ok=True
        )

        next_number = 1

        for entry in output_root.iterdir():

            if (
                entry.is_dir()
                and entry.name.isdigit()
            ):

                next_number = max(
                    next_number,
                    int(
                        entry.name
                    )
                    + 1
                )

        return (
            output_root
            /
            f"{next_number:03d}"
        )

    @staticmethod
    def resolve_episode_directory(
        episode_id
    ):
        """
        Accepts an episode id like "media/output/shorts/3" or
        just "3" and returns the episode directory.
        """
        episode_id = str(
            episode_id or ""
        ).strip()

        if not episode_id:

            raise ValueError(
                "Episode ID is required."
            )

        if episode_id.isdigit():

            episode_directory = (
                HearHisVoicePipeline.project_root()
                /
                OUTPUT_ROOT
                /
                f"{int(episode_id):03d}"
            )

        else:

            cleaned = (
                episode_id.replace("\\", "/").strip().strip("/")
            )

            leaf = (
                cleaned.rsplit("/", 1)[-1].strip()
            )

            if leaf.isdigit():

                episode_directory = (
                    HearHisVoicePipeline.project_root()
                    /
                    OUTPUT_ROOT
                    /
                    f"{int(leaf):03d}"
                )

            else:

                episode_directory = (
                    HearHisVoicePipeline.project_root()
                    /
                    cleaned
                )

                allowed_root = (
                    HearHisVoicePipeline.project_root()
                    /
                    OUTPUT_ROOT
                ).resolve()

                if (
                    allowed_root
                    not in episode_directory.resolve().parents
                ):

                    raise ValueError(
                        "Invalid episode path."
                    )

        if not episode_directory.is_dir():

            raise ValueError(
                "Episode does not exist."
            )

        return episode_directory

    def _generate_content(
        self,
        episode_directory,
        reference=None
    ):
        """
        Reads the exact WEBC passage, tells it as original narration,
        then asks the AI to direct the visuals for that telling and
        saves the episode content.

        The passage is fetched first and stays the authority: the
        narration is written from it under the fidelity checks, the
        exact passage is kept in content.json as source_text, and no
        part of the visual direction can reach the spoken words.
        """
        if reference:

            scripture = (
                self.scripture_selector
                .get_reference(reference)
            )

        else:

            scripture = (
                self.scripture_selector.select()
            )

        self._notify(
            5,
            f"Scripture: {scripture['reference']} "
            f"({scripture['translation']})",
            "prompt"
        )

        narration = (
            self.content_generator
            .write_narration(scripture)
        )

        self._notify(
            15,
            f"Narration written "
            f"({len(narration.split())} words).",
            "prompt"
        )

        # The narration, not the passage, is what will be spoken and
        # captioned, so it is the narration that gets cut into the
        # visual segments. Keep this working copy separate: the original
        # Scripture dict remains the authoritative WEBC source passed to
        # visual generation and saved as source_text.
        narration_script = dict(
            scripture,
            text=narration,
        )

        segments = (
            self.content_generator.build_segments(
                narration_script
            )
        )

        self._notify(
            25,
            f"Directing {len(segments)} visual segments...",
            "prompt"
        )

        content = (
            self.content_generator.generate(
                scripture,
                segments,
                narration
            )
        )

        self._notify(
            95,
            f"Episode content ready: {content['title']}",
            "prompt"
        )

        write_content_files(
            episode_directory,
            content
        )

        return content

    def create_episode(
        self,
        prompt_only=False,
        episode_id=None,
        reference=None
    ):
        """
        Runs the Scripture and Prompt stages and, unless prompt_only is
        set, the video stage as well.
        """
        if episode_id:

            episode_directory = (
                self.resolve_episode_directory(
                    episode_id
                )
            )

        else:

            episode_directory = (
                self._next_episode_directory()
            )

        episode_directory.mkdir(
            parents=True,
            exist_ok=True
        )

        content = (
            self._generate_content(
                episode_directory,
                reference
            )
        )

        return {
            "episode_id": episode_directory.name,
            "episode_path": str(
                episode_directory.relative_to(
                    self.project_root()
                )
            ),
            "content": content
        }

    def create_prompt(
        self,
        episode_id=None,
        reference=None
    ):
        """
        Runs only the Scripture and Prompt stages.
        """
        if episode_id:

            episode_directory = (
                self.resolve_episode_directory(
                    episode_id
                )
            )

        else:

            episode_directory = (
                self._next_episode_directory()
            )

        episode_directory.mkdir(
            parents=True,
            exist_ok=True
        )

        content = (
            self._generate_content(
                episode_directory,
                reference
            )
        )

        return {
            "episode_id": episode_directory.name,
            "episode_path": str(
                episode_directory.relative_to(
                    self.project_root()
                )
            ),
            "content": content
        }

    def generate_video_from_prompt(
        self,
        prompt_item,
        episode_id
    ):
        """
        Runs the video stage for an episode whose content already
        exists. content.json is the source of truth; prompt_item is
        unused because the telling and its visuals are already fixed
        there.
        """
        episode_directory = (
            self.resolve_episode_directory(
                episode_id
            )
        )

        content = (
            read_content_file(
                episode_directory
            )
        )

        if content is None:

            raise ValueError(
                "This episode has no generated content. "
                "Run the prompt stage first."
            )

        video = (
            self.production.run(
                episode_directory,
                content
            )
        )

        return {
            "episode_id": episode_directory.name,
            "episode_path": str(
                episode_directory.relative_to(
                    self.project_root()
                )
            ),
            "video": video
        }
