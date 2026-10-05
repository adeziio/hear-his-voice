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
            -> Prompt stage: the AI directs the visuals for it
            -> Generate Video stage: narration audio, stock footage,
               music, captions, render

    The Scripture comes from the WEBC source and is never written by the
    AI. The AI only decides what the viewer sees while those exact
    words are spoken. The pipeline only enforces the structured output
    and the technical production requirements.
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

    def resolve_episode_directory(
        self,
        episode_id
    ):
        """
        Resolves an episode id to its directory. Ids are the directory
        names under media/output/shorts, so "1" and "001" both work.
        """
        if episode_id is None:

            raise ValueError(
                "An episode id is required."
            )

        wanted = str(
            episode_id
        ).strip()

        if not wanted.isdigit():

            raise ValueError(
                f"'{episode_id}' is not a valid episode id."
            )

        name = f"{int(wanted):03d}"

        directory = (
            self.project_root()
            /
            OUTPUT_ROOT
            /
            name
        )

        if not directory.is_dir():

            raise FileNotFoundError(
                f"Episode {name} does not exist."
            )

        return directory

    def _generate_content(
        self,
        episode_directory,
        reference=None
    ):
        """
        Reads the exact WEBC passage and asks the AI to direct its
        visuals, then saves the episode content.

        The passage is fetched first and is never sent back through the
        model as text to rewrite: the narration written to
        content.json is the verbatim WEBC string.
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

        segments = (
            self.content_generator.build_segments(
                scripture
            )
        )

        self._notify(
            20,
            f"Directing {len(segments)} visual segments...",
            "prompt"
        )

        content = (
            self.content_generator.generate(
                scripture,
                segments
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
        unused because the narration must never come from the AI.
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
