import subprocess

from pathlib import Path

import av


# PyAV renamed its base FFmpeg exception across versions; resolve
# whichever base class this environment provides.

try:

    FFMPEG_ERROR = av.error.FFmpegError

except AttributeError:  # pragma: no cover - older PyAV

    FFMPEG_ERROR = av.AVError


class VeoWatermarkError(
    RuntimeError
):

    """
    Raised for any VeoWatermarkRemover failure: missing executable,
    non-zero exit, timeout, or invalid output. The message is always
    safe to surface directly in job state.

    """

    def __init__(
        self,
        message
    ):

        super().__init__(
            message
        )


def validate_video_file(
    path
):

    """
    Verifies a video file exists, is non-empty, and can be opened by
    PyAV (valid container, at least one video stream, positive
    duration). Raises VeoWatermarkError on any failure.
    """

    path = Path(
        path
    )

    if not path.is_file():

        raise VeoWatermarkError(
            f"Video file is missing: {path.name}"
        )

    if path.stat().st_size <= 0:

        raise VeoWatermarkError(
            f"Video file is empty: {path.name}"
        )

    try:

        with av.open(
            str(
                path
            )
        ) as container:

            if not container.streams.video:

                raise VeoWatermarkError(
                    f"Video file contains no video "
                    f"stream: {path.name}"
                )

            if (
                container.duration is None
                or container.duration <= 0
            ):

                raise VeoWatermarkError(
                    f"Video file has no playable "
                    f"duration: {path.name}"
                )

    except FFMPEG_ERROR as error:

        raise VeoWatermarkError(
            f"Video file is corrupt or unreadable: "
            f"{path.name} ({error})"
        )


class VeoWatermarkRemover:

    """
    CPU-mode wrapper for the open-source VeoWatermarkRemover CLI
    (https://github.com/TrungCang165/VeoWatermarkRemover).

    Only the standard/default reverse-alpha-blending mode is used;
    the ML (--ml) mode is never enabled by this wrapper. The CLI
    corrects the watermarked region and re-muxes the stream, so the
    output keeps the original resolution, FPS, duration, and audio
    untouched. No upscaling or enhancement is applied.

    All connection details come from config/ai_models.json ->
    models.video_model.snapgenai.watermark_remover.
    """

    DEFAULT_EXECUTABLE = (
        "tools/GeminiWatermarkTool-Video.exe"
    )

    SUPPORTED_SUFFIXES = (
        ".mp4",
        ".mkv",
        ".mov"
    )

    ERROR_OUTPUT_CHARS = 400

    def __init__(
        self,
        config,
        notify=None
    ):

        video_config = (
            config["ai_models"]
            ["models"]
            ["video_model"]
        )

        snapgenai_config = (
            video_config.get(
                "snapgenai",
                {}
            )
        )

        self.settings = (
            snapgenai_config.get(
                "watermark_remover",
                {}
            )
        )

        if not isinstance(
            self.settings,
            dict
        ):

            self.settings = {}

        self.notify_callback = (
            notify
        )

    def _notify(
        self,
        message
    ):

        if self.notify_callback is None:

            return

        try:

            self.notify_callback(
                str(
                    message
                )
            )

        except Exception:

            pass

    def _setting(
        self,
        name,
        default=""
    ):

        return self.settings.get(
            name,
            default
        )

    def _executable(
        self
    ):

        executable = Path(
            str(
                self._setting(
                    "executable",
                    self.DEFAULT_EXECUTABLE
                )
            )
        )

        if not executable.is_file():

            raise VeoWatermarkError(
                "VeoWatermarkRemover executable not "
                f"found at '{executable}'. Download it "
                "from https://github.com/TrungCang165/"
                "VeoWatermarkRemover/releases and set "
                "models.video_model.snapgenai."
                "watermark_remover.executable in "
                "config/ai_models.json."
            )

        return executable

    def _extra_args(
        self
    ):

        extra_args = (
            self._setting(
                "extra_args",
                []
            )
            or []
        )

        if not isinstance(
            extra_args,
            list
        ):

            raise VeoWatermarkError(
                "watermark_remover.extra_args must "
                "be a list of strings."
            )

        args = [
            str(
                argument
            )
            for argument in extra_args
        ]

        # The default (CPU, reverse-alpha-blending) mode is
        # mandatory for this workflow - ML mode is never used.

        for argument in args:

            if argument.strip().lower() == "--ml":

                raise VeoWatermarkError(
                    "The ML (--ml) watermark-removal mode "
                    "is not supported by this workflow; "
                    "remove it from watermark_remover."
                    "extra_args."
                )

        return args

    def _timeout_seconds(
        self
    ):

        return float(
            self._setting(
                "timeout_seconds",
                3600
            )
        )

    def _output_flag(
        self
    ):

        return str(
            self._setting(
                "output_flag",
                ""
            )
        ).strip()

    def _snapshot_files(
        self,
        directory
    ):

        try:

            return {
                (
                    path.name,
                    path.stat().st_mtime
                )
                for path in directory.iterdir()
                if path.is_file()
            }

        except OSError:

            return set()

    def _locate_output(
        self,
        output_path,
        input_path,
        before,
        required=True
    ):

        # Some CLI builds only write next to the input file. When
        # the requested output was not created, fall back to the
        # newest video file the tool produced during the run and
        # move it into place. Returns the located path, or None
        # when required=False and the tool produced nothing.

        if output_path.is_file():

            return output_path

        candidates = []

        for candidate in input_path.parent.iterdir():

            if candidate == input_path:

                continue

            if not candidate.is_file():

                continue

            if candidate.suffix.lower() not in (
                self.SUPPORTED_SUFFIXES
            ):

                continue

            if (
                candidate.name,
                candidate.stat().st_mtime
            ) in before:

                continue

            candidates.append(
                candidate
            )

        if not candidates:

            if required:

                raise VeoWatermarkError(
                    "VeoWatermarkRemover finished but the "
                    f"cleaned output file is missing: "
                    f"{output_path.name}"
                )

            return None

        newest = max(
            candidates,
            key=(
                lambda path: path.stat().st_mtime
            )
        )

        newest.replace(
            output_path
        )

        return output_path

    def remove_watermark(
        self,
        input_path,
        output_path
    ):

        """
        Runs the remover CLI on input_path in the standard CPU mode
        and writes the cleaned video to output_path. The output is
        validated before returning; the original input is kept
        intact so the caller decides when to discard it.
        """

        input_path = Path(
            input_path
        )

        output_path = Path(
            output_path
        )

        validate_video_file(
            input_path
        )

        if input_path.suffix.lower() not in (
            self.SUPPORTED_SUFFIXES
        ):

            raise VeoWatermarkError(
                "VeoWatermarkRemover supports .mp4, "
                f".mkv and .mov inputs only, got: "
                f"{input_path.suffix or '(none)'}"
            )

        executable = (
            self._executable()
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        if output_path.exists():

            try:

                output_path.unlink()

            except OSError as error:

                raise VeoWatermarkError(
                    f"Could not replace existing output "
                    f"file {output_path.name}: {error}"
                )

        command = [
            str(
                executable
            ),
            str(
                input_path
            )
        ]

        output_flag = (
            self._output_flag()
        )

        if output_flag:

            command.append(
                output_flag
            )

        command.append(
            str(
                output_path
            )
        )

        command.extend(
            self._extra_args()
        )

        self._notify(
            "Removing Veo watermark "
            "(standard CPU mode)..."
        )

        before = self._snapshot_files(
            input_path.parent
        )

        try:

            process = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=(
                    self._timeout_seconds()
                )
            )

        except subprocess.TimeoutExpired:

            raise VeoWatermarkError(
                "VeoWatermarkRemover timed out after "
                f"{self._timeout_seconds():g} seconds."
            )

        except OSError as error:

            raise VeoWatermarkError(
                f"Could not run VeoWatermarkRemover: "
                f"{error}"
            )

        if process.returncode != 0:

            output_tail = (
                (
                    process.stderr
                    or process.stdout
                    or ""
                ).strip()
                [
                    -self.ERROR_OUTPUT_CHARS:
                ]
            )

            # Some builds exit non-zero even after writing the
            # cleaned video, so only treat the failure as fatal
            # when no output could be located.

            located = self._locate_output(
                output_path,
                input_path,
                before,
                required=False
            )

            if located is None:

                raise VeoWatermarkError(
                    "VeoWatermarkRemover failed with exit "
                    f"code {process.returncode}."
                    + (
                        f" Output: {output_tail}"
                        if output_tail
                        else ""
                    )
                )

            self._notify(
                "VeoWatermarkRemover reported exit code "
                f"{process.returncode} but produced the "
                "cleaned video; continuing."
            )

        else:

            self._locate_output(
                output_path,
                input_path,
                before
            )

        validate_video_file(
            output_path
        )

        self._notify(
            "Veo watermark removed and "
            "output validated."
        )

        return str(
            output_path
        )
