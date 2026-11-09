from pathlib import Path

import av


class VideoValidationError(
    RuntimeError
):

    """
    Raised for any video validation failure: missing file, empty
    file, or unreadable/corrupt container. The message is always
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
    duration). Raises VideoValidationError on any failure.
    """

    path = Path(
        path
    )

    if not path.is_file():

        raise VideoValidationError(
            f"Video file is missing: {path.name}"
        )

    if path.stat().st_size <= 0:

        raise VideoValidationError(
            f"Video file is empty: {path.name}"
        )

    try:

        with av.open(
            str(
                path
            )
        ) as container:

            if not container.streams.video:

                raise VideoValidationError(
                    f"Video file contains no video "
                    f"stream: {path.name}"
                )

            if (
                container.duration is None
                or container.duration <= 0
            ):

                raise VideoValidationError(
                    f"Video file has no playable "
                    f"duration: {path.name}"
                )

    except av.error.FFmpegError as error:

        raise VideoValidationError(
            f"Video file is corrupt or unreadable: "
            f"{path.name} ({error})"
        )