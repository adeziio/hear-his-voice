from production.footage.base import (
    VideoProvider,
    VideoProviderError
)


def create_video_provider(
    config,
    notify=None
):

    """
    Factory that resolves the active video provider from
    the generic configuration value:

        { "video_provider": "snapgenai" }

    The video-generation pipeline only works with the generic
    VideoProvider interface, so provider-specific behavior stays out
    of the rest of the pipeline.
    """

    app_config = (
        config.get(
            "app",
            {}
        )
    )

    name = str(
        app_config.get(
            "video_provider",
            "snapgenai"
        )
    ).strip().lower()

    if name == "snapgenai":

        from production.footage.snapgenai import (
            SnapGenAiVideoProvider
        )

        return SnapGenAiVideoProvider(
            config,
            notify=notify
        )

    raise VideoProviderError(
        f"Unknown video provider: {name}. "
        "Check video_provider in config/app.json."
    )