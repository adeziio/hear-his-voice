import sys

from core.pipeline import HearHisVoicePipeline


def main():
    print(r"""
       HEAR HIS VOICE
    THE WORD, SPOKEN AND SHOWN
    """)

    pipeline = HearHisVoicePipeline()

    # An optional argument pins the passage for this run, e.g.
    #     python main.py "John 3:16-18"
    # Without it, the next unplayed reference is selected automatically.
    reference = (
        sys.argv[1].strip()
        if len(sys.argv) > 1
        else None
    )

    pipeline.create_episode(
        reference=(
            reference
            if reference
            else None
        )
    )


if __name__ == "__main__":

    main()