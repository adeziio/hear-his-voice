from core.pipeline import HearHisVoicePipeline


def main():
    print(r"""
       HEAR HIS VOICE
        THE GOSPEL, ONE SHORT FILM AT A TIME
""")

    pipeline = HearHisVoicePipeline()

    pipeline.create_episode()


if __name__ == "__main__":

    main()