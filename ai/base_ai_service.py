class BaseAIService:

    def __init__(
        self,
        config,
        log_prefix="SYSTEM"
    ):

        self.config = config

        self.hardware = config.get(
            "hardware",
            {
                "device":
                "cpu",

                "torch_dtype":
                "float32"
            }
        )

        self.log_prefix = log_prefix

    @property
    def device(self):

        return self.hardware["device"]

    @property
    def torch_dtype(self):

        return self.hardware["torch_dtype"]

    def log(
        self,
        message
    ):

        print(
            f"[{self.log_prefix}] {message}"
        )
