import platform


class HardwareDetector:

    def detect(
        self
    ):

        hardware = {

            "os":
            platform.system(),

            "device":
            "cpu",

            "accelerator":
            "cpu",

            "gpu_available":
            False,

            "gpu_name":
            None,

            "torch_dtype":
            "float32",

            "supports_bfloat16":
            False

        }

        try:

            import torch

            if torch.cuda.is_available():

                index = (
                    torch.cuda.current_device()
                )

                supports_bfloat16 = (
                    torch.cuda.is_bf16_supported()
                )

                hardware.update(

                    {

                        "device":
                        "cuda",

                        "accelerator":
                        "nvidia",

                        "gpu_available":
                        True,

                        "gpu_name":
                        torch.cuda.get_device_name(
                            index
                        ),

                        "torch_dtype":
                        (
                            "bfloat16"
                            if supports_bfloat16
                            else "float16"
                        ),

                        "supports_bfloat16":
                        supports_bfloat16

                    }

                )

            elif (

                hasattr(
                    torch.backends,
                    "mps"
                )

                and

                torch.backends.mps.is_available()

            ):

                hardware.update(

                    {

                        "device":
                        "mps",

                        "accelerator":
                        "apple",

                        "gpu_available":
                        True,

                        "gpu_name":
                        "Apple Silicon GPU",

                        "torch_dtype":
                        "float16",

                        "supports_bfloat16":
                        False

                    }

                )

        except Exception:

            pass

        return hardware