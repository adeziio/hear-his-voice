import os
import time

from pathlib import Path

import requests

from dotenv import (
    load_dotenv
)


class LtxApiError(
    RuntimeError
):

    """
    Raised for any LTX Fast API failure: submission errors, failed
    jobs, timeouts, malformed responses, and network problems. The
    message is always safe to surface directly in job state.
    """

    def __init__(
        self,
        message
    ):

        super().__init__(
            message
        )

class LtxApiProvider:

    """
    Asynchronous LTX-2.3 Fast API backend.

    Workflow: submit a generation request -> receive a job ID -> poll
    the status endpoint until completed/failed -> download the result
    video (with audio) to the requested output path.

    All connection details come from config/ai_models.json ->
    models.video_model.ltx. The API key is read from the environment
    variable named by api_key_env (default LTX_API_KEY) and is never
    hardcoded anywhere.
    """

    DEFAULT_JOB_ID_FIELDS = [
        "id",
        "job_id",
        "generation_id",
        "request_id",
        "task_id",
        "uid"
    ]

    DEFAULT_RESULT_URL_FIELDS = [
        "video_url",
        "result_url",
        "output_url",
        "download_url",
        "url"
    ]

    def __init__(
        self,
        config,
        progress_callback=None
    ):

        self.config = config

        self.video_config = (
            config["ai_models"]
            ["models"]
            ["video_model"]
        )

        # Callers may already have loaded .env; loading again is
        # harmless and keeps the provider usable standalone.

        load_dotenv()

        self.settings = (
            self.video_config.get(
                "ltx",
                {}
            )
        )

        if not isinstance(
            self.settings,
            dict
        ):

            self.settings = {}

        self.progress_callback = (
            progress_callback
        )

    def _setting(
        self,
        name,
        default=""
    ):

        value = self.settings.get(
            name,
            default
        )

        return value

    def _base_url(self):

        base = str(
            self._setting("base_url")
        ).strip().rstrip("/")

        if not base:

            raise LtxApiError(
                "LTX API base_url is not configured. "
                "Set models.video_model.ltx.base_url "
                "in config/ai_models.json."
            )

        return base

    def _api_key(self):

        env_name = str(
            self._setting(
                "api_key_env",
                "LTX_API_KEY"
            )
        ).strip()

        api_key = os.getenv(
            env_name,
            ""
        ).strip()

        if not api_key:

            raise LtxApiError(
                f"Environment variable {env_name} is not set. "
                "Add your LTX API key to .env."
            )

        return api_key

    def _headers(self):

        headers = {
            "Content-Type": "application/json"
        }

        auth_header = str(
            self._setting(
                "auth_header",
                "Authorization"
            )
        ).strip() or "Authorization"

        auth_scheme = str(
            self._setting(
                "auth_scheme",
                "Bearer"
            )
        ).strip()

        headers[auth_header] = (
            f"{auth_scheme} {self._api_key()}".strip()
        )

        return headers

    def _request_timeout(self):

        try:

            return float(
                self._setting(
                    "request_timeout_seconds",
                    30
                )
            )

        except (
            TypeError,
            ValueError
        ):

            return 30.0

    def _poll_interval(self):

        try:

            interval = float(
                self._setting(
                    "poll_interval_seconds",
                    5
                )
            )

        except (
            TypeError,
            ValueError
        ):

            interval = 5.0

        return max(
            0.5,
            interval
        )

    def _deadline_seconds(self):

        try:

            timeout = float(
                self._setting(
                    "timeout_seconds",
                    900
                )
            )

        except (
            TypeError,
            ValueError
        ):

            timeout = 900.0

        return max(
            30.0,
            timeout
        )

    # ------------------------------------------------------------------
    # HTTP and parsing helpers
    # ------------------------------------------------------------------

    def _notify(
        self,
        message
    ):

        if self.progress_callback:

            self.progress_callback(
                message
            )

    def _parse_json(
        self,
        response,
        description
    ):

        try:

            payload = response.json()

        except ValueError:

            snippet = response.text[:300]

            raise LtxApiError(
                f"{description} returned malformed JSON "
                f"(HTTP {response.status_code}). Body: {snippet}"
            )

        if not isinstance(
            payload,
            dict
        ):

            raise LtxApiError(
                f"{description} returned an unexpected JSON "
                "structure (expected an object)."
            )

        error = payload.get("error")

        if isinstance(error, dict):

            error = (
                error.get("message")
                or ""
            )

        if response.status_code >= 400:

            message = (
                str(error).strip()
                if error
                else f"HTTP {response.status_code}"
            )

            error = LtxApiError(
                f"{description} failed: {message}"
            )

            raise error

        return payload

    def _find_first_key(
        self,
        payload,
        keys
    ):

        """
        Returns the value of the first matching key, searching
        top-level first and then one level deep in nested objects.
        """

        if not isinstance(
            payload,
            dict
        ):

            return None

        for key in keys:

            value = payload.get(key)

            if value not in (
                None,
                ""
            ):

                return value

        for value in payload.values():

            if isinstance(
                value,
                dict
            ):

                for key in keys:

                    nested = value.get(key)

                    if nested not in (
                        None,
                        ""
                    ):

                        return nested

        return None

    @staticmethod
    def _top_level_keys(payload):

        if isinstance(payload, dict):

            return sorted(
                str(key)
                for key in payload.keys()
            )

        return ["unknown"]

    @staticmethod
    def _error_detail(payload):

        """
        Extracts a human-readable message from the LTX error shape:
        either {"error": {"type": ..., "message": ...}} at the top
        level or the same structure nested inside the job object.
        """

        candidates = [
            payload.get("error"),
            (
                payload.get("job") or {}
            ).get("error")
            if isinstance(payload.get("job"), dict)
            else None
        ]

        for candidate in candidates:

            if isinstance(candidate, dict):

                message = (
                    candidate.get("message")
                    or candidate.get("type")
                    or ""
                )

                if message:

                    return str(message)

            elif candidate:

                return str(candidate)

        return ""

    def _extract_job_id(
        self,
        payload
    ):

        fields = list(
            self._setting("job_id_fields", [])
            or self.DEFAULT_JOB_ID_FIELDS
        )

        job_id = self._find_first_key(
            payload,
            fields
        )

        job_id = (
            str(job_id).strip()
            if job_id is not None
            else ""
        )

        if not job_id:

            raise LtxApiError(
                "Submission response did not contain a job ID. "
                f"Response keys: {self._top_level_keys(payload)}. "
                "Configure api.job_id_fields if the provider uses a "
                "different name."
            )

        return job_id

    def _status_value(
        self,
        payload
    ):

        status_field = str(
            self._setting(
                "status_field",
                "status"
            )
        ).strip() or "status"

        value = payload.get(status_field)

        if value is None:

            value = self._find_first_key(
                payload,
                [
                    status_field,
                    "state",
                    "status_code"
                ]
            )

        return (
            str(value).strip().lower()
            if value is not None
            else ""
        )

    def _result_video_url(
        self,
        payload
    ):

        fields = list(
            self._setting("result_url_fields", [])
            or self.DEFAULT_RESULT_URL_FIELDS
        )

        candidate = self._find_first_key(
            payload,
            fields
        )

        if isinstance(candidate, dict):

            candidate = self._find_first_key(
                candidate,
                fields
            )

        url = (
            str(candidate).strip()
            if candidate is not None
            else ""
        )

        if url.lower().startswith(("http://", "https://")):

            return url

        # Fallback: search recursively for any MP4 URL.

        found = []

        def walk(node):

            if isinstance(node, dict):

                for value in node.values():

                    walk(value)

            elif isinstance(node, list):

                for value in node:

                    walk(value)

            elif isinstance(node, str):

                lowered = node.strip().lower()

                if lowered.startswith(
                    ("http://", "https://")
                ) and ".mp4" in lowered.split("?")[0]:

                    found.append(node.strip())

        walk(payload)

        if found:

            return found[0]

        raise LtxApiError(
            "Completed job did not contain a downloadable video "
            "URL. Configure api.result_url_fields if the provider "
            "uses a different name. Response keys: "
            f"{self._top_level_keys(payload)}."
        )

    def _download(
        self,
        video_url,
        output_path
    ):

        self._notify(
            "Downloading generated video..."
        )

        output_path = Path(output_path)

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        temporary_path = output_path.with_suffix(
            output_path.suffix + ".part"
        )

        try:

            with requests.get(
                video_url,
                stream=True,
                timeout=self._request_timeout()
            ) as response:

                if response.status_code != 200:

                    raise LtxApiError(
                        "Video download failed: "
                        f"HTTP {response.status_code}."
                    )

                with open(temporary_path, "wb") as handle:

                    for chunk in response.iter_content(
                        chunk_size=1024 * 256
                    ):

                        if chunk:

                            handle.write(chunk)

            temporary_path.replace(output_path)

        finally:

            if temporary_path.exists():

                try:

                    temporary_path.unlink()

                except OSError:

                    pass

        return str(output_path)

    def _submit(
        self,
        prompt
    ):

        base_url = self._base_url()

        submit_path = str(
            self._setting("submit_path", "/v2/text-to-video")
        ).strip() or "/v2/text-to-video"

        submit_url = (
            base_url + "/" + submit_path.lstrip("/")
        )

        payload = {}

        prompt_field = str(
            self._setting("prompt_field", "prompt")
        ).strip() or "prompt"

        payload[prompt_field] = str(prompt).strip()

        model_name = str(self._setting("model", "")).strip()

        if model_name:

            payload["model"] = model_name

        # The LTX async API expects duration (seconds) and resolution
        # ("widthxheight"). These come from the channel content config
        # so the API output matches local output settings.

        send_media_settings = bool(
            self._setting(
                "send_duration_and_resolution",
                True
            )
        )

        if send_media_settings:

            video_output = (
                self.config.get("content", {}).get("video", {})
            )

            try:

                duration = float(
                    video_output.get("duration_seconds", 8)
                )

                payload["duration"] = int(duration)

            except (
                TypeError,
                ValueError
            ):

                pass

            resolution = video_output.get("resolution") or {}

            width = resolution.get("width")
            height = resolution.get("height")

            if width and height:

                payload["resolution"] = f"{width}x{height}"

        extra_params = self._setting("extra_params", {})

        if isinstance(extra_params, dict):

            payload.update(extra_params)

        self._notify(
            "Submitting generation request to LTX Fast API..."
        )

        try:

            response = requests.post(
                submit_url,
                json=payload,
                headers=self._headers(),
                timeout=self._request_timeout()
            )

        except requests.RequestException as error:

            raise LtxApiError(
                f"Could not reach the LTX API at {submit_url}: "
                f"{error}"
            )

        body = self._parse_json(
            response,
            "Generation submission"
        )

        job_id = self._extract_job_id(body)

        self._notify(
            f"Generation accepted (job {job_id}). Waiting for "
            "completion..."
        )

        return job_id

    def generate_clip(
        self,
        prompt,
        output_path
    ):

        base_url = self._base_url()

        job_id = self._submit(prompt)

        status_template = str(
            self._setting(
                "status_path",
                "/generations/{job_id}"
            )
        ).strip() or "/generations/{job_id}"

        status_url = (
            base_url
            + "/"
            + status_template.replace("{job_id}", job_id)
            .lstrip("/")
        )

        completed_statuses = {
            str(value).strip().lower()
            for value in (
                self._setting("completed_statuses", [])
                or [
                    "completed",
                    "succeeded",
                    "success",
                    "finished",
                    "done"
                ]
            )
        }

        failed_statuses = {
            str(value).strip().lower()
            for value in (
                self._setting("failed_statuses", [])
                or [
                    "failed",
                    "error",
                    "canceled",
                    "cancelled"
                ]
            )
        }

        poll_interval = self._poll_interval()

        deadline_seconds = self._deadline_seconds()

        deadline = time.monotonic() + deadline_seconds

        attempt = 0

        status_body = {}

        while True:

            attempt += 1

            remaining = deadline - time.monotonic()

            if remaining <= 0:

                raise LtxApiError(
                    f"Timed out waiting for LTX job {job_id} after "
                    f"{deadline_seconds:g} seconds. Increase "
                    "api.timeout_seconds if the provider is simply "
                    "slow."
                )

            try:

                poll_response = requests.get(
                    status_url,
                    headers=self._headers(),
                    timeout=self._request_timeout()
                )

            except requests.RequestException as error:

                # Transient network failures are retryable until the
                # overall job deadline.

                self._notify(
                    f"Poll request failed (retrying): {error}"
                )

                time.sleep(poll_interval)

                continue

            if poll_response.status_code == 404:

                raise LtxApiError(
                    f"LTX job {job_id} not found or expired "
                    "(results are retained for 24 hours after a job "
                    "finishes)."
                )

            if poll_response.status_code >= 500:

                # Server-side hiccups are retryable until the
                # deadline per the provider's guidance.

                self._notify(
                    f"LTX API returned HTTP "
                    f"{poll_response.status_code}; retrying..."
                )

                time.sleep(poll_interval)

                continue

            status_body = self._parse_json(
                poll_response,
                f"Job {job_id} status"
            )

            status = self._status_value(status_body)

            if status in failed_statuses:

                raise LtxApiError(
                    f"LTX job {job_id} failed. Final status: "
                    f"{status}. {self._error_detail(status_body)}".strip()
                )

            if status in completed_statuses:

                break

            self._notify(
                f"Job {job_id} still {status or 'processing'}... "
                f"(poll cycles: {attempt})"
            )

            time.sleep(poll_interval)

        video_url = self._result_video_url(status_body)

        return self._download(video_url, output_path)

    # --- END OF PART 6 ---
