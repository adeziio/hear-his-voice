import os
import random
import shutil
import time

from pathlib import Path

from dotenv import (
    load_dotenv
)

from selenium import webdriver
from selenium.common.exceptions import (
    WebDriverException
)
from selenium.webdriver.common.action_chains import (
    ActionChains
)
from selenium.webdriver.common.by import (
    By
)

from selenium.webdriver.common.keys import (
    Keys
)

from ai.providers.veo_watermark_remover import (
    VeoWatermarkRemover,
    validate_video_file
)


class SnapGenAiError(
    RuntimeError
):

    """
    Raised for any SnapGenAI automation failure: missing
    credentials, login failure, generation failure, download
    failure, timeouts, and browser problems. The message is always
    safe to surface directly in job state and never contains
    credentials.

    """

    def __init__(
        self,
        message
    ):

        super().__init__(
            message
        )


XPATH_UPPER = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

XPATH_LOWER = "abcdefghijklmnopqrstuvwxyz"

PROMPT_SELECTORS = [
    "textarea",
    "input[type='text']"
]

SUBMIT_SELECTORS = [
    "//button[contains(translate(normalize-space(.), "
    f"'{XPATH_UPPER}', '{XPATH_LOWER}'), 'generate')]",
    "//button[@type='submit']",
    "//input[@type='submit']"
]

DOWNLOAD_SELECTORS = [
    "//a[contains(translate(normalize-space(.), "
    f"'{XPATH_UPPER}', '{XPATH_LOWER}'), 'download')]",
    "//button[contains(translate(normalize-space(.), "
    f"'{XPATH_UPPER}', '{XPATH_LOWER}'), 'download')]",
    "//a[contains(@href, '.mp4')]"
]

LOGIN_EMAIL_SELECTORS = [
    "input[type='email']",
    "input[name='username']",
    "input[autocomplete='username']",
    "input[placeholder*='email' i]"
]

LOGIN_PASSWORD_SELECTORS = [
    "input[type='password']",
    "input[name='password']",
    "input[autocomplete='current-password']"
]

LOGIN_SUBMIT_SELECTORS = [
    "//button[contains(translate(normalize-space(.), "
    f"'{XPATH_UPPER}', '{XPATH_LOWER}'), 'continue')]",
    "//button[contains(translate(normalize-space(.), "
    f"'{XPATH_UPPER}', '{XPATH_LOWER}'), 'sign in')]",
    "//button[contains(translate(normalize-space(.), "
    f"'{XPATH_UPPER}', '{XPATH_LOWER}'), 'log in')]",
    "//button[contains(translate(normalize-space(.), "
    f"'{XPATH_UPPER}', '{XPATH_LOWER}'), 'submit')]",
    "//button[@type='submit']",
    "//input[@type='submit']"
]

LOGIN_URL_MARKERS = (
    "login",
    "signin",
    "sign-in",
    "sign_in",
    "auth",
    "signup",
    "register"
)

PARTIAL_SUFFIXES = (
    ".crdownload",
    ".part",
    ".tmp"
)


def parse_selector(
    selector
):

    """
    Treats selectors starting with "//" as XPath and everything
    else as CSS, so every selector in config/ai_models.json works
    with the same syntax used on the page.
    """

    selector = str(
        selector
    ).strip()

    if selector.startswith("//"):

        return (
            By.XPATH,
            selector
        )

    return (
        By.CSS_SELECTOR,
        selector
    )


class SnapGenAiProvider:

    """
    Browser-automation backend for SnapGenAI
    (https://snapgen.ai/).

    Workflow: open the site -> enter the episode prompt -> click
    Generate -> complete login in a popup/tab when it appears using
    credentials from the environment -> resubmit the prompt when
    authentication dropped the original submission -> wait for the
    generation to finish using element state (no fixed sleeps) ->
    download the video through the browser's download manager ->
    remove the Veo watermark with VeoWatermarkRemover (standard CPU
    mode) and validate the cleaned result.

    When attaching to the dedicated Chrome launched by
    run_windows.bat, that browser window is automatically brought to
    the foreground so the user can watch the generation without
        switching to it by hand. Other Chrome windows are never touched.
        After a fully
    successful generation the same window is minimized again (never
    closed).

    The Chrome profile is persisted across runs so the
    authenticated session/cookies are reused when practical.
    Credentials come only from snapgenai_email / snapgenai_password
    in .env and are never logged or included in error messages.

    All connection details come from config/ai_models.json ->
    models.video_model.snapgenai.
    """

    DEFAULT_BASE_URL = "https://snapgen.ai/"

    DEFAULT_PROFILE_DIRECTORY = (
        "media/browser_profile/snapgenai"
    )

    DEFAULT_DOWNLOAD_DIRECTORY = (
        "media/downloads/snapgenai"
    )

    PROJECT_ROOT = Path(__file__).resolve().parents[2]

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
                "snapgenai",
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

        self.watermark_remover = VeoWatermarkRemover(
            config,
            notify=self._notify
        )

    def _notify(
        self,
        message
    ):

        if self.progress_callback is None:

            return

        try:

            self.progress_callback(
                str(
                    message
                )
            )

        except Exception:

            pass

    def _pause_min(
        self
    ):

        return self._seconds(
            "pause_min_seconds",
            5
        )

    def _pause_max(
        self
    ):

        return self._seconds(
            "pause_max_seconds",
            10
        )

    def _human_pause(
        self
    ):

        # Wait a random amount of time between steps (default 5-10
        # seconds) so the automated flow does not fire every action
        # back-to-back. This is plain pacing - it never tampers
        # with the page, cookies, or any challenge.

        low = self._pause_min()

        high = max(
            self._pause_max(),
            low
        )

        if high <= low:

            high = low + 1

        wait = random.uniform(
            low,
            high
        )

        time.sleep(
            wait
        )

    def _setting(
        self,
        name,
        default=""
    ):

        return self.settings.get(
            name,
            default
        )

    def _seconds(
        self,
        name,
        default
    ):

        try:

            value = float(
                self._setting(
                    name,
                    default
                )
            )

        except (
            TypeError,
            ValueError
        ):

            return float(
                default
            )

        if value <= 0:

            return float(
                default
            )

        return value

    def _flag(
        self,
        name,
        default=False
    ):

        value = self._setting(
            name,
            default
        )

        if isinstance(
            value,
            bool
        ):

            return value

        return str(
            value
        ).strip().lower() in (
            "1",
            "true",
            "yes",
            "on"
        )

    def _base_url(
        self
    ):

        base_url = str(
            self._setting(
                "base_url",
                self.DEFAULT_BASE_URL
            )
        ).strip()

        return (
            base_url
            or self.DEFAULT_BASE_URL
        )

    def _headless(
        self
    ):

        # Default is False so the browser stays visible while the
        # automation is being developed and tested.

        return self._flag(
            "headless",
            False
        )

    def _attach_to_existing_chrome(
        self
    ):

        # When True, Selenium does not launch its own Chrome
        # instance. It attaches to an already-running Chrome
        # through its remote debugging interface, so the real
        # browser session, cookies, and state are reused. This is
        # the recommended mode when Cloudflare's human-verification
        # checkbox loops inside Selenium-driven Chrome.

        return self._flag(
            "attach_to_existing_chrome",
            False
        )

    def _debugging_address(
        self
    ):

        # runner.bat exports SNAPGENAI_DEBUG_HOST/SNAPGENAI_DEBUG_PORT so
        # each project drives its own browser. That matters when several
        # projects run at once: Selenium attaches to a single browser
        # instance, and the download behaviour the provider applies is a
        # browser-wide DevTools setting, so two jobs sharing a browser
        # would redirect each other's downloads mid-run. The environment
        # wins over config for the same reason.
        host = str(
            os.getenv(
                "SNAPGENAI_DEBUG_HOST",
                ""
            )
            or ""
        ).strip()

        port = str(
            os.getenv(
                "SNAPGENAI_DEBUG_PORT",
                ""
            )
            or ""
        ).strip()

        if port:

            return f"{host or '127.0.0.1'}:{port}"

        address = str(
            self._setting(
                "debugging_address",
                ""
            )
        ).strip()

        if not address:

            return "127.0.0.1:9222"

        return address

    def _profile_directory(
        self
    ):

        profile_directory = self._project_path(
            Path(
            str(
                self._setting(
                    "profile_directory",
                    self.DEFAULT_PROFILE_DIRECTORY
                )
            )
            )
        )

        profile_directory.mkdir(
            parents=True,
            exist_ok=True
        )

        return profile_directory

    def _download_directory(
        self
    ):

        download_directory = self._project_path(
            Path(
            str(
                self._setting(
                    "download_directory",
                    self.DEFAULT_DOWNLOAD_DIRECTORY
                )
            )
            )
        )

        download_directory.mkdir(
            parents=True,
            exist_ok=True
        )

        return download_directory

    @classmethod
    def _project_path(
        cls,
        path
    ):

        path = Path(path)

        if not path.is_absolute():
            path = cls.PROJECT_ROOT / path

        return path.resolve()

    def _credentials(
        self
    ):

        email = os.getenv(
            "snapgenai_email",
            ""
        ).strip()

        password = os.getenv(
            "snapgenai_password",
            ""
        ).strip()

        if not email or not password:

            raise SnapGenAiError(
                "SnapGenAI credentials are missing. "
                "Set snapgenai_email and "
                "snapgenai_password in .env."
            )

        return (
            email,
            password
        )

    def _selectors(
        self,
        name,
        fallbacks
    ):

        selectors = []

        configured = str(
            self._setting(
                name,
                ""
            )
        ).strip()

        if configured:

            selectors.append(
                configured
            )

        selectors.extend(
            fallbacks
        )

        return selectors

    def _prompt_selectors(
        self
    ):

        return self._selectors(
            "prompt_selector",
            PROMPT_SELECTORS
        )

    def _submit_selectors(
        self
    ):

        return self._selectors(
            "submit_selector",
            SUBMIT_SELECTORS
        )

    def _download_selectors(
        self
    ):

        return self._selectors(
            "download_selector",
            DOWNLOAD_SELECTORS
        )

    def _login_email_selectors(
        self
    ):

        return self._selectors(
            "login_email_selector",
            LOGIN_EMAIL_SELECTORS
        )

    def _login_password_selectors(
        self
    ):

        return self._selectors(
            "login_password_selector",
            LOGIN_PASSWORD_SELECTORS
        )

    def _login_submit_selectors(
        self
    ):

        return self._selectors(
            "login_submit_selector",
            LOGIN_SUBMIT_SELECTORS
        )

    def _find_visible(
        self,
        driver,
        selectors
    ):

        for selector in selectors:

            by, value = parse_selector(
                selector
            )

            try:

                elements = driver.find_elements(
                    by,
                    value
                )

            except WebDriverException:

                continue

            for element in elements:

                try:

                    if (
                        element.is_displayed()
                        and element.is_enabled()
                    ):

                        return element

                except WebDriverException:

                    continue

        return None

    def _safe_click(
        self,
        driver,
        element
    ):

        try:

            element.click()

        except WebDriverException:

            driver.execute_script(
                "arguments[0].click();",
                element
            )

    def _create_driver(
        self,
        download_directory
    ):

        if self._attach_to_existing_chrome():

            return self._attach_driver(
                download_directory
            )

        options = webdriver.ChromeOptions()

        # A persistent profile keeps the authenticated
        # session/cookies between runs.

        options.add_argument(
            f"--user-data-dir={self._profile_directory()}"
        )

        if self._headless():

            options.add_argument(
                "--headless=new"
            )

            options.add_argument(
                "--disable-gpu"
            )

        options.add_argument(
            "--window-size=1280,1600"
        )

        options.add_argument(
            "--no-first-run"
        )

        options.add_argument(
            "--no-default-browser-check"
        )

        options.add_experimental_option(
            "excludeSwitches",
            ["enable-automation"]
        )

        options.add_experimental_option(
            "prefs",
            {
                "download.default_directory": str(
                    download_directory
                ),
                "download.prompt_for_download": False,
                "download.directory_upgrade": True,
                "safebrowsing.enabled": True
            }
        )

        try:

            driver = webdriver.Chrome(
                options=options
            )

        except WebDriverException as error:

            raise SnapGenAiError(
                "Could not start Chrome for SnapGenAI "
                "automation. Make sure Google Chrome is "
                "installed and not already running with "
                f"the same profile. ({error})"
            )

        driver.set_page_load_timeout(
            self._seconds(
                "page_timeout_seconds",
                60
            )
        )

        return driver

    def _attach_driver(
        self,
        download_directory
    ):

        # Attach mode reuses an already-running Chrome instead of
        # launching a new instance. No user-data-dir, headless, or
        # automation switches are applied here: the attached Chrome
        # keeps its own profile, cookies, and browsing state. No
        # stealth, fingerprint, or CAPTCHA-bypass techniques are
        # used.

        debug_address = (
            self._debugging_address()
        )

        self._notify(
            f"Attaching to the running Chrome at "
            f"{debug_address}..."
        )

        options = webdriver.ChromeOptions()

        options.debugger_address = debug_address

        try:

            driver = webdriver.Chrome(
                options=options
            )

        except WebDriverException as error:

            raise SnapGenAiError(
                "Could not attach to the running Chrome "
                "for SnapGenAI automation. Start Chrome "
                "with remote debugging enabled "
                f"(--remote-debugging-port="
                f"{debug_address.split(':', 1)[-1]}) and "
                "try again; no debugging session was "
                f"found at {debug_address}. ({error})"
            )

        driver.set_page_load_timeout(
            self._seconds(
                "page_timeout_seconds",
                60
            )
        )

        try:

            # Route downloads from the attached session to the
            # configured directory so the download detection
            # below keeps watching the same folder.

            driver.execute_cdp_cmd(
                "Browser.setDownloadBehavior",
                {
                    "behavior": "allow",
                    "downloadPath": str(
                        download_directory
                    )
                }
            )

        except Exception as error:

            self._notify(
                "Could not set the download folder on "
                f"the attached Chrome: {error}"
            )

        self._notify(
            "Bringing the SnapGenAI Chrome window "
            "to the front..."
        )

        # The workflow runs in the dedicated Chrome launched by
        # run_windows.bat. Give that window the foreground focus so
        # the user can watch the generation without switching to it
        # by hand. The calls below only ever target the window this
        # Selenium session is attached to, never other Chrome
        # windows belonging to other browser sessions.

        self._bring_chrome_to_foreground(
            driver
        )

        return driver

    def _bring_chrome_to_foreground(
        self,
        driver
    ):

        # Focus the attached SnapGenAI browser. Every call below only
        # ever targets the dedicated Chrome instance that this Selenium
        # session is attached to, never other Chrome windows.

        if self._headless():

            return

        try:

            current = (
                driver.current_window_handle
            )

            if current:

                driver.switch_to.window(
                    current
                )

            # Tell the browser to bring the page's window to the
            # front and give it keyboard focus.
            driver.execute_cdp_cmd(
                "Page.bringToFront",
                {}
            )

        except Exception:

            pass

        if os.name != "nt":

            return

        try:

            self._bring_windows_chrome_window_to_front()

        except Exception:

            pass

    def _bring_windows_chrome_window_to_front(
        self
    ):

        # Best-effort OS-level focus for the dedicated SnapGenAI
        # Chrome instance on Windows.

        hwnd = (
            self._dedicated_chrome_window_handle()
        )

        if not hwnd:

            return

        self._activate_window_handle(
            hwnd
        )

    def _dedicated_chrome_window_handle(
        self
    ):

        # Resolves the top-level window handle of the dedicated
        # SnapGenAI Chrome instance on Windows, or None. That
        # instance is identified by the remote debugging port and
        # profile directory that run_windows.bat passes on the
        # command line, so a normal Chrome window can never be matched.
        # Shared by the focus and minimize helpers so both always
        # target the same window.

        debug_address = (
            self._debugging_address()
        )

        port = (
            debug_address.split(":", 1)[-1]
            .strip()
        )

        profile = (
            str(
                self._profile_directory()
            )
            .replace("\\", "/")
            .lower()
        )

        if not port or not profile:

            return None

        try:

            import psutil

        except ImportError:

            return None

        target_pids = set()

        for proc in psutil.process_iter(
            [
                "pid",
                "name",
                "cmdline"
            ]
        ):

            try:

                name = (
                    str(
                        proc.info.get("name") or ""
                    )
                    .lower()
                )

                cmdline = (
                    " ".join(
                        proc.info.get("cmdline") or []
                    )
                    .replace("\\", "/")
                    .lower()
                )

            except Exception:

                continue

            if name not in (
                "chrome.exe",
                "msedge.exe"
            ):

                continue

            if (
                f"--remote-debugging-port={port}"
                not in cmdline
            ):

                continue

            if profile not in cmdline:

                continue

            target_pids.add(
                int(
                    proc.info["pid"]
                )
            )

        if not target_pids:

            return None

        return self._find_chrome_window_handle(
            target_pids
        )

    def _minimize_chrome_window(
        self
    ):

        # Best-effort OS-level minimize for the dedicated SnapGenAI
        # Chrome window after a fully successful generation. Only
        # the instance identified by the remote debugging port and
        # profile directory is targeted, never other Chrome windows
        # belonging to another browser session. The browser itself is
        # never closed.

        if self._headless():

            return

        if os.name != "nt":

            return

        try:

            hwnd = (
                self._dedicated_chrome_window_handle()
            )

            if not hwnd:

                return

            import ctypes

            user32 = (
                ctypes.windll.user32
            )

            # SW_MINIMIZE = 6

            user32.ShowWindow(
                int(hwnd),
                6
            )

        except Exception:

            pass

    def _find_chrome_window_handle(
        self,
        target_pids
    ):

        # Returns the first visible top-level window owned by one of
        # the given process ids. Chrome renderers have no visible
        # top-level frame of their own, so this naturally resolves to
        # the main browser window of the dedicated instance.

        import ctypes

        from ctypes import wintypes

        user32 = (
            ctypes.windll.user32
        )

        EnumWindowsProc = (
            ctypes.WINFUNCTYPE(
                wintypes.BOOL,
                wintypes.HWND,
                wintypes.LPARAM
            )
        )

        user32.EnumWindows.argtypes = (
            EnumWindowsProc,
            wintypes.LPARAM
        )

        user32.EnumWindows.restype = (
            wintypes.BOOL
        )

        user32.GetWindowThreadProcessId.argtypes = (
            wintypes.HWND,
            ctypes.POINTER(
                wintypes.DWORD
            )
        )

        user32.GetWindowThreadProcessId.restype = (
            wintypes.DWORD
        )

        user32.IsWindowVisible.argtypes = (
            wintypes.HWND,
        )

        user32.IsWindowVisible.restype = (
            wintypes.BOOL
        )

        user32.GetWindowTextLengthW.argtypes = (
            wintypes.HWND,
        )

        user32.GetWindowTextLengthW.restype = (
            ctypes.c_int
        )

        found = []

        def _enum_callback(
            hwnd,
            lparam
        ):

            if not user32.IsWindowVisible(
                hwnd
            ):

                return True

            pid = (
                wintypes.DWORD()
            )

            user32.GetWindowThreadProcessId(
                hwnd,
                ctypes.byref(pid)
            )

            if (
                int(pid.value)
                not in target_pids
            ):

                return True

            if (
                user32.GetWindowTextLengthW(
                    hwnd
                )
                <= 0
            ):

                return True

            found.append(
                hwnd
            )

            return False

        user32.EnumWindows(
            EnumWindowsProc(
                _enum_callback
            ),
            0
        )

        if not found:

            return None

        return found[0]

    def _activate_window_handle(
        self,
        hwnd
    ):

        # Restores (if minimized), raises, and gives OS-level
        # foreground focus to the window identified by hwnd.

        import ctypes

        from ctypes import wintypes

        user32 = (
            ctypes.windll.user32
        )

        kernel32 = (
            ctypes.windll.kernel32
        )

        user32.IsIconic.argtypes = (
            wintypes.HWND,
        )

        user32.IsIconic.restype = (
            wintypes.BOOL
        )

        user32.ShowWindow.argtypes = (
            wintypes.HWND,
            ctypes.c_int
        )

        user32.ShowWindow.restype = (
            wintypes.BOOL
        )

        user32.BringWindowToTop.argtypes = (
            wintypes.HWND,
        )

        user32.BringWindowToTop.restype = (
            wintypes.BOOL
        )

        user32.GetForegroundWindow.argtypes = ()

        user32.GetForegroundWindow.restype = (
            wintypes.HWND
        )

        user32.SetForegroundWindow.argtypes = (
            wintypes.HWND,
        )

        user32.SetForegroundWindow.restype = (
            wintypes.BOOL
        )

        user32.AttachThreadInput.argtypes = (
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.BOOL
        )

        user32.AttachThreadInput.restype = (
            wintypes.BOOL
        )

        user32.keybd_event.argtypes = (
            wintypes.BYTE,
            wintypes.BYTE,
            wintypes.DWORD,
            ctypes.c_void_p
        )

        user32.keybd_event.restype = None

        kernel32.GetCurrentThreadId.restype = (
            wintypes.DWORD
        )

        if (
            user32.GetForegroundWindow()
            == hwnd
        ):

            return

        if user32.IsIconic(
            hwnd
        ):

            # Restore the window first so it has a real frame to
            # show before it is focused.
            user32.ShowWindow(
                hwnd,
                9
            )

        user32.ShowWindow(
            hwnd,
            5
        )

        user32.BringWindowToTop(
            hwnd
        )

        # Windows normally refuses to let a background process steal
        # the foreground. Simulating a press of the Alt key unlocks
        # the foreground so the focus switch below is accepted.
        user32.keybd_event(
            0x12,
            0,
            0,
            0
        )

        user32.keybd_event(
            0x12,
            0,
            0x0002,
            0
        )

        thread_id = (
            user32.GetWindowThreadProcessId(
                hwnd,
                None
            )
        )

        current_thread = (
            kernel32.GetCurrentThreadId()
        )

        if (
            thread_id
            !=
            current_thread
        ):

            user32.AttachThreadInput(
                current_thread,
                thread_id,
                True
            )

        try:

            user32.SetForegroundWindow(
                hwnd
            )

        finally:

            if (
                thread_id
                !=
                current_thread
            ):

                user32.AttachThreadInput(
                    current_thread,
                    thread_id,
                    False
                )

    def _host_of(
        self,
        url
    ):

        # Reduce a URL to its host, ignoring scheme and any path,
        # so tabs can be matched to the SnapGenAI site.

        host = str(
            url
        ).strip()

        for scheme in (
            "https://",
            "http://"
        ):

            if host.lower().startswith(
                scheme
            ):

                host = host[
                    len(scheme):
                ]

                break

        return (
            host.split("/", 1)[0]
            .split(":", 1)[0]
            .lower()
        )

    def _focus_snapgenai_tab(
        self,
        driver
    ):

        # The attached Chrome may already have several tabs open.
        # Bring the SnapGenAI tab to the foreground so all steps
        # run against the right page. This is plain tab focus, not
        # any kind of automation trickery.

        base_host = self._host_of(
            self._base_url()
        )

        if not base_host:

            return

        try:

            handles = driver.window_handles

            original = driver.current_window_handle

        except WebDriverException:

            return

        if len(handles) <= 1:

            return

        for handle in handles:

            if handle == original:

                continue

            try:

                driver.switch_to.window(
                    handle
                )

                url = str(
                    driver.current_url
                    or ""
                )

            except WebDriverException:

                continue

            if (
                self._host_of(url)
                == base_host
            ):

                self._notify(
                    "Focused the SnapGenAI tab."
                )

                return

        try:

            driver.switch_to.window(
                original
            )

        except WebDriverException:

            pass

    def _aspect_ratio_selectors(
        self,
        label
    ):

        upper = XPATH_UPPER

        lower = XPATH_LOWER

        needle = str(
            label
        ).strip().lower()

        def text_xpath(
            tag
        ):

            return (
                f"//{tag}[contains(translate("
                f"normalize-space(.), '{upper}', "
                f"'{lower}'), '{needle}')]"
            )

        return [
            text_xpath("button"),
            text_xpath("a"),
            (
                "//*[@role='button'][contains(translate("
                f"normalize-space(.), '{upper}', "
                f"'{lower}'), '{needle}')]"
            )
        ]

    def _aspect_ratio_option_selectors(
        self,
        label
    ):

        # Dropdown options are usually menu items or option rows
        # rather than plain buttons, so prefer the common ARIA
        # roles before falling back to generic elements whose
        # visible text contains the label.

        upper = XPATH_UPPER

        lower = XPATH_LOWER

        needle = str(
            label
        ).strip().lower()

        def text_xpath(
            tag
        ):

            return (
                f"//{tag}[contains(translate("
                f"normalize-space(.), '{upper}', "
                f"'{lower}'), '{needle}')]"
            )

        return [
            text_xpath("*[@role='option']"),
            text_xpath("*[@role='menuitem']"),
            text_xpath("*[@role='menuitemradio']"),
            text_xpath("li"),
            text_xpath("button"),
            text_xpath("a"),
            text_xpath("div"),
            text_xpath("span")
        ]

    def _select_aspect_ratio(
        self,
        driver
    ):

        # The generation page lets you pick the output aspect ratio.
        # The trigger starts at "16:9". Clicking it opens a dropdown
        # menu instead of toggling, so the target option (default
        # "9:16") is then selected from that menu. This is a normal
        # UI interaction, not any bypass technique.

        target = str(
            self._setting(
                "aspect_ratio_target",
                "9:16"
            )
        ).strip() or "9:16"

        source_label = "16:9"

        if self._find_visible(
            driver,
            self._aspect_ratio_selectors(
                target
            )
        ) is not None:

            self._notify(
                f"Aspect ratio is already "
                f"{target}."
            )

            return

        source_selector = str(
            self._setting(
                "aspect_ratio_button_selector",
                ""
            )
        ).strip()

        element = None

        if source_selector:

            element = self._find_visible(
                driver,
                [source_selector]
            )

        if element is None:

            element = self._find_visible(
                driver,
                self._aspect_ratio_selectors(
                    source_label
                )
            )

        if element is None:

            raise SnapGenAiError(
                "Could not find the aspect ratio "
                f"button ({source_label}) on the "
                "generation page."
            )

        self._notify(
            f"Selecting {target} aspect ratio..."
        )

        self._safe_click(
            driver,
            element
        )

        self._human_pause()

        # The click opens a dropdown - pick the target entry from
        # it. An explicit selector override is honored first.

        option_selector = str(
            self._setting(
                "aspect_ratio_option_selector",
                ""
            )
        ).strip()

        option = None

        if option_selector:

            option = self._find_visible(
                driver,
                [option_selector]
            )

        if option is None:

            option = self._find_visible(
                driver,
                self._aspect_ratio_option_selectors(
                    target
                )
            )

        if option is None:

            raise SnapGenAiError(
                "Could not find the aspect ratio "
                f"option ({target}) in the dropdown. "
                "Set aspect_ratio_option_selector in "
                "config/ai_models.json if the page "
                "uses a different control."
            )

        self._notify(
            f"Choosing {target} from the "
            "aspect ratio dropdown."
        )

        self._safe_click(
            driver,
            option
        )

        try:

            deadline = time.monotonic() + self._seconds(
                "aspect_ratio_timeout_seconds",
                15
            )

        except TypeError:

            deadline = time.monotonic() + 15

        while time.monotonic() < deadline:

            if self._find_visible(
                driver,
                self._aspect_ratio_selectors(
                    target
                )
            ) is not None:

                self._notify(
                    f"Aspect ratio set to {target}."
                )

                return

            time.sleep(0.5)

        raise SnapGenAiError(
            "The aspect ratio selection was not "
            f"confirmed as {target} within "
            "aspect_ratio_timeout_seconds."
        )

    def _open_generation_page(
        self,
        driver
    ):

        base_url = (
            self._base_url()
        )

        self._notify(
            f"Opening {base_url}..."
        )

        driver.get(
            base_url
        )

        page_timeout = self._seconds(
            "page_timeout_seconds",
            60
        )

        deadline = time.monotonic() + page_timeout

        while time.monotonic() < deadline:

            if driver.find_elements(
                By.TAG_NAME,
                "body"
            ):

                return

            time.sleep(0.5)

        raise SnapGenAiError(
            f"The SnapGenAI page did not load within "
            f"{page_timeout:g} seconds."
        )

    def _enter_prompt(
        self,
        driver,
        prompt
    ):

        self._notify(
            "Entering the episode prompt..."
        )

        timeout = self._seconds(
            "page_timeout_seconds",
            60
        )

        deadline = time.monotonic() + timeout

        while time.monotonic() < deadline:

            element = self._find_visible(
                driver,
                self._prompt_selectors()
            )

            if element is not None:

                try:

                    element.clear()

                    element.send_keys(
                        prompt
                    )

                except WebDriverException:

                    time.sleep(0.5)

                    continue

                if self._prompt_entered(
                    element,
                    prompt
                ):

                    return

            time.sleep(0.5)

        raise SnapGenAiError(
            "Could not enter the prompt into the "
            "SnapGenAI generation page. Adjust "
            "models.video_model.snapgenai."
            "prompt_selector in config/ai_models.json "
            "if the page uses a different field."
        )

    def _prompt_entered(
        self,
        element,
        prompt
    ):

        # Best-effort verification: read the value (inputs) or the
        # text (contenteditable fields) and confirm the prompt is
        # present before submitting.

        try:

            current = (
                element.get_attribute("value")
                or element.text
                or ""
            )

        except WebDriverException:

            return True

        return prompt.strip() in current.strip()

    def _submit_generation(
        self,
        driver
    ):

        self._notify(
            "Submitting the generation request..."
        )

        timeout = self._seconds(
            "submit_timeout_seconds",
            30
        )

        deadline = time.monotonic() + timeout

        while time.monotonic() < deadline:

            element = self._find_visible(
                driver,
                self._submit_selectors()
            )

            if element is not None:

                self._safe_click(
                    driver,
                    element
                )

                return

            time.sleep(0.5)

        raise SnapGenAiError(
            "Could not find the Generate/Submit button "
            "on the SnapGenAI page. Adjust "
            "models.video_model.snapgenai."
            "submit_selector in config/ai_models.json "
            "if the page uses a different button."
        )

    def _login_fields_present(
        self,
        driver
    ):

        # Email OR password counts as a login form, so stepwise
        # login flows (email first, password afterwards) are also
        # detected.

        if self._find_visible(
            driver,
            self._login_email_selectors()
        ) is not None:

            return True

        return self._find_visible(
            driver,
            self._login_password_selectors()
        ) is not None

    def _window_is_login(
        self,
        driver
    ):

        try:

            url = str(
                driver.current_url
                or ""
            ).lower()

        except WebDriverException:

            return False

        if any(
            marker in url
            for marker in LOGIN_URL_MARKERS
        ):

            return True

        return self._login_fields_present(
            driver
        )

    def _wait_for_login_popup(
        self,
        driver,
        original_handle
    ):

        # Bounded, event-based wait for a login tab/popup that may
        # appear after the generation request is submitted.

        timeout = self._seconds(
            "login_popup_wait_seconds",
            15
        )

        deadline = time.monotonic() + timeout

        while time.monotonic() < deadline:

            try:

                handles = driver.window_handles

            except WebDriverException:

                handles = []

            for handle in handles:

                if handle == original_handle:

                    continue

                driver.switch_to.window(
                    handle
                )

                if self._window_is_login(
                    driver
                ):

                    return handle

            driver.switch_to.window(
                original_handle
            )

            time.sleep(0.5)

        return None

    def _login_form_visible(
        self,
        driver
    ):

        timeout = self._seconds(
            "login_form_probe_seconds",
            5
        )

        deadline = time.monotonic() + timeout

        while time.monotonic() < deadline:

            if self._login_fields_present(
                driver
            ):

                return True

            time.sleep(0.5)

        return False

    def _complete_login(
        self,
        driver,
        email,
        password
    ):

        # Credentials are passed only to the browser form fields;
        # they never appear in notifications or error messages.

        self._notify(
            "Login required - signing in to "
            "SnapGenAI..."
        )

        timeout = self._seconds(
            "login_timeout_seconds",
            120
        )

        deadline = time.monotonic() + timeout

        email_field = self._wait_for_login_field(
            driver,
            self._login_email_selectors(),
            deadline
        )

        if email_field is None:

            raise SnapGenAiError(
                "A login page appeared but the SnapGenAI "
                "email field could not be found. Adjust "
                "models.video_model.snapgenai."
                "login_email_selector in "
                "config/ai_models.json if the page uses "
                "a different field."
            )

        self._fill_login_field(
            email_field,
            email
        )

        password_field = self._find_visible(
            driver,
            self._login_password_selectors()
        )

        if password_field is not None:

            # Single-step form: email and password on the same
            # page, one submit at the end.

            self._fill_login_field(
                password_field,
                password
            )

            self._click_login_submit(
                driver,
                password_field
            )

        else:

            # Stepwise form: submit the email (Continue), then
            # wait for the password step to appear. Some accounts
            # sign in directly after the email step - that is
            # handled by the completion check below.

            self._click_login_submit(
                driver,
                email_field
            )

            while time.monotonic() < deadline:

                if not self._login_fields_present(
                    driver
                ):

                    self._notify(
                        "Signed in to SnapGenAI."
                    )

                    return

                password_field = self._find_visible(
                    driver,
                    self._login_password_selectors()
                )

                if password_field is not None:

                    break

                time.sleep(0.5)

            if password_field is None:

                raise SnapGenAiError(
                    "SnapGenAI login did not reach the "
                    "password step after submitting the "
                    "email address."
                )

            self._fill_login_field(
                password_field,
                password
            )

            self._click_login_submit(
                driver,
                password_field
            )

        while time.monotonic() < deadline:

            if not self._login_fields_present(
                driver
            ):

                self._notify(
                    "Signed in to SnapGenAI."
                )

                return

            time.sleep(0.5)

        raise SnapGenAiError(
            "SnapGenAI login failed - the login form "
            "did not clear after submission. Check the "
            "configured credentials."
        )

    def _wait_for_login_field(
        self,
        driver,
        selectors,
        deadline
    ):

        while time.monotonic() < deadline:

            element = self._find_visible(
                driver,
                selectors
            )

            if element is not None:

                return element

            time.sleep(0.5)

        return None

    def _fill_login_field(
        self,
        element,
        value
    ):

        try:

            element.clear()

            element.send_keys(
                value
            )

        except WebDriverException as error:

            raise SnapGenAiError(
                "Could not fill the SnapGenAI login "
                f"form. ({error})"
            )

    def _click_login_submit(
        self,
        driver,
        fallback_field
    ):

        submit_button = self._find_visible(
            driver,
            self._login_submit_selectors()
        )

        if submit_button is not None:

            self._safe_click(
                driver,
                submit_button
            )

            return

        # No recognizable submit control: press Enter in the
        # field, or submit its form as a last resort.

        try:

            fallback_field.send_keys(
                Keys.ENTER
            )

        except WebDriverException:

            fallback_field.submit()

    def _handle_login_if_needed(
        self,
        driver,
        email,
        password
    ):

        original_handle = (
            driver.current_window_handle
        )

        popup_handle = self._wait_for_login_popup(
            driver,
            original_handle
        )

        if popup_handle is not None:

            driver.switch_to.window(
                popup_handle
            )

            self._complete_login(
                driver,
                email,
                password
            )

            driver.close()

            driver.switch_to.window(
                original_handle
            )

            return True

        if self._login_form_visible(
            driver
        ):

            self._complete_login(
                driver,
                email,
                password
            )

            return True

        return False

    def _wait_for_generation(
        self,
        driver
    ):

        # Waits for the generation to finish by looking for the
        # download button once per check interval (default 2
        # minutes). The page is never refreshed; if the download
        # is not ready yet, the check simply repeats after the
        # next interval until the control appears.

        timeout = self._seconds(
            "generation_timeout_seconds",
            1800
        )

        deadline = time.monotonic() + timeout

        # How long to leave the page completely alone right after
        # submit, and how long to wait between download-button
        # checks afterwards. Defaults to 2 minutes.
        check_interval = self._seconds(
            "refresh_interval_seconds",
            120
        )

        failure_selector = str(
            self._setting(
                "failure_selector",
                ""
            )
        ).strip()

        self._notify(
            "Waiting for SnapGenAI generation "
            "to finish..."
        )

        started_at = time.monotonic()

        # Silent settle: do NOTHING on the page for a full check
        # interval (default 2 minutes) right after submit. No
        # polling and no refreshing - the page just sits so the
        # request registers cleanly. This also covers any
        # captcha/verification popup that may have just been
        # clicked past.

        settle_until = (
            started_at
            +
            check_interval
        )

        while time.monotonic() < settle_until:

            if time.monotonic() >= deadline:

                raise SnapGenAiError(
                    "Timed out waiting for SnapGenAI "
                    f"generation after {timeout:g} "
                    "seconds."
                )

            time.sleep(
                min(
                    0.5,
                    max(
                        0.0,
                        settle_until
                        - time.monotonic()
                    )
                )
            )

        # After the quiet period, look for the download control
        # once per check interval. The page is never refreshed -
        # if the download is not ready yet, wait another interval
        # and look again until it appears.

        while True:

            if self._find_visible(
                driver,
                self._download_selectors()
            ) is not None:

                self._notify(
                    "Generation finished - "
                    "download is available."
                )

                return

            if failure_selector:

                if self._find_visible(
                    driver,
                    [failure_selector]
                ) is not None:

                    raise SnapGenAiError(
                        "SnapGenAI reported a "
                        "generation failure on the "
                        "page."
                    )

            if time.monotonic() >= deadline:

                raise SnapGenAiError(
                    "Timed out waiting for SnapGenAI "
                    f"generation after {timeout:g} "
                    "seconds."
                )

            elapsed = int(
                time.monotonic() - started_at
            )

            self._notify(
                f"Still generating... "
                f"({elapsed}s elapsed) - download "
                "not ready yet, checking again "
                f"in {check_interval:g} seconds."
            )

            # Sleep until the next check without touching the
            # page, waking early only if the deadline passes.

            next_check = (
                time.monotonic()
                +
                check_interval
            )

            while True:

                now = time.monotonic()

                if (
                    now >= next_check
                    or
                    now >= deadline
                ):

                    break

                time.sleep(
                    min(
                        0.5,
                        next_check - now,
                        deadline - now
                    )
                )

    def _snapshot(
        self,
        directory
    ):

        return {
            path
            for path in directory.iterdir()
            if path.is_file()
        }

    def _download_result(
        self,
        driver,
        download_directory
    ):

        self._notify(
            "Downloading the generated video..."
        )

        before = self._snapshot(
            download_directory
        )

        element = self._find_visible(
            driver,
            self._download_selectors()
        )

        if element is None:

            raise SnapGenAiError(
                "Generation finished but no download "
                "button or link was found on the page. "
                "Adjust models.video_model.snapgenai."
                "download_selector in "
                "config/ai_models.json if the page "
                "uses a different control."
            )

        self._safe_click(
            driver,
            element
        )

        timeout = self._seconds(
            "download_timeout_seconds",
            300
        )

        deadline = time.monotonic() + timeout

        poll_interval = self._seconds(
            "download_poll_interval_seconds",
            2
        )

        stable_size = None

        stable_polls = 0

        while time.monotonic() < deadline:

            new_files = (
                self._snapshot(download_directory)
                - before
            )

            completed = [
                path
                for path in new_files
                if path.suffix.lower()
                not in PARTIAL_SUFFIXES
            ]

            if completed:

                latest = max(
                    completed,
                    key=(
                        lambda path: (
                            path.stat().st_mtime
                        )
                    )
                )

                size = latest.stat().st_size

                if (
                    size > 0
                    and size == stable_size
                ):

                    stable_polls += 1

                    # Require the file size to settle across
                    # two consecutive polls so partially
                    # written downloads are never accepted.

                    if stable_polls >= 2:

                        validate_video_file(
                            latest
                        )

                        self._notify(
                            f"Video downloaded: "
                            f"{latest.name}"
                        )

                        return latest

                else:

                    stable_size = size

                    stable_polls = 0

            time.sleep(
                poll_interval
            )

        raise SnapGenAiError(
            "Timed out waiting for the SnapGenAI "
            f"video download after {timeout:g} "
            "seconds."
        )

    def generate_clip(
        self,
        prompt,
        output_path
    ):

        """
        Full SnapGenAI workflow for one clip: browser generation ->
        download -> Veo watermark removal (standard CPU mode) ->
        validated cleaned video at output_path. The downloaded
        original is kept until the cleaned output has been
        successfully created.
        """

        prompt = str(
            prompt
        ).strip()

        if not prompt:

            raise SnapGenAiError(
                "The SnapGenAI prompt is empty."
            )

        email, password = (
            self._credentials()
        )

        output_path = Path(
            output_path
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        download_directory = (
            self._download_directory()
        )

        driver = self._create_driver(
            download_directory
        )

        try:

            self._focus_snapgenai_tab(
                driver
            )

            self._open_generation_page(
                driver
            )

            self._human_pause()

            self._select_aspect_ratio(
                driver
            )

            self._human_pause()

            self._enter_prompt(
                driver,
                prompt
            )

            self._human_pause()

            # Deliberately pause after typing the prompt and
            # before clicking submit so the automated flow reads
            # less like a script.

            self._submit_generation(
                driver
            )

            self._human_pause()

            # No retry/resubmission happens after the prompt is
            # submitted. If a login form/tab appears it is handled
            # once; after that we simply wait for the generation to
            # finish and the download button to appear.

            self._handle_login_if_needed(
                driver,
                email,
                password
            )

            self._human_pause()

            self._wait_for_generation(
                driver
            )

            self._human_pause()

            downloaded_file = (
                self._download_result(
                    driver,
                    download_directory
                )
            )

        finally:

            # In attach mode the browser belongs to the user and is
            # reused for the session, cookies, and state, so it must
            # be left running. Only a Selenium-launched Chrome is
            # closed here.

            if not self._attach_to_existing_chrome():

                try:

                    driver.quit()

                except WebDriverException:

                    pass

        staging_path = (
            output_path.parent
            / (
                "_snapgenai_download"
                + (
                    downloaded_file.suffix
                    or ".mp4"
                )
            )
        )

        if staging_path.exists():

            try:

                staging_path.unlink()

            except OSError as error:

                raise SnapGenAiError(
                    f"Could not replace the previous "
                    f"download {staging_path.name}: "
                    f"{error}"
                )

        try:

            shutil.move(
                str(
                    downloaded_file
                ),
                str(
                    staging_path
                )
            )

        except OSError as error:

            raise SnapGenAiError(
                f"Could not store the downloaded video "
                f"in the episode folder: {error}"
            )

        # The remover keeps the original intact and the cleaned
        # output is validated before this call returns, so the
        # original download is only discarded after a verified
        # success.

        cleaned_path = (
            self.watermark_remover
            .remove_watermark(
                staging_path,
                output_path
            )
        )

        try:

            staging_path.unlink()

        except OSError as error:

            self._notify(
                f"Could not remove the original "
                f"download {staging_path.name}: "
                f"{error}"
            )

        # Generation and every post-processing step succeeded.
        # Minimize the dedicated Chrome window that was used so it
        # is out of the way, without closing the browser.

        self._minimize_chrome_window()

        return cleaned_path
