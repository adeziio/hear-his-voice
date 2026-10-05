"""
World English Bible Catholic (WEBC) Scripture source.

This is the single source of truth for every word of Scripture that
Hear His Voice speaks or captions. The LLM never generates, paraphrases,
rewrites, or invents Bible text - it only directs visuals.

SOURCE VERIFICATION
-------------------
The World English Bible is in the Public Domain and explicitly permits
commercial / monetized use. From eBible.org (the publisher of the WEBC
edition):

    "The World English Bible is in the Public Domain. That means that
     it is not copyrighted. However, 'World English Bible' is a Trademark
     of eBible.org. You may copy, publish, proclaim, distribute,
     redistribute, sell, give away, quote, memorize, read publicly,
     broadcast, transmit, share, back up, post on the Internet, print,
     reproduce, preach, teach from, and use the World English Bible as
     much as you want, and others may also do so. All we ask is that if
     you CHANGE the actual text of the World English Bible in any way,
     you not call the result the World English Bible any more."

Because the text must not be changed, this module serves the text
verbatim and never normalizes wording.

WHY NOT Midvash
---------------
Midvash (https://api.midvash.com) was evaluated as the preferred host
and rejected: `GET /v1/versions?language=en` returns 11 English
translations and WEBC is not among them. `GET /v1/versions/webc` returns

    {"error":{"code":"VERSION_NOT_FOUND",
              "message":"Version \"webc\" not found."}}

Midvash only serves `web` (the Protestant 66-book World English Bible),
which is a different translation set from WEBC and therefore does not
satisfy the WEBC requirement. Rather than silently substituting another
translation, the official WEBC USFM release from eBible.org is used.
"""


CACHE_DIRECTORY = "media/cache/scripture"
import io
import re
import zipfile
import urllib.request

from pathlib import Path


# Official WEBC USFM release from eBible.org. The archive contains the
# 76 books in Catholic order, including the deuterocanonical books
# (Tobit, Judith, Greek Esther, Wisdom, Sirach, Baruch, 1-2 Maccabees)
# and the Septuagint additions to Esther and Daniel.
WEBC_URL = (
    "https://ebible.org/Scriptures/eng-web-c_usfm.zip"
)

WEBC_ARCHIVE = "eng-web-c_usfm.zip"

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
)

REQUEST_TIMEOUT_SECONDS = 180


class ScriptureError(
    RuntimeError
):

    pass


# USFM file code -> canonical book name, in Catholic canonical order.
# The numeric prefix in the USFM filename is the USFM book number, which
# is a stable identifier for the file inside the archive.
BOOK_FILE_CODES = {
    "GEN": "Genesis",
    "EXO": "Exodus",
    "LEV": "Leviticus",
    "NUM": "Numbers",
    "DEU": "Deuteronomy",
    "JOS": "Joshua",
    "JDG": "Judges",
    "RUT": "Ruth",
    "1SA": "1 Samuel",
    "2SA": "2 Samuel",
    "1KI": "1 Kings",
    "2KI": "2 Kings",
    "1CH": "1 Chronicles",
    "2CH": "2 Chronicles",
    "EZR": "Ezra",
    "NEH": "Nehemiah",
    "JOB": "Job",
    "PSA": "Psalms",
    "PRO": "Proverbs",
    "ECC": "Ecclesiastes",
    "SNG": "Song of Songs",
    "ISA": "Isaiah",
    "JER": "Jeremiah",
    "LAM": "Lamentations",
    "EZK": "Ezekiel",
    "HOS": "Hosea",
    "JOL": "Joel",
    "AMO": "Amos",
    "OBA": "Obadiah",
    "JON": "Jonah",
    "MIC": "Micah",
    "NAM": "Nahum",
    "HAB": "Habakkuk",
    "ZEP": "Zephaniah",
    "HAG": "Haggai",
    "ZEC": "Zechariah",
    "MAL": "Malachi",
    "TOB": "Tobit",
    "JDT": "Judith",
    "ESG": "Esther",
    "WIS": "Wisdom",
    "SIR": "Sirach",
    "BAR": "Baruch",
    "1MA": "1 Maccabees",
    "2MA": "2 Maccabees",
    # WEBC ships Daniel with the Septuagint additions folded into a
    # single "DAG" (Daniel, Greek) file.
    "DAG": "Daniel",
    "MAT": "Matthew",
    "MRK": "Mark",
    "LUK": "Luke",
    "JHN": "John",
    "ACT": "Acts",
    "ROM": "Romans",
    "1CO": "1 Corinthians",
    "2CO": "2 Corinthians",
    "GAL": "Galatians",
    "EPH": "Ephesians",
    "PHP": "Philippians",
    "COL": "Colossians",
    "1TH": "1 Thessalonians",
    "2TH": "2 Thessalonians",
    "1TI": "1 Timothy",
    "2TI": "2 Timothy",
    "TIT": "Titus",
    "PHM": "Philemon",
    "HEB": "Hebrews",
    "JAS": "James",
    "1PE": "1 Peter",
    "2PE": "2 Peter",
    "1JN": "1 John",
    "2JN": "2 John",
    "3JN": "3 John",
    "JUD": "Jude",
    "REV": "Revelation",
}


BOOK_CODES_BY_NAME = {
    name: code
    for code, name in BOOK_FILE_CODES.items()
}


def _request(url):
    return urllib.request.Request(
        url,
        headers={
            "User-Agent": USER_AGENT
        }
    )


def download_archive(
    cache_directory=CACHE_DIRECTORY
):
    """
    Downloads the WEBC USFM archive once and returns the path to
    the cached zip. A cached copy is reused on later runs.
    """
    cache_directory = Path(cache_directory)
    cache_directory.mkdir(
        parents=True,
        exist_ok=True
    )

    archive_path = (
        cache_directory
        /
        WEBC_ARCHIVE
    )

    if (
        archive_path.is_file()
        and archive_path.stat().st_size > 0
    ):

        return archive_path

    try:

        with urllib.request.urlopen(
            _request(WEBC_URL),
            timeout=REQUEST_TIMEOUT_SECONDS
        ) as response:

            payload = response.read()

    except Exception as error:

        raise ScriptureError(
            f"Could not download the WEBC archive from "
            f"{WEBC_URL}: {error}"
        ) from error

    if not payload:

        raise ScriptureError(
            "The downloaded WEBC archive was empty."
        )

    archive_path.write_bytes(payload)

    return archive_path


def _clean_usfm_fragment(text):
    """
    Removes USFM character markers and translator footnotes from one
    verse fragment while leaving the actual words untouched.

    Three USFM shapes appear in the WEBC release and all must be handled:

    1. Footnotes, which are apparatus rather than Scripture text and are
       emitted inline as "\\f + \\fr 1:2 \\ft ...text... \\f*".
    2. Marked words, either wrapping them
       ("\\w word|strong=\"G1234\"\\w*") or interleaving "\\+w"
       markers with the word itself, sometimes inside a word (the
       apostrophe in "didn't").
    3. Hebrew/Greek source text inside "\\+wh ... \\+wh*" spans, which
       is interlinear apparatus and not part of the English text.
    """
    # Footnotes: \f + \fr 1:1 \ft ... \f*
    #    These are apparatus rather than Scripture text, and they are
    #    always paired in the WEBC release, so the span is dropped.
    text = re.sub(
        r"\\f\s*\+.*?\\f\*",
        "",
        text,
        flags=re.DOTALL
    )

    # Cross references: \x + \xo 1:23 \xt Isaiah 7:14\x*
    #    Same idea, a different marker family.
    text = re.sub(
        r"\\x\s*\+.*?\\x\*",
        "",
        text,
        flags=re.DOTALL
    )

    # Inline cross references left after the above, e.g. "as he said,
    # + 2:6 Amos 8:10 "Your feasts..."" where the quotation that
    # follows IS part of the verse, so only the reference is removed.
    text = re.sub(
        r"\s+\+\s+\d+:\d+[^\u2018\u201c\"]*$",
        "",
        text
    )

    text = re.sub(
        r"\s+\+\s+\d+:\d+(?:\s+[0-9A-Za-z]+(?::\d+[a-z]?)?"
        r"(?:\s*;\s*[0-9A-Za-z]+(?::\d+[a-z]?)?)*\s*)*",
        " ",
        text
    )

    # 2. Original-language spans: \+wh Hebrew \+wh*
    text = re.sub(
        r"\\\+wh\s*.*?\\\+wh\*",
        "",
        text,
        flags=re.DOTALL
    )

    # Marked-up attributes that trail a word, e.g. |strong="G1234"
    text = re.sub(
        r"\s*\|[^\\\s]*",
        "",
        text
    )

    # Every remaining opening/closing USFM character marker.
    text = re.sub(
        r"\\[+a-zA-Z0-9]+\*?",
        "",
        text
    )

    # A marker inside a word ("didn't") left a space before the
    # trailing letter; rejoin those apostrophe constructions.
    text = re.sub(
        r"([\u2019'])\s+([a-zA-Z])",
        r"\1\2",
        text
    )

    # Removing a marker that sat directly after an opening quote leaves
    # a stray space: '" Rabbi' -> '"Rabbi'.
    text = re.sub(
        r"([\u201c\u2018])\s+",
        r"\1",
        text
    )

    text = re.sub(
        r"[ \t]+",
        " ",
        text
    )

    return text.strip()


# The conventional abbreviations viewers type for each book, mapped
# onto the USFM code. Matching is explicit rather than heuristic so that
# "Jn" always means John and never Jonah.
BOOK_ABBREVIATIONS = {
    "gen": "GEN",
    "ex": "EXO",
    "exod": "EXO",
    "lev": "LEV",
    "num": "NUM",
    "deut": "DEU",
    "jos": "JOS",
    "judg": "JDG",
    "ruth": "RUT",
    "1sa": "1SA",
    "2sa": "2SA",
    "1kings": "1KI",
    "2kings": "2KI",
    "1ki": "1KI",
    "2ki": "2KI",
    "1chron": "1CH",
    "2chron": "2CH",
    "1ch": "1CH",
    "2ch": "2CH",
    "ezra": "EZR",
    "neh": "NEH",
    "esther": "ESG",
    "est": "ESG",
    "job": "JOB",
    "ps": "PSA",
    "psa": "PSA",
    "pss": "PSA",
    "psalm": "PSA",
    "psalms": "PSA",
    "psalter": "PSA",
    "prov": "PRO",
    "pr": "PRO",
    "eccl": "ECC",
    "eccles": "ECC",
    "song": "SNG",
    "songofsolomon": "SNG",
    "canticles": "SNG",
    "isa": "ISA",
    "jer": "JER",
    "lam": "LAM",
    "lamentations": "LAM",
    "ezek": "EZK",
    "hos": "HOS",
    "Joel": "JOL",
    "amos": "AMO",
    "obad": "OBA",
    "jonah": "JON",
    "mic": "MIC",
    "nah": "NAM",
    "hab": "HAB",
    "zeph": "ZEP",
    "hag": "HAG",
    "zech": "ZEC",
    "mal": "MAL",
    "tob": "TOB",
    "judith": "JDT",
    "wis": "WIS",
    "wisdom": "WIS",
    "sirach": "SIR",
    "sir": "SIR",
    "ecclus": "SIR",
    "baruch": "BAR",
    "tob": "TOB",
    "1mac": "1MA",
    "2mac": "2MA",
    "dan": "DAG",
    "daniel": "DAG",
    "mt": "MAT",
    "matt": "MAT",
    "matthew": "MAT",
    "mk": "MRK",
    "mark": "MRK",
    "marcus": "MRK",
    "luke": "LUK",
    "lk": "LUK",
    "jn": "JHN",
    "jhn": "JHN",
    "john": "JHN",
    "acts": "ACT",
    "rom": "ROM",
    "ro": "ROM",
    "1cor": "1CO",
    "2cor": "2CO",
    "gal": "GAL",
    "eph": "EPH",
    "phil": "PHP",
    "php": "PHP",
    "col": "COL",
    "1thess": "1TH",
    "2thess": "2TH",
    "1tim": "1TI",
    "2tim": "2TI",
    "tit": "TIT",
    "philem": "PHM",
    "heb": "HEB",
    "jas": "JAS",
    "james": "JAS",
    "1pet": "1PE",
    "2pet": "2PE",
    "1john": "1JN",
    "2john": "2JN",
    "3john": "3JN",
    "jude": "JUD",
    "rev": "REV",
    "revelation": "REV",
}


class WEBCScripture:
    """
    Reads exact WEBC text out of the official USFM release.

    The archive is read once per process and the parsed books are kept in
    memory, so selecting a passage for an episode costs no extra network
    calls after the first.
    """

    def __init__(
        self,
        cache_directory=CACHE_DIRECTORY
    ):

        self.cache_directory = cache_directory

        self._books = None

    @property
    def books(self):
        """
        {book_name: {chapter: {verse: text}}} for the whole WEBC.
        """
        if self._books is None:

            self._books = self._load_books()

        return self._books

    def _load_books(self):
        archive_path = download_archive(
            self.cache_directory
        )

        try:

            archive = zipfile.ZipFile(
                archive_path
            )

        except Exception as error:

            raise ScriptureError(
                f"The cached WEBC archive could not be opened: "
                f"{error}"
            ) from error

        books = {}

        with archive:

            for name in archive.namelist():

                if not name.endswith(".usfm"):

                    continue

                stem = name[:-len(".usfm")]

                # Files are named like "73-JHNeng-web-c": the USFM book
                # number, then the three letter book code.
                code_match = re.search(
                    r"(\d{2}-[A-Z0-9]{3})",
                    stem
                )

                if code_match is None:

                    continue

                code = code_match.group(1).split("-")[1]

                if code not in BOOK_FILE_CODES:

                    continue

                source_text = archive.read(
                    name
                ).decode(
                    "utf-8",
                    "ignore"
                )

                books[BOOK_FILE_CODES[code]] = parse_usfm(
                    source_text
                )

        if not books:

            raise ScriptureError(
                "The WEBC archive contained no readable USFM books."
            )

        return books

    def get_verse(
        self,
        book,
        chapter,
        verse
    ):
        """
        Returns the exact WEBC wording of one verse, or None.

        The book may be given as a full name or a common abbreviation.
        """
        try:

            book = self._resolve_book(
                book
            )

        except ScriptureError:

            return None

        chapters = self.books.get(
            book
        )

        if chapters is None:

            return None

        return chapters.get(
            str(chapter),
            {}
        ).get(str(verse))

    def get_passage(
        self,
        book,
        chapter,
        start_verse,
        end_verse=None
    ):
        """
        Returns the exact WEBC wording of a verse range as a list of
        {"verse": number, "text": text} entries in order.

        The text is served verbatim. Nothing here paraphrases, rewrites,
        or summarises Scripture.
        """
        chapters = self.books.get(
            book
        )

        if chapters is None:

            raise ScriptureError(
                f"'{book}' is not a WEBC book. "
                "Check the reference."
            )

        verses = chapters.get(
            str(chapter)
        )

        if verses is None:

            raise ScriptureError(
                f"{book} {chapter} is not a WEBC chapter."
            )

        end_verse = (
            end_verse
            if end_verse is not None
            else start_verse
        )

        passage = []

        for number in range(
            int(start_verse),
            int(end_verse) + 1
        ):

            text = verses.get(str(number))

            if text is None:

                continue

            passage.append({
                "verse": number,
                "text": text
            })

        if not passage:

            raise ScriptureError(
                f"{book} {chapter}:{start_verse}-{end_verse} "
                "returned no WEBC text."
            )

        return passage

    def get_reference_text(
        self,
        reference
    ):
        """
        Resolves a "Book Chapter:Verse[-Verse]" reference string to
        {"reference", "book", "chapter", "translation", "passage",
        "text"}. The text is the verbatim WEBC wording.
        """
        match = re.match(
            r"^\s*(.+?)\s+(\d+):(\d+)(?:\s*-\s*(\d+))?\s*$",
            str(reference)
        )

        if match is None:

            raise ScriptureError(
                f"'{reference}' is not a valid Scripture reference. "
                "Use the form 'Book Chapter:Verse' or "
                "'Book Chapter:Verse-Verse'."
            )

        book = self._resolve_book(
            match.group(1)
        )

        chapter = match.group(2)
        start_verse = match.group(3)
        end_verse = match.group(4)

        passage = self.get_passage(
            book,
            chapter,
            start_verse,
            end_verse
        )

        return {
            "reference": (
                f"{book} {chapter}:{start_verse}"
                + (
                    f"-{end_verse}"
                    if end_verse
                    else ""
                )
            ),
            "book": book,
            "chapter": chapter,
            "translation": "World English Bible Catholic (WEBC)",
            "passage": passage,
            "text": " ".join(
                item["text"]
                for item in passage
            ),
        }

    def _resolve_book(
        self,
        name
    ):
        """
        Maps a user-supplied book name onto a WEBC book name.
        """
        wanted = re.sub(
            r"\s+",
            " ",
            str(name).strip().lower()
        )

        for book in self.books:

            if book.lower() == wanted:

                return book

        # Accept the conventional abbreviations first. The mapping is
        # explicit, so "Jn" resolves to John and never to Jonah.
        key = re.sub(
            r"[^a-z0-9]",
            "",
            str(name).strip().lower()
        )

        code = BOOK_ABBREVIATIONS.get(key)

        if code is not None:

            return BOOK_FILE_CODES[code]

        # Then fall back to the full book name, ignoring spaces.
        squashed = re.sub(
            r"[^a-z]",
            "",
            str(name).strip().lower()
        )

        for book in self.books:

            if re.sub(
                r"[^a-z]",
                "",
                book.lower()
            ) == squashed:

                return book

        raise ScriptureError(
            f"'{name}' did not match a WEBC book."
        )


def parse_usfm(source_text):
    """
    Parses one USFM book into an ordered dict of chapters, where each
    chapter maps verse number -> verse text.

    In the poetic books the "\\q1"/"\\q2" lines are acrostic markers: they
    carry the next line of the verse that is already in progress rather
    than starting a new verse. Their text is therefore appended to the
    current verse.
    """
    chapters = {}
    chapter = None
    current_verse = None

    for line in source_text.splitlines():

        line = line.rstrip()

        if not line.strip():

            continue

        if line.startswith("\\c "):

            chapter = line[3:].strip()
            current_verse = None

            chapters.setdefault(
                chapter,
                {}
            )

            continue

        if chapter is None:

            continue

        # "\q1"/"\q2" acrostic markers only label a poetic verse's
        # lines; the words that follow belong to the verse in progress.
        unwrapped = re.sub(
            r"\\q\d+\s*",
            " ",
            line
        )

        if unwrapped.startswith("\\v "):

            raw_verse = unwrapped[3:].strip()

            # Split the leading verse number from the text. The number
            # may be a single verse or a range such as "16-18" or "1a".
            match = re.match(
                r"^(\d+[a-z]?(?:-\d+[a-z]?)?)\s+(.*)$",
                raw_verse,
                re.DOTALL
            )

            if match is None:

                continue

            number = match.group(1)

            # A long verse wraps onto several source lines, each of
            # which repeats the same "\v N" marker. A repeated number
            # means the verse continues rather than restarts.
            if (
                number == current_verse
                and number in chapters[chapter]
            ):

                extra = _clean_usfm_fragment(
                    match.group(2)
                )

                if extra:

                    chapters[chapter][number] = (
                        f"{chapters[chapter][number]} {extra}"
                    )

                continue

            body = _clean_usfm_fragment(
                match.group(2)
            )

            if not body:

                continue

            current_verse = number

            chapters.setdefault(
                chapter,
                {}
            )[number] = body

            continue

        # Any other line inside a chapter that is not a verse marker is
        # a continuation of the verse in progress (this is how the
        # poetic books split a verse across lines).
        if current_verse is None:

            continue

        body = _clean_usfm_fragment(
            unwrapped
        )

        if not body:

            continue

        if current_verse in chapters[chapter]:

            chapters[chapter][current_verse] = (
                f"{chapters[chapter][current_verse]} {body}"
            )

    return chapters