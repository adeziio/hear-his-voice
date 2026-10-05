# ✝️ Hear His Voice

Hear His Voice is an automated short-form video studio that lets Scripture
speak for itself. Each episode takes one passage of the **World English Bible
Catholic (WEBC)**, speaks the exact words, and builds a vertical film around
them: stock footage, narration, synchronized captions, and background music.

---

## 🎯 What it does

```text
WEBC Scripture source
        ↓  exact, verbatim passage
AI creative direction (title, mood, per-segment visual queries)
        ↓
Pexels footage → narration (TTS) → synchronized captions → music
        ↓
Final vertical MP4
        ↓
Publish: YouTube Shorts / Instagram Reels
```

The **Scripture source is the source of truth**. The AI never writes, rewrites,
paraphrases, or invents Bible text — it only decides what the viewer sees while
the exact words are spoken.

---

## 📖 The Scripture source

| | |
|---|---|
| **Translation** | World English Bible Catholic (**WEBC**), eBible.org edition `eng-web-c` |
| **Source** | `https://ebible.org/Scriptures/eng-web-c_usfm.zip` |
| **License** | Public Domain |
| **Commercial / monetized use** | **Explicitly permitted** |
| **Caching** | `media/cache/scripture/eng-web-c_usfm.zip` |

### Commercial-use permission (verified)

The publisher's own statement:

> "The World English Bible is in the Public Domain. That means that it is not
> copyrighted. However, 'World English Bible' is a Trademark of eBible.org.
> You may copy, publish, proclaim, distribute, redistribute, **sell**, give
> away, quote, memorize, read publicly, broadcast, transmit, share, back up,
> post on the Internet, print, reproduce, preach, teach from, and use the
> World English Bible as much as you want, and others may also do so. All we
> ask is that if you **CHANGE** the actual text of the World English Bible in
> any way, you not call the result the World English Bible any more."

Because monetized channels are the point of this project, this was verified
rather than assumed. The text is served **verbatim and never edited**.

### Why not Midvash?

Midvash (`https://api.midvash.com`) was the preferred host and was evaluated
first. It does **not** serve WEBC:

- `GET /v1/versions?language=en` → 11 English translations, WEBC not among them
- `GET /v1/versions/webc` → `{"error":{"code":"VERSION_NOT_FOUND", ...}}`

Midvash only serves `web`, the **Protestant** 66-book World English Bible,
which is a different translation set. Rather than silently substituting another
translation, the official WEBC USFM release from eBible.org is used.

---

## 🧠 The AI is the director, not the author

The AI response schema (`DIRECTION_SCHEMA` in `ai/content_generator.py`)
deliberately contains **no field for Scripture text**. It can only return:

| Field | Purpose |
|---|---|
| `title` | Episode title |
| `summary` | Episode-listing description |
| `mood` | 1–3 words used to select background music |
| `visuals[].search_query` | A practical Pexels search phrase per segment |
| `visuals[].visual_direction` | One sentence describing the shot |

The spoken segments are produced **deterministically** by
`scripture/segmenter.py` from the WEBC text, and `assemble_content()` writes
`content["narration"]` straight from the source. There is no code path by which
model output can reach the narration or the captions.

This is verified by tests, including one that feeds the assembler a deliberately
hostile response and asserts the narration still matches WEBC byte for byte.

---

## 🎬 Production architecture

Hear His Voice reuses the **same proven production architecture** as the
Curious About Things project. The `production/` package is shared:

| Stage | Module | Notes |
|---|---|---|
| Scripture fetch | `scripture/webc.py` | WEBC USFM download, parse, exact verse text |
| Passage choice | `scripture/selector.py` | Rotates references; stores only references |
| Segmentation | `scripture/segmenter.py` | Sentence splitting into spoken segments |
| Creative direction | `ai/content_generator.py` | AI returns visual direction only |
| Footage | `production/footage/pexels.py` | Search, download, de-duplicate, segment matching |
| Narration | `production/narration.py` | edge-tts + word-level timings |
| Captions | `production/captions.py` | Phrase-based cues + SRT |
| Music | `production/music/freesafemusic.py` | Licensed for commercial use |
| Composition | `production/composer.py` | Timing, mixing, render, validation |
| Orchestration | `production/video.py`, `core/pipeline.py` | Stage flow and progress |
| Web UI / worker | `web/server.py`, `web/job_worker.py` | Two-stage async job flow |

### Runtime

Episodes aim for **about 50 seconds**, set by
`config/app.json` → `shorts.target_duration_seconds`.

The target is reached by **choosing how much Scripture to speak**, not by
changing the pace. A configured passage is a *window*: the episode starts at its
first verse and then takes as many of the following verses as it needs to land
near the target, stopping at whichever verse boundary is nearer.

| | |
|---|---|
| `John 3:16-30` window | → speaks `John 3:16-21` (~50s) |
| `Mark 4:35-41` window | → speaks `Mark 4:35-41` (~51s) |
| `Matthew 25:1-13` window | → speaks `Matthew 25:1-9` (~46s) |

Only **whole verses** are used, because cutting one in half would break its
meaning. That also means the 50 seconds is a guide rather than a limit — the
result sits either side of it whenever the verse boundaries do not line up, and
a complete short teaching such as `John 14:6` stays short rather than being
padded with unrelated verses. Nothing is ever sped up, slowed down, re-spoken,
trimmed, or joined to another passage to reach the number.

The narration and the captions remain the verbatim WEBC text whichever verses
are chosen.

### Captions

Captions are **meaningful phrases** synchronized to the narration, built from the
real word timings. They are never word-by-word and never a lone word. A trailing
one-word cue is folded back into the previous cue so every caption on screen is
a complete phrase.

---

## ⚙️ Configuration

The project is **configuration-driven**.

| File | Purpose |
|---|---|
| `config/app.json` | **Target duration**, resolution, fps, narration voice, music volume, caption styling |
| `config/content.json` | Channel voice, visual rules, creative direction, **and all Scripture settings** — translation, source URL, licence, narration word rate, and the rotation references |
| `config/ai_models.json` | Ollama model and sampling options |
| `config/pexels.json` | Footage provider settings, with format-specific ones under `shorts` |
| `config/freesafemusic.json` | Music provider settings |

**One number drives the length.** `app.json` → `shorts.target_duration_seconds`
is the only definition of how long an episode is; no other config repeats it.
Everything downstream derives from it, the same way the other project does:

```text
word_target       = target_duration_seconds × words_per_second
words_per_segment = word_target ÷ segments_per_episode
```

`shorts.segments_per_episode` is the number of spoken segments — and therefore
visuals — per episode, set to `14` to match the other project's 14 narration
sentences and 14 visuals. Segment *size* is derived, so changing `50` to `150`
gives each visual three times the room without touching anything else.

## 📐 Format sections

Settings that only make sense for one kind of output live under that format's own
parent, so a second format can ask for a different shape without shadowing the
first. `app.json` already had one for `shorts`; `pexels.json` now has the same
shape for the settings that shape the footage search:

```json
"shorts": { "orientation": "vertical", "resolution_name": "4K" }
```

Everything above that in `pexels.json` — timeouts, pauses, and the filter
recovery settings — applies to the provider whatever the format. A `videos`
section can be added later with no code change.

How far a passage can actually reach is bounded by the size of its window in
`content.json` → `scripture.references` — a window is what the passage is allowed
to read, and the selector never reads past its end. So a large target is only
useful with windows large enough to fill it.

## 📖 Which passages

**The channel reads the four Gospels straight through**, in order, rotating
between them:

```
Matthew → Mark → Luke → John → Matthew → ...
```

Each Gospel keeps its **own position**. When the rotation lands on a book, that
episode continues from the first verse after the last verse the *previous*
episode in *that same book* finished. So the four books advance on four
independent tracks while the rotation interleaves them.

Nothing is ever skipped or repeated. The only way a verse is read twice is an
explicit reset (`PassageSelector().reset()`, or deleting the state file).
Progress lives in `state/scripture_rotation.json` and survives restarts, so
reading is continuous across separate pipeline runs. That file is deliberately
**not** under `media/` — `media/` is git-ignored for generated output, and the
reading position is meaningful, long-lived project state. Keeping it in
`state/` means the channel's progress through the Gospels is committed to the
repository and visible to anyone who clones it.

The four-book limit is the one hard rule, enforced in code
(`GOSPEL_BOOKS` / `ROTATION_ORDER` in `scripture/selector.py`) rather than by
any list — a pasted Psalm or Acts reference fails loudly at startup.

**No passage list drives selection.** `content.json` →
`scripture.guidance_examples` holds example passages showing the *kind* of
material wanted — teachings, parables, miracles, the arc of his life and death
and resurrection — for whoever is writing prompts or reviewing tone. Removing
that list changes nothing about what plays. Because the pool is the Gospel text
itself, the channel naturally covers far more than Jesus' spoken words.

Each episode takes as many **consecutive whole verses** as the target duration
calls for. A passage may run across a chapter boundary rather than stopping
short — that is what stops a verse opening near the end of a chapter from
becoming a stub — and its reference then names both chapters.

The narration and captions are always the **exact WEBC text**, read from the
cached eBible.org archive. Nothing in the pipeline generates, paraphrases,
condenses, or "improves" Scripture — the AI is only ever asked for visual
direction and a title.

To see where the reading has reached:

```python
selector.progress()
# {'next_book': 'Luke',
#  'rotation_order': ['Matthew', 'Mark', 'Luke', 'John'],
#  'books': {'Matthew': {'next_chapter': '9', 'next_verse': 3,
#                        'cycles': 0, 'verses': 1071}, ...}}
```

## ✝️ Captions

Captions show **whole phrases**, not one word at a time, which is a deliberate
contrast with the other project's punchy single-word pop-ups. This channel is
devotional, so the type is calmer and smaller (`font_size` 48 vs 65) to fit more
words on screen at once. Because a seven-word phrase is much wider than the
frame, the composer word-wraps every cue to a safe area and shrinks the font
only if a single long word still will not fit — words are never broken in half.
---

## ▶️ Running it

```bash
python main.py                    # next unplayed passage
python main.py "John 3:16-18"     # a specific passage
python -m web.server              # web UI
```

### Tests

```bash
python -m pytest tests/ -q          # unit + integration
python tests/live_content_test.py   # real WEBC + real LLM
python tests/live_video_test.py     # real WEBC + real TTS + real render
```

`live_video_test.py` stubs only the Pexels download (which needs an interactive
browser); narration, captions, timing, music, composition and rendering are all
the real code running on the real passage.

---

## 📁 Episode output

```text
media/output/shorts/004/
├── content.json          # source of truth: reference, translation, narration
├── prompt.txt            # human-readable record
├── audio/
│   ├── narration.mp3     # TTS of the exact WEBC text
│   └── words.json        # word timings
├── captions/
│   └── captions.srt      # phrase captions
├── footage/              # one folder per visual segment
├── music/
│   └── metadata.json     # track, artist, license
└── episode.mp4           # final 1080×1920 vertical video
```

---

## 🧹 Retired

The earlier **SnapGenAI / LTX / diffusers** text-to-video path has been removed.
Text-to-video models cannot guarantee verbatim Scripture, cannot reproduce
brand and copyrighted text reliably, and cannot produce accurate captions — so
the channel now uses real stock footage, real narration, and real captions
instead.
| `config/freesafemusic.json` | Music provider settings |

Narration defaults to a calm, measured delivery
(`en-US-AndrewMultilingualNeural`) at `-2%` with music at `0.07` volume, so the
Word always sits clearly above the score.
### Why WEBC and not the Protestant `web`

WEBC is the Catholic edition. The tests assert the distinguishing markers:

- **73 books** in Catholic order, including the deuterocanonical books —
  Tobit, Judith, Greek Esther, Wisdom, Sirach, Baruch, 1 & 2 Maccabees.
- **`LORD`**, not `Yahweh` — WEBC follows Catholic worship practice.