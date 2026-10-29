# ALPminer

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21519943.svg)](https://doi.org/10.5281/zenodo.21519943)

A resumable pipeline that builds a JSON database of structured records from
the scientific literature. It ships configured for ALD process recipes and
ALE etch recipes, and an **extraction profile** system lets you retarget it
at any domain (MOF synthesis, CVD, electrolytes, ...) by declaring your own
fields and prompts — no code changes. The pipeline:

1. **harvest** — pulls metadata for every journal article matching your query
   (default: the active profile's query, e.g. `"atomic layer deposition" OR
   "atomic layer epitaxy"` for `ald`) from **OpenAlex** (free, no key,
   covers essentially all DOI-registered articles), with checkpointed cursor
   pagination.
2. **download** — fetches the **legal open-access PDF** for each paper
   (OpenAlex `best_oa_location`, then Unpaywall). Anything without a working
   OA copy goes to a **manual queue**: you get a clickable HTML/CSV list,
   download those PDFs through your institution's library access, drop them
   into `data/manual_inbox/`, and the tool files them.
3. **extract** — a cheap **triage** pass ("does this paper report the
   authors' own ALD experiment?") followed by full **recipe extraction**
   (structured tool output, validated with pydantic). The model reports
   every quantity with the unit the paper used, and ALPminer converts it
   in code to SI units (°C, s, Pa, nm, nm per cycle, W, eV). Reactant names
   are mapped to one canonical formula and a PubChem compound ID. See
   [Units and chemical names](#units-and-chemical-names). Runs on the LLM
   provider of your choice — **Anthropic**, **OpenAI**, **Google Gemini**,
   or any local/OpenAI-compatible server (Ollama, LM Studio, vLLM) — switch
   with one line in the config, no code changes.
4. **export** — writes `ald_recipes.json` (per-paper, with provenance), plus
   flat `recipes_flat.json` and `recipes_flat.csv`.

Nothing is ever lost on errors or Ctrl-C: **SQLite is the single source of
truth**, every unit of work commits independently, all file writes are atomic,
and **every raw LLM response is cached to disk before parsing** — so a crash,
rate limit, or parsing bug never re-spends tokens. Re-running any command
simply continues from where it stopped.

## Install

Requires **Python >= 3.11**. The simplest way on any OS is the one-click
launcher, which builds the environment and starts the app for you. On Windows
it needs no execution-policy change:

- **Windows** — double-click `start-gui-windows.bat`
- **macOS** — double-click `start-gui-macos.command` in Finder
- **Linux** — run `./start-gui-linux.sh` from a terminal

(See [The dashboard (GUI)](#the-dashboard-gui) below, including the one-time
`chmod +x` note for macOS and Linux.)

To set the environment up by hand instead:

**macOS / Linux (bash/zsh):**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .            # run from the alpminer source folder
```

**Windows (PowerShell):** you do not have to activate the environment, and
skipping activation avoids the "running scripts is disabled on this system"
error entirely. Just call the venv's own executables by path:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .   # run from the source folder
.\.venv\Scripts\alpminer.exe --version           # use alpminer.exe for any command
```

Prefer typing plain `alpminer`? You can activate with
`.\.venv\Scripts\Activate.ps1`; if PowerShell blocks it, allow signed local
scripts for your user once with `Set-ExecutionPolicy -Scope CurrentUser
RemoteSigned`, then activate again. The launcher and the direct-exe commands
above both skip this step.

Confirm it worked with `alpminer --version` (it should print `alpminer 2.0.0`).

**Troubleshooting: `ModuleNotFoundError: No module named
'pydantic_core._pydantic_core'`** (or a similar error mentioning `fitz` /
PyMuPDF). This means the virtual environment was created with one Python
version and is now run with another, so its compiled packages no longer match
the interpreter — most often from re-running `python -m venv .venv` over an
existing `.venv` after a Python upgrade. Rebuild the environment cleanly
(`--clear` wipes the stale packages) and reinstall:

```powershell
py -3 -m venv --clear .venv
.\.venv\Scripts\python.exe -m pip install -e .
```

## The dashboard (GUI)

Everything above is also available without the terminal. This is the easiest
way to start the app:

```bash
alpminer gui          # opens http://127.0.0.1:8642 in your browser
```

**One-click launcher (no terminal needed).** Each OS has a double-clickable
launcher in the code folder that creates or reuses the virtual environment
and then starts the GUI. It rebuilds the environment automatically if the
folder was copied to another computer or Python changed, so it keeps working
after a move:

- **Windows** — double-click `start-gui-windows.bat`.
- **macOS** — double-click `start-gui-macos.command` in Finder. The first
  time, you may need to make it executable once: `chmod +x
  start-gui-macos.command` (a download or unzip can drop the executable bit).
- **Linux** — run `./start-gui-linux.sh` from a terminal (also `chmod +x` it
  once if needed; double-click-to-run depends on your file manager).

Keep the launcher's window open while you use the app; closing it, or pressing
Ctrl-C, stops the server.

On **Windows**, you can also run the command directly in PowerShell once the
venv is activated; if you did not activate it, use the full path instead:

```powershell
.\.venv\Scripts\alpminer.exe gui
```

Run it in your project folder (or an empty one — it offers to set the
project up). The dashboard has six tabs: **Dashboard** (live stage bars,
recipe count, quick run), **Pipeline** (run any stage with its options; a
Stop button safely ends a job — progress commits per paper), **Manual queue**
(open the next N links as browser tabs, auto-file PDFs as they land, point
the watcher at any folder), **Recipes** (search, browse, and inspect every
extracted recipe with a DOI link back to the source), **Settings**
(every `alpminer.toml` option as a form, plus API keys — keys entered here
live only in the running process and are never written to disk or echoed
back), and **About** (a guided walkthrough of the whole workflow, from
first setup to exported dataset). A console strip at the bottom streams live logs from whatever is
running. The server binds to 127.0.0.1 only; `--port` changes the port and
`--no-browser` skips opening a tab.

## Choosing a provider

`alpminer.toml` has a `provider` setting: `"anthropic"` (default), `"openai"`,
`"gemini"`, or `"openai_compatible"` (for local/other servers). Only the
matching API key needs to be set. In the GUI Settings tab, pick the provider
from the dropdown; the two model boxes always show the pair used by the
selected provider.

| | Anthropic | OpenAI | Gemini | OpenAI-compatible (local/other) |
|---|---|---|---|---|
| Cost | pay-as-you-go (roughly $0.03–0.07 per extracted paper) | pay-as-you-go | free tier available | free if local (Ollama/LM Studio) |
| Get a key | [console.anthropic.com](https://console.anthropic.com) → API Keys | [platform.openai.com](https://platform.openai.com) → API keys | [aistudio.google.com/apikey](https://aistudio.google.com/apikey), no credit card | your server's console (none for local) |
| Env var | `ANTHROPIC_API_KEY` | `OPENAI_API_KEY` | `GEMINI_API_KEY` | `OPENAI_API_KEY` (or set `api_key_env`) |
| Default models (extraction / triage) | `claude-sonnet-5-5` / `claude-haiku-5-5` | `gpt-6.1-sol` / `gpt-6-luna` | `gemini-3.8-flash` / `gemini-3.5-flash-lite` | `qwen3.8` / `qwen3.8`, or whatever you name |
| Notes | reference implementation; most tested | uses the Responses API at `api.openai.com` (GPT-6 models call functions only there) | free tier has a small **daily** request quota ([current limits](https://ai.google.dev/gemini-api/docs/rate-limits)) — fine for pilots, too small for large campaigns | Ollama, LM Studio, vLLM, DeepSeek — any `/v1/chat/completions` server with function calling; set `base_url` under `[provider_settings]` |

The defaults are the current models each provider recommends (checked
October 2026). Older names keep working while the provider serves them, but
OpenAI retires `gpt-4o` and `gpt-4o-mini` on 14 January 2027, and since
September 2026 Google serves the Gemini 2.5 models only to accounts that
used them before. A project created with ALPminer 1.0 keeps the model names
written in its `alpminer.toml`, and `alpminer extract` warns when those are
the 1.0 defaults. Update them under `[models.<provider>]`, or use **Reset
settings to factory defaults** in the Settings tab.

Current models think before they answer. `reasoning_effort` (default
`low`) sets how much thinking the extraction model does, and triage always
runs at `low`. Higher levels cost more output tokens; models without a
reasoning control ignore the setting.

Keys are pasted exactly as your provider shows them — alpminer makes no
assumptions about their format. To use Gemini, set `provider = "gemini"` and
put your key in the `GEMINI_API_KEY` environment variable (`export
GEMINI_API_KEY=...` on macOS/Linux, `$env:GEMINI_API_KEY = "..."` in Windows
PowerShell). For an OpenAI-compatible endpoint:

```toml
provider = "openai_compatible"
[provider_settings]
base_url = "http://localhost:11434/v1"   # e.g. Ollama; omit for api.openai.com
# api_key_env = "OPENAI_API_KEY"         # env var holding the key, if any
```

**Fully custom LLMs** are plugins: `alpminer providers new my_llm` scaffolds
`llm_providers/my_llm.py` in your project with a documented, one-method
contract (`call_tool(model, system, user_text, tool, max_tokens) -> dict`).
Implement it, set `provider = "my_llm"`, put any settings it needs under
`[provider_settings]`, and the whole pipeline — caching, retries, resume —
works unchanged. `alpminer providers list` shows what's available.

Everything else (harvest, download, manual queue, export, resumability) is
identical regardless of provider.

## Quick start

**macOS / Linux (bash/zsh):**

```bash
mkdir ald-db && cd ald-db
alpminer init --email you@example.edu   # writes alpminer.toml + data/
export GEMINI_API_KEY="your-key-here"   # or ANTHROPIC_API_KEY / OPENAI_API_KEY

alpminer run --limit 25                 # pilot: 25 papers end to end
alpminer status                         # progress dashboard + token estimate
```

**Windows (PowerShell):**

```powershell
mkdir ald-db; cd ald-db
alpminer init --email you@example.edu   # writes alpminer.toml + data\
$env:GEMINI_API_KEY = "your-key-here"   # or ANTHROPIC_API_KEY / OPENAI_API_KEY

alpminer run --limit 25                 # pilot: 25 papers end to end
alpminer status                         # progress dashboard + token estimate
```

Paste the key exactly as your provider shows it; alpminer makes no assumptions
about its format. The `$env:` form sets the key for the current PowerShell
window only. To keep it across sessions, run `setx GEMINI_API_KEY
"your-key-here"` once and open a **new** window (`setx` does not affect the
window it runs in). The key is read from the environment, never written to
`alpminer.toml`.

Review `data/exports/ald_recipes.json` from the pilot, tune the config if
needed, then scale up (these commands are the same in every shell):

```bash
alpminer run                            # or the stages individually:
alpminer harvest                        # (resumable; safe to interrupt)
alpminer download
alpminer manual list                    #  -> data/exports/manual_queue.html
#   ...download those PDFs via the library, save as <paper_id>.pdf
#   into data/manual_inbox/, then:
alpminer manual ingest
alpminer extract
alpminer export
```

Every command is idempotent — run it as many times as you like; completed
work is skipped, failed work is retried.

## The manual queue

Publishers block bulk scraping and paywalled downloads violate their terms,
so alpminer only auto-downloads **legal open-access copies** and asks you for
the rest. Automating downloads through a university proxy login is the one
thing this tool deliberately does not do: publishers detect scripted proxy
traffic and respond by suspending access for the whole institution.

The queue is populated **as soon as you harvest** — every harvested paper is
"awaiting a PDF" until a file arrives for it, so you can start dropping PDFs
immediately without first running a download pass. Auto-download and manually
filed PDFs both pull papers out of the queue as their files land; a paper the
auto-downloader has already given up on is flagged so you know it genuinely
needs your library access (an untried one just shows "not tried yet").

The manual loop is designed to be fast anyway:

```bash
alpminer manual open --n 10    # opens the next 10 DOI links as browser tabs
# ...click Download on each (through your library access); ANY filename works
alpminer manual watch          # run in a second terminal: files each PDF the
                               # moment it lands, matched by the DOI on page 1
```

No renaming is needed: `watch`/`ingest` match each PDF to its paper by the
DOI printed on its first page (falling back to an exact title match against
any harvested paper still awaiting a PDF, or to the old `<paper_id>.pdf`
filename convention, which is still matched first).
Files the browser is still writing (`.crdownload`, `.part`) are ignored until
complete; an accidentally saved login page is rejected with a warning and
left in place; a PDF for a paper already in the store is reported as a
duplicate.

Watch can cover **several folders at once**, and folders outside the project
inbox are **copy-only** — matched PDFs are copied in and your originals stay
exactly where they were; unrelated PDFs in shared folders are ignored
silently:

```bash
alpminer manual watch --dir "C:\Users\you\Downloads" --include-temp
```

`--include-temp` adds the system temp folder, which catches viewers and
plugins that drop PDF copies there. An honest note on browser internals:
Chrome/Edge's built-in PDF viewer streams into a proprietary cache and does
NOT leave whole `.pdf` files in temp, so "just viewing" a paper is not
reliably capturable — the dependable one-keystroke flow remains Ctrl+S (or
the download button) into any watched folder, where the watcher files it
within seconds.

About a third of the ALD literature has a legal open-access copy (36% of
the articles the default ALD query returns, per OpenAlex in October 2026),
so expect a substantial manual queue if you truly want *every* article —
the pipeline runs fine on the OA subset while you work through it
incrementally.

**Papers not in OpenAlex.** Have a PDF that OpenAlex does not index (a
preprint, a book chapter, an internal report)? Add it directly:

```bash
alpminer add path/to/paper.pdf          # optional: --title "…" --doi 10.x/…
alpminer extract                        # then process it like any other paper
```

The DOI and title are read from the PDF when you omit them. In the GUI, the
**Manual queue** tab has an *Add a paper not in OpenAlex* card that does the
same by file path.

**Curating the queue (GUI).** The **Manual queue** tab lists every paper still
awaiting a PDF, paginated 50 at a time, with a search box (title, DOI,
journal, or id) to jump straight to a paper. Each row has an *Open* button
(opens its DOI/landing page) and a *Remove* button; checkboxes plus *Remove
selected* handle whole pages at once. Removing takes a paper out of the
queue: it stops being counted as awaiting a PDF, is skipped by *Open next
tabs*, and is no longer matched by a dropped PDF — but nothing is deleted.
Removed papers collect in a *Removed papers* list at the bottom of the tab,
where *Restore* (or *Restore all*) brings them back, so removal is always
reversible. The same works from the CLI:

```bash
alpminer manual remove W4406076005 W4406071753   # take papers out (kept)
alpminer manual restore                          # list removed papers
alpminer manual restore --all                    # bring them all back
```

## Extraction profiles (ALD, ALE, or your own domain)

A **profile** defines *what* gets extracted: the record fields (name, type,
description, limits, unit), the triage and extraction prompts, and the
default literature query. Built-ins:

- `ald` — deposition recipes (30 fields: precursors, timings, GPC, ...)
- `ale` — atomic layer etching recipes (28 fields: modification/removal
  chemistry, EPC, etch depth, ion energy, synergy, ...)

Switch with one line in `alpminer.toml` (`profile = "ale"`) or the Settings
tab. To target a new domain — MOF synthesis, perovskites, anything:

```bash
alpminer profiles new mof_synthesis     # writes profiles/mof_synthesis.toml
```

Edit the generated file: each `[[field]]` you declare becomes a slot the LLM
must fill, a validated database column, and a CSV/JSON export column; the
two prompts define relevance and extraction rules in your domain's language.
Give a number field a `unit` (for example `unit = "°C"`) to have ALPminer
convert whatever unit the paper uses, and mark a reactant field
`chemical = true` to get canonical name, formula, and PubChem CID columns
for it (see [Units and chemical names](#units-and-chemical-names)).
Project profiles shadow built-ins of the same name, so you can also copy and
tune `ald` itself. `alpminer profiles list` shows everything available.

You don't have to hand-edit TOML: the **Settings** tab has an *Extraction
rules* editor where you change the OpenAlex query, the triage and extraction
prompts, and **add, remove, or rename the extracted fields** (name, type,
required, unit, description) directly in the browser. Saving writes a project
copy that shadows the built-in, so the 30/28 built-in fields are a starting
point, not a limit. Changing fields affects future extractions; reset the project (or
start a new folder) if you want a clean slate.

**Create a profile under your own name.** Two GUI paths, both ending in a
project profile named by you (snake_case; built-in and already-taken names
are refused), written to `profiles/<name>.toml`, with the project switched to
it:

- **From the dropdown:** pick *＋ Create new profile…* in the Extraction
  profile selector; a panel asks for the name and whether to start from a
  **blank starter template** or a **copy of any existing profile** (built-in
  or your own).
- **From the editor:** tweak the loaded rules and click *Save as new
  profile…* to save the edited state under a new name.

The built-ins are never modified, and your named profiles survive a factory
reset.

**Deleting a profile.** Select a custom profile in the dropdown and click
*Delete this profile…* (shown only for profiles you created). The active
profile must be switched away from first, and built-ins can never be deleted
— a project copy shadowing a built-in's name is reverted via *Reset to
factory settings* instead. Data already extracted with a deleted profile
stays in the database.

**Restore factory settings.** Edited things and want the shipped defaults back?
The Settings tab's *Reset settings to factory defaults* button reverts
everything on that tab — provider, models, query, limits, and any GUI-edited
extraction rules — to what alpminer ships with, keeping your contact email and
data folder. It does **not** touch harvested papers, PDFs, or recipes; that is
the separate *Reset project data* in the Danger zone.

One project = one profile: the database locks to the first profile used, and
extraction refuses a mismatch with instructions (start a new folder for a
new domain). Exports carry the profile name, the unit of every quantity
field, and a versioned schema tag (`alpminer.records.v3`). Raw LLM caches
and stored records from alpminer 1.x still load; their non-SI fields are
renamed and converted on the way (see
[Units and chemical names](#units-and-chemical-names)).

## Configuration (`alpminer.toml`)

| key | default | meaning |
|---|---|---|
| `email` | — (required) | OpenAlex/Unpaywall polite-pool contact |
| `query` | `''` (→ profile default) | title+abstract search (quotes = phrase); empty uses the active profile's default query |
| `from_year` / `to_year` | unset | publication-year window |
| `profile` | `"ald"` | extraction profile: `ald`, `ale`, or your own (see [Extraction profiles](#extraction-profiles-ald-ale-or-your-own-domain)) |
| `provider` | `"anthropic"` | `anthropic`, `openai`, `gemini`, `openai_compatible`, or a plugin name |
| `[models.<provider>]` | shipped pairs | every provider gets the same two slots: `extraction` (reads full papers) and `triage` (cheap relevance filter); switching providers never mixes model names |
| `[provider_settings]` | empty | free-form table passed to `openai_compatible`/plugin providers (`base_url`, `api_key_env`) |
| `triage_enabled` | `true` | disable to send everything to extraction |
| `max_paper_chars` | 250000 | text cap per paper (~62k tokens) |
| `max_output_tokens` | 16000 | raise if a recipe-dense paper truncates (thinking counts toward it) |
| `reasoning_effort` | `"low"` | how much the extraction model thinks: `low`, `medium`, `high`, `xhigh`, or `max` |
| `extract_workers` | 1 | parallel LLM calls during extraction |
| `pubchem_lookup` | `true` | look up reactant names the built-in table does not know on PubChem (cached); `false` keeps extraction offline |
| `request_delay_s` | 1.0 | politeness delay between HTTP requests |

Models are declared per provider, all providers equal:

```toml
[models.anthropic]
extraction = "claude-sonnet-5-5"
triage = "claude-haiku-5-5"

[models.openai_compatible]        # your local Ollama/vLLM/LM Studio server
extraction = "qwen3.8"            # any model your server runs with tool calling
triage = "qwen3.8"
```

(Configs from earlier versions that used flat keys like
`gemini_extraction_model` still load; the next save rewrites them in this
format.) The API key is read from an environment variable only, named after
the provider (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GEMINI_API_KEY`, or
whatever `api_key_env` names), and is never written to disk.

## Scope and cost expectations

With the default queries, OpenAlex currently returns about **23,400**
journal articles for the `ald` profile and about **720** for `ale`, roughly
**24,000** together (checked October 2026). Counting proceedings, preprints,
and other work types as well gives about 33,000. The harvester searches
titles and abstracts. OpenAlex's full-text search matches about 58,000 works
for the ALD query, since it also counts papers that mention ALD only in
their body text. The ALD literature adds about 1,100 to 1,350 articles a
year.

A typical article is ~10–15k input tokens, so full-corpus extraction is a
real budget item (roughly: triage on everything at Haiku rates + extraction
on the ~60–70% that pass triage at Sonnet rates).
`alpminer status` shows the estimated remaining input tokens at any point,
plus the **actual spend so far** — every provider response's reported token
usage is accumulated per project (cached re-runs add nothing), so estimate
and reality can be compared as a run progresses. On large corpora,
`extract_workers = 3` (Settings or `alpminer.toml`) runs that many LLM calls
in parallel to cut wall-clock time; keep it modest to respect rate limits.
Use `from_year`/`to_year` or a narrower query to carve the corpus into
affordable slices; the DB accumulates across runs, and re-running never
re-pays for anything already cached in `data/raw_llm/`. On `provider =
"gemini"`, the practical ceiling is the free-tier daily request quota rather
than dollars — `alpminer extract` will simply need to be re-run on
consecutive days to work through a large corpus once that quota is hit.

## Recipe schema (per recipe)

`material` (required), `technique`, `metal_precursor(+abbrev)`,
`co_reactant(+abbrev)`, `additional_reactants`, `substrate`, `reactor`,
`carrier_gas`, `deposition_temperature_c`, `temperature_window_c`,
`precursor_source_temperature_c`, `pulse/purge_metal_s`,
`pulse/purge_coreactant_s`, `cycle_sequence`, `number_of_cycles`,
`pressure_pa`, `plasma_power_w`, `plasma_gas`, `gpc_nm_per_cycle`,
`gpc_as_reported`, `film_thickness_nm`, `crystallinity`, `application`,
`notes`, `evidence_location` (section/table pointer for spot-checking),
`confidence` (0–1). The field name ends in the unit each value is stored
in. ALPminer adds derived columns: `unit_conversions` (every conversion it
made) and, for each reactant field, `_name`, `_formula`, and `_cid` (for
example `metal_precursor_formula = "AlMe3"`, `metal_precursor_cid =
16682925`). Every exported record also carries source metadata
(`paper_id`, `doi`, `title`, `year`, `journal`) plus an **`ocr` boolean**:
`true` means the paper's text came from OCR rather than the PDF's own text
layer, so OCR-sourced values can be filtered or double-checked.

Extraction rules enforced by the prompt: only processes **performed by the
authors of that paper** (no cited-literature parameters, reviews, or purely
computational work); one recipe per distinct chemistry (parameter sweeps
become a `temperature_window_c`, not N recipes); unknown values stay `null` —
never guessed.

**Derived GPC.** The LLM is told never to infer an unstated value, so a paper
that gives a film thickness and a cycle count but no growth-per-cycle leaves
`gpc_nm_per_cycle` null. After extraction, the pipeline fills that gap
deterministically from `film_thickness_nm ÷ number_of_cycles` and appends an
`auto-derived …` provenance note so a computed rate is never mistaken for a
reported one; `gpc_as_reported` stays empty in that case. A GPC the paper
actually reports (directly or via `gpc_as_reported`) always wins and is
never overwritten. The same rule gives the `ale` profile its etch per cycle
from `etch_depth_nm`, and applies to any custom profile that declares a
rate/thickness/cycles field trio (see `PER_CYCLE_DERIVATIONS`).

## Units and chemical names

ALPminer follows the conventions of the AtomicLimits perspective on AI-ready
ALD/ALE data ([Poupaki et al., *Chem. Mater.* 2026](https://doi.org/10.1021/acs.chemmater.6c02250)).
Every quantity carries a value, a unit, and a quantity kind. Chemical names
follow IUPAC recommendations and are linked to a stable identifier.

**Units.** Each quantity field has one canonical unit, chosen from SI units
and the units IUPAC accepts for use with them
([Green Book](https://iupac.org/what-we-do/books/greenbook/)):

| quantity | unit | fields (ald / ale) |
|---|---|---|
| temperature | °C | `deposition_temperature_c`, `precursor_source_temperature_c`, `temperature_window_c` / `temperature_c`, `temperature_window_c` |
| time | s | pulse, purge, and step times |
| pressure | Pa | `pressure_pa` |
| growth or etch per cycle | nm/cycle | `gpc_nm_per_cycle` / `epc_nm_per_cycle` |
| thickness, depth | nm | `film_thickness_nm` / `etch_depth_nm` |
| power | W | `plasma_power_w` |
| ion energy | eV | `ion_energy_ev`, `ion_energy_window_ev` |
| synergy | % | `synergy_percent` |

The model does not convert anything. It reports each quantity as the paper
states it, for example `{"value": 1.5, "unit": "Torr"}`, and ALPminer
converts the value in code (K, °F, ms, min, mTorr, mbar, kPa, atm, Å, pm,
µm, kW, keV, and more). A bias voltage in V counts as an ion energy in eV,
which holds for singly charged ions. Every conversion is listed in the
record's `unit_conversions` column, for example `pressure_pa: 1.5 Torr ->
199.984 Pa`, including rates read from `gpc_as_reported` and values carried
over from 1.x records. A value whose unit ALPminer cannot read is left null
with the reason written there, and a value the model gave without any unit
is kept in the field's unit and flagged there as well. Ranges such as
`423-573 K` are rewritten in the field's unit (`149.85-299.85`), and any
words after the unit are kept.

The JSON export describes every quantity field with its IUPAC unit symbol
and its [QUDT](https://qudt.org/) unit and quantity-kind IRIs:

```json
"units": {
  "pressure_pa": {"unit": "Pa",
                  "qudt_unit": "http://qudt.org/vocab/unit/PA",
                  "quantity_kind": "http://qudt.org/vocab/quantitykind/Pressure"},
  "gpc_nm_per_cycle": {"unit": "nm/cycle", "per": "cycle",
                       "qudt_unit": "http://qudt.org/vocab/unit/NanoM",
                       "quantity_kind": "http://qudt.org/vocab/quantitykind/Thickness"}
}
```

**Chemical names.** Papers name one reactant many ways (diethylzinc,
diethyl zinc, DEZ, DEZn, Zn(C2H5)2). Each reactant field gains three
columns. `_formula` is the formula with ligand abbreviations, written the
way the AtomicLimits database writes it (`AlMe3`, `ZnEt2`, `Ti(NMe2)4`,
`Hf(NEtMe)4`, `O2 plasma`). `_name` is a canonical name, and `_cid` is the
PubChem compound ID. The text the model extracted stays in the original
field. A built-in table covers about 110 common reactants, chosen from those
the AtomicLimits ALD and ALE databases use most, and every CID in it was
checked against PubChem. Names outside the table go to PubChem's name search when
`pubchem_lookup` is on; answers, misses included, are cached in
`data/pubchem_cache.json`. Only written-out names are sent there, because
PubChem matches some abbreviations to unrelated compounds and stores several
records for some precursors. For those, the table decides. `_formula` comes
from the table only, so a reactant found only on PubChem gets a name and a
CID.

**Projects from alpminer 1.x.** Version 1 stored pressure in Torr (ALE:
mTorr) and growth or etch per cycle in Å/cycle, under the fields
`pressure_torr`, `pressure_mtorr`, `gpc_angstrom_per_cycle`, and
`epc_angstrom_per_cycle`. Raw LLM caches and stored records with those
fields are renamed and converted when they are read, so re-running
`alpminer extract` or `alpminer export` on an old project costs no tokens.
A profile saved by the 1.x rules editor keeps its own fields, units, and
prompts, and its records stay in 1.x units (Å/cycle values are shown in
nm/cycle in the Recipes tab). To move such a project to the 2.0 fields,
use **Reset settings to factory defaults** for a copy of a built-in. For a
profile saved under your own name, rename its quantity fields in the rules
editor (`pressure_pa` with unit Pa, `gpc_nm_per_cycle` with unit nm/cycle).
Either way the stored records are upgraded when they are exported or shown,
so nothing has to be extracted again.

## Failure handling reference

| failure | behavior |
|---|---|
| network drop / 429 / 5xx | exponential-backoff retries, then the item is marked and skipped; batch continues |
| Ctrl-C | clean exit between items; everything completed is committed |
| PDF is an HTML login page | rejected by magic-byte check → manual queue |
| scanned PDF (no text layer) | OCR'd automatically when `pip install "alpminer[ocr]"` + Tesseract are present; otherwise flagged `text: failed` in `status` (not silently dropped) |
| LLM output truncated | marked failed with instruction to raise `max_output_tokens` |
| LLM returns invalid JSON/fields | raw response already cached; paper marked failed; fix/re-run reparses free |
| quantity in a unit ALPminer cannot read | value left null, reason written to `unit_conversions`; the rest of the record is kept |
| model answers without calling the tool | Claude (which cannot be forced to call it) is asked once more in the same run; other providers force the call. Then marked failed and retried next run |
| model declines a paper (safety refusal) | Claude: marked failed with the refusal category, and for some categories the default extraction model first retries on a fallback model. Other providers: marked failed as "no function call" |
| triage output cut off | marked failed with a triage-specific message (triage has a fixed 4096-token budget); use another triage model or turn triage off |
| PubChem unreachable | reactant columns from the built-in table only; lookups stop for the run after three failures |
| extraction failed papers | automatically retried on the next `alpminer extract` |

## Development

```bash
pip install -e ".[dev]"
python -m pytest tests/     # offline (network + LLM calls are mocked) except
                            # one real-browser smoke test, which skips itself
                            # when no working headless browser is found
```

## Release notes (v2.0.0)

This release brings the models up to date and normalizes units and reactant
names following the AtomicLimits perspective (Poupaki et al., *Chem. Mater.*
2026), which cites ALPminer 1.0. The export schema is now
`alpminer.records.v3`.

- **Current models.** New defaults for every provider: Claude Sonnet 5.5
  and Haiku 5.5, GPT-6.1 Sol and GPT-6 Luna, Gemini 3.8 Flash and 3.5
  Flash-Lite, and Qwen3.8 for local servers. The OpenAI provider now uses
  the Responses API, which GPT-6 models need for function calls. The Claude
  backend no longer forces the tool call (the newest Claude models reject
  that), asks once more when a model answers without it, streams responses,
  reports refusals, and retries overloaded and unavailable errors. A new
  `reasoning_effort` setting (default `low`) sets how much the extraction
  model thinks, and triage has room for brief thinking. `max_output_tokens`
  now defaults to 16000. Extraction warns when a config still names the 1.0
  default models.
- **SI units, converted in code.** Quantities are stored in SI units or
  units IUPAC accepts (°C, s, Pa, nm, nm/cycle, W, eV, %). The model
  reports each value with the unit the paper used, and ALPminer converts it
  and lists every conversion in `unit_conversions`. `pressure_torr` and
  `pressure_mtorr` became `pressure_pa`, and `gpc_angstrom_per_cycle` and
  `epc_angstrom_per_cycle` became `gpc_nm_per_cycle` and
  `epc_nm_per_cycle`. Exports describe each quantity with its QUDT unit and
  quantity kind. Caches and records from 1.x are converted when read.
- **Reactant names.** Precursor, co-reactant, and etchant fields gain
  canonical name, AtomicLimits-style formula, and PubChem CID columns, from
  a checked built-in table of about 110 reactants and an optional cached
  PubChem lookup (`pubchem_lookup`).
- **ALE profile.** A new `etch_depth_nm` field lets etch per cycle be
  derived from depth and cycle count, as GPC is for ALD.
- **Corrections.** The literature size above now matches OpenAlex (about
  24,000 ALD and ALE journal articles for the default queries, where 1.0
  said 80–100k), and so does the open-access share (about a third).
  `alpminer providers list` names the OpenAI provider, and PyMuPDF is
  imported under its current name.

## Release notes (v1.0.0)

First public release. ALPminer runs the whole pipeline end to end (harvest,
download, manual queue, triage, extraction, export), with:

- **Any LLM provider, all equal**: Anthropic, OpenAI, Gemini, or a
  local/OpenAI-compatible server (Ollama, LM Studio, vLLM), chosen in one
  line. Models live in per-provider `[models.<provider>]` tables, so
  switching providers never mixes model names; fully custom providers are
  plugins. API keys are read from the environment and never written to disk.
- **Web dashboard**: run every stage, curate the manual queue, browse and
  search recipes, edit extraction rules, and manage profiles from the
  browser. One-click launchers for Windows, macOS, and Linux set the
  environment up for you.
- **Domain profiles**: built-in `ald` and `ale`, plus create, copy, or
  delete your own for any process (each `[[field]]` is at once the LLM slot,
  the validation rule, and the export column).
- **Two-pass extraction**: a cheap triage pass filters irrelevant papers,
  then a stronger model extracts parameters with forced structured output
  validated by pydantic, normalizes units, and derives growth-per-cycle from
  thickness and cycle count when a paper omits it.
- **OCR fallback** for scanned PDFs (inline or deferred), with an `ocr` flag
  on every exported record so OCR-sourced values stay distinguishable.
- **Built for large corpora**: parallel workers, per-paper commits, raw
  response caching, and real token/cost accounting. Every command is
  resumable and idempotent, so a crash, rate limit, or Ctrl-C never
  re-spends tokens or loses progress.
- **Portable projects**: relative paths, PDF paths that self-heal after a
  move, and a `.gitignore` that keeps data and secrets out of the repository.

**Before a large run.** The live OpenAlex, Unpaywall, and provider calls
follow their documented APIs but are exercised against mocks in the test
suite, so run the `--limit 25` pilot first and spot-check
`data/exports/ald_recipes.json` against a few papers you know well (the
`evidence_location` field makes this fast) before committing budget to a
large slice.

## Citation

If you use ALPminer in your research, please cite it. Citation metadata is in
[`CITATION.cff`](CITATION.cff), and GitHub's "Cite this repository" button
(top-right of the repo page) generates APA and BibTeX from it. Once the
release is archived on Zenodo, please cite the DOI:

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21519943.svg)](https://doi.org/10.5281/zenodo.21519943)
