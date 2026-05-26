# Files To Delete

Generated: 2026-05-18

Purpose: this is a delete-only list. Delete only the paths listed below.

Do not delete, move, archive, or clean up any other files or folders as part of this task.

Do not touch `stages/`.

## Delete These Paths

1. `prototypes/universal_form/`
2. `prototypes/universal_form_v2/`
3. `scraper/danword/.firefox_profile/cache2/`
4. `scraper/danword/.chrome_profile/`
5. `stop_streamlit.bat`

## Verification Notes

- `prototypes/universal_form/` contains only `__pycache__` files.
- `prototypes/universal_form_v2/` contains only `__pycache__` files and cache remnants under `runs/`.
- `scraper/danword/.firefox_profile/cache2/` is browser cache.
- `scraper/danword/.chrome_profile/` is browser profile/cache data.
- `stop_streamlit.bat` is a three-line helper:

```bat
@echo off
taskkill /F /IM streamlit.exe 2>nul
echo Streamlit stopped.
```

## Explicit Non-Goals

Do not process `documents/`.

Do not process `data/`.

Do not process `impressions/`.

Do not process `cryptic_taxonomy/`.

Do not process `logs/`.

Do not process `runs/`.

Do not process root test files.

Do not infer additional deletion candidates.

After deleting only the five listed paths, show `git status --short`.

