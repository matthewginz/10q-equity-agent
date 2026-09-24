@echo off
REM After the nightly backtest: a read-only headless Claude run reviews the
REM results + log and appends one dated entry (run summary, one site idea, one
REM backtest idea, one question for Matthew) to notes/nightly_notes.md.
REM Local only -- notes/ is gitignored, never pushed to the public repo.
REM Read/Grep/Glob only: this run cannot edit code, run commands, or push.
cd /d "%~dp0\.."
if not exist notes mkdir notes
set OUT=notes\.nightly_entry.tmp

REM full path: Task Scheduler's cmd doesn't have the npm global dir on PATH
type scripts\nightly_notes_prompt.md | "%APPDATA%\npm\claude.cmd" -p --model sonnet --allowedTools "Read,Grep,Glob" > %OUT%
if errorlevel 1 (
    echo notes: claude run failed, no entry written
    del %OUT% 2>nul
    exit /b 1
)
for %%A in (%OUT%) do if %%~zA==0 (
    echo notes: empty output, no entry written
    del %OUT%
    exit /b 1
)
echo.>> notes\nightly_notes.md
type %OUT% >> notes\nightly_notes.md
del %OUT%
echo notes: entry appended to notes\nightly_notes.md
