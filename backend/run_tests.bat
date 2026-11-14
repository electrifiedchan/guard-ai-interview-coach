@echo off
REM Runs the backend test scripts against the project venv (which is where the
REM deps live). Kept as a .bat so paths with spaces don't have to survive
REM several layers of shell quoting.
set "ROOT=%~dp0.."
cd /d "%~dp0"
"%ROOT%\venv\Scripts\python.exe" test_session_lifecycle.py
if errorlevel 1 exit /b 1
echo.
"%ROOT%\venv\Scripts\python.exe" test_dashboard_summary.py
if errorlevel 1 exit /b 1
echo.
"%ROOT%\venv\Scripts\python.exe" test_brevity.py
if errorlevel 1 exit /b 1
echo.
"%ROOT%\venv\Scripts\python.exe" test_verdict_narration.py
if errorlevel 1 exit /b 1
echo.
"%ROOT%\venv\Scripts\python.exe" test_persona_contract.py
if errorlevel 1 exit /b 1
echo.
"%ROOT%\venv\Scripts\python.exe" test_llm_config.py
if errorlevel 1 exit /b 1
echo.
"%ROOT%\venv\Scripts\python.exe" test_bea_face_confirmation.py
if errorlevel 1 exit /b 1
echo.
"%ROOT%\venv\Scripts\python.exe" test_episodes.py
if errorlevel 1 exit /b 1
echo.
"%ROOT%\venv\Scripts\python.exe" test_interrupt.py
if errorlevel 1 exit /b 1
echo.
"%ROOT%\venv\Scripts\python.exe" test_peak_risk.py
if errorlevel 1 exit /b 1
echo.
"%ROOT%\venv\Scripts\python.exe" test_resume_gate.py
if errorlevel 1 exit /b 1
echo.
"%ROOT%\venv\Scripts\python.exe" test_timeline_migration.py
if errorlevel 1 exit /b 1
echo.
"%ROOT%\venv\Scripts\python.exe" test_focus_events.py
if errorlevel 1 exit /b 1
echo.
"%ROOT%\venv\Scripts\python.exe" test_focus_consistency.py
if errorlevel 1 exit /b 1
