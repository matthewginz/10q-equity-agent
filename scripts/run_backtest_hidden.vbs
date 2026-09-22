Set objShell = CreateObject("WScript.Shell")
objShell.Run """" & Left(WScript.ScriptFullName, InStrRev(WScript.ScriptFullName, "\")) & "run_backtest_daily.bat""", 0, True
