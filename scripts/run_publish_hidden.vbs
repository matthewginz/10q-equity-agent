' Task Scheduler entrypoint for 10QAgent-PublishBacktest: publish mid-run backtest
' results (scripts\publish_backtest.py) without a console window. Logs to
' logs\publish.log -- not backtest.log, which the running backtest holds open.
Set objShell = CreateObject("WScript.Shell")
dir = Left(WScript.ScriptFullName, InStrRev(WScript.ScriptFullName, "\"))
objShell.Run "cmd /c """"" & dir & "publish_backtest.bat"" >> """ & dir & "..\logs\publish.log"" 2>&1""", 0, True
