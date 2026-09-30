"""List python processes (pid, start, command line) via PowerShell CIM; optional: kill <pid> [<pid>...]"""
import subprocess
import sys

if len(sys.argv) > 2 and sys.argv[1] == "kill":
    for pid in sys.argv[2:]:
        print(pid, subprocess.run(["taskkill", "/PID", pid, "/F", "/T"], capture_output=True, text=True, errors="replace").stdout.strip())
    sys.exit()
ps = ("Get-CimInstance Win32_Process -Filter \"name='python.exe'\" | "
      "ForEach-Object { \"$($_.ProcessId) | $($_.CreationDate) | $($_.CommandLine)\" }")
out = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True, errors="replace").stdout
for line in out.splitlines():
    if line.strip():
        pid, start, cmd = (line.split(" | ", 2) + ["", ""])[:3]
        print(pid, start[8:14], cmd[-170:])
