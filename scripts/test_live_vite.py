"""
Tests live Vite dev server startup and HTTP probe.
"""

import subprocess
import urllib.request
import time
import sys
import os
import signal

def main():
    print("Starting Vite dev server in frontend/...")
    proc = subprocess.Popen(
        ["npm", "run", "dev"],
        cwd="frontend",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=True,
    )
    url = "http://localhost:5173"
    print(f"Waiting for Vite dev server on {url}...")
    
    success = False
    for i in range(20):
        try:
            with urllib.request.urlopen(url, timeout=1.0) as resp:
                if resp.status == 200:
                    content = resp.read().decode("utf-8")
                    if '<div id="root">' in content or 'src="/src/main.tsx"' in content:
                        print(f"[PASS] Vite dev server returned HTTP 200 and valid index.html.")
                        success = True
                        break
        except Exception:
            time.sleep(0.5)

    # Terminate process tree
    subprocess.call(["taskkill", "/F", "/T", "/PID", str(proc.pid)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    if not success:
        print("[FAIL] Vite dev server failed to respond.")
        sys.exit(1)
    
    print("[PASS] Vite dev server startup verification completed.")
    sys.exit(0)

if __name__ == "__main__":
    main()

