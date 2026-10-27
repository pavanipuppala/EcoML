"""
EcoML — Live Server Endpoint Verification.

Starts the FastAPI server in-process, tests HTTP endpoints via urllib/requests,
and verifies GET /api/quantum/info and POST /api/quantum/optimize against the running HTTP server.
"""

import sys
import time
import json
import urllib.request
import urllib.error
from pathlib import Path
import threading
import uvicorn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.main import app


def run_server():
    config = uvicorn.Config(app, host="127.0.0.1", port=8008, log_level="warning")
    server = uvicorn.Server(config)
    server.run()


def test_live_endpoints():
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()

    base_url = "http://127.0.0.1:8008"
    print(f"Waiting for backend to bind to {base_url}...")

    # Wait for server ready
    ready = False
    for _ in range(30):
        try:
            with urllib.request.urlopen(f"{base_url}/api/health", timeout=1.0) as resp:
                if resp.status == 200:
                    ready = True
                    break
        except Exception:
            time.sleep(0.3)

    if not ready:
        print("[FAIL] Backend server failed to start within timeout.")
        sys.exit(1)

    print("[PASS] Backend server successfully running on 127.0.0.1:8008.")

    # 1. Test GET /api/quantum/info
    print("Testing GET /api/quantum/info...")
    req_info = urllib.request.Request(f"{base_url}/api/quantum/info")
    with urllib.request.urlopen(req_info, timeout=2.0) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode())
        print(f"  Response: simulator={data['simulator_type']}, max_qubits={data['max_qubits']}, depths={data['supported_depths']}")
        assert data["max_qubits"] >= 10
        assert 2 in data["supported_depths"]
    print("[PASS] GET /api/quantum/info succeeded via live HTTP.")

    # 2. Test POST /api/quantum/optimize
    print("Testing POST /api/quantum/optimize...")
    payload = {
        "depth_p": 1,
        "shots": 256,
        "classical_optimizer": "COBYLA",
        "penalty_factor": 2.0,
        "weights": {
            "w_performance": 0.40,
            "w_train_time": 0.15,
            "w_inference_latency": 0.15,
            "w_ram": 0.15,
            "w_carbon": 0.15,
        },
    }
    req_opt = urllib.request.Request(
        f"{base_url}/api/quantum/optimize",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req_opt, timeout=10.0) as resp:
        assert resp.status == 200
        res_data = json.loads(resp.read().decode())
        print(f"  Response: dataset={res_data['dataset_name']}")
        print(f"  Classical winner: {res_data['classical_baseline']['selected_model_name']}")
        print(f"  QAOA winner:      {res_data['qaoa_result']['best_feasible_model']['selected_model']}")
        print(f"  Agreement:        {res_data['comparison']['agreement']}")
        assert len(res_data["candidates"]) >= 3
        assert res_data["classical_baseline"]["is_feasible"] is True
        assert res_data["qaoa_result"]["best_feasible_model"]["is_feasible"] is True
    print("[PASS] POST /api/quantum/optimize succeeded via live HTTP.")

    print("\nALL LIVE SERVER ENDPOINT TESTS PASSED.")
    sys.exit(0)


if __name__ == "__main__":
    test_live_endpoints()

