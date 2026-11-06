"""Run Live Commerce: the simulator and the engine, each in its own process.

    uv run python -m secondbrain.commerce                       # waits for Start on the wall
    uv run python -m secondbrain.commerce --autostart --speed 1day
    uv run python -m secondbrain.commerce --merchants 300 --customers 60000

Two processes, like production, where the platforms' feeds and the engine run
apart and only meet on the broker. Ctrl-C stops both.
"""

import argparse
import signal
import subprocess
import sys
import time

from secondbrain.commerce import contract


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--merchants", type=int, default=600)
    parser.add_argument("--customers", type=int, default=120000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--speed", choices=list(contract.SPEEDS), default=contract.DEFAULT_SPEED)
    parser.add_argument("--autostart", action="store_true", help="start the clock without waiting for the wall")
    parser.add_argument("--autopilot", type=float, default=0.5, help="share of merchants that apply offers automatically")
    args = parser.parse_args()

    py = [sys.executable, "-m"]
    # The engine first, so it is subscribed before the simulator sends the history backfill.
    engine = subprocess.Popen(py + ["secondbrain.commerce.engine", "--autopilot", str(args.autopilot)])
    time.sleep(3)
    if engine.poll() is not None:
        sys.exit("The engine failed to start; see the error above.")
    sim = subprocess.Popen(py + ["secondbrain.commerce.simulator", "--merchants", str(args.merchants),
                                 "--customers", str(args.customers), "--seed", str(args.seed), "--speed", args.speed]
                           + (["--autostart"] if args.autostart else []))
    print(f"Live Commerce running: engine API on :{contract.ENGINE_PORT}, wall at http://localhost:8765/commerce", flush=True)

    def stop(*_):
        for p in (sim, engine):
            if p.poll() is None:
                p.terminate()

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    while sim.poll() is None and engine.poll() is None:
        time.sleep(0.5)
    stop()
    sim.wait()
    engine.wait()


if __name__ == "__main__":
    main()
