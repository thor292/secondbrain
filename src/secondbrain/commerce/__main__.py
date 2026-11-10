"""Run Live Commerce: the simulator and the engine, each in its own process.

    uv run python -m secondbrain.commerce                       # waits for Start on the wall
    uv run python -m secondbrain.commerce --autostart --speed 1day
    uv run python -m secondbrain.commerce --merchants 300 --customers 60000
    uv run python -m secondbrain.commerce --tsdb                # also TimescaleDB (commerce/tsdb.py)

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
    parser.add_argument("--tsdb", action="store_true",
                        help="also write every order to TimescaleDB and publish its SQL signals (commerce/tsdb.py)")
    args = parser.parse_args()

    py = [sys.executable, "-m"]
    # The engine first, so it is subscribed before the simulator sends the history backfill.
    engine = subprocess.Popen(py + ["secondbrain.commerce.engine", "--autopilot", str(args.autopilot)])
    time.sleep(3)
    if engine.poll() is not None:
        sys.exit("The engine failed to start; see the error above.")
    procs = [engine]
    if args.tsdb:   # before the simulator too, so the history backfill lands in TimescaleDB
        tsdb = subprocess.Popen(py + ["secondbrain.commerce.tsdb"])
        time.sleep(2)
        if tsdb.poll() is not None:
            engine.terminate()
            sys.exit("The TimescaleDB writer failed to start; see the error above.")
        procs.append(tsdb)
    sim = subprocess.Popen(py + ["secondbrain.commerce.simulator", "--merchants", str(args.merchants),
                                 "--customers", str(args.customers), "--seed", str(args.seed), "--speed", args.speed]
                           + (["--autostart"] if args.autostart else []))
    print(f"Live Commerce running: engine API on :{contract.ENGINE_PORT}"
          + (", TimescaleDB writer and watcher" if args.tsdb else "") + "; the wall is the app's /commerce page", flush=True)

    procs.insert(0, sim)

    def stop(*_):
        for p in procs:
            if p.poll() is None:
                p.terminate()

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    while all(p.poll() is None for p in procs):
        time.sleep(0.5)
    stop()
    for p in procs:
        try:
            p.wait(timeout=5)
        except subprocess.TimeoutExpired:  # never leave a stray engine holding the API port
            p.kill()


if __name__ == "__main__":
    main()
