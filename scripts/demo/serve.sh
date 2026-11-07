#!/usr/bin/env bash
# Start or stop the isolated demo server: scripts/demo/serve.sh start|stop|start-cmd
set -eu
source "$(dirname "$0")/env.sh"
pidfile=$DEMO/serve.pid
stop() {
  [ -f "$pidfile" ] || return 0
  pid=$(cat "$pidfile"); kill "$pid" 2>/dev/null || true
  for _ in $(seq 90); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
  rm -f "$pidfile"
}
case "${1:-start}" in
  stop) stop ;;
  start|start-cmd)
    stop
    cmd=serve; [ "$1" = start-cmd ] && cmd=start
    COLUMNS=100 setsid arc-llama $cmd > "$DEMO/$cmd.log" 2>&1 < /dev/null &
    echo $! > "$pidfile"
    for _ in $(seq 60); do curl -s -o /dev/null 127.0.0.1:11437/health && exit 0; sleep 1; done
    echo "demo server did not come up" >&2; exit 1 ;;
esac
