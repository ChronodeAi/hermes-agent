#!/usr/bin/env bash
# gateway-stall-watchdog.sh — auto-capture GIL-holder frames on stall storms.
#
# Watches the gateway errors.log for new "event loop stalled" warnings and,
# on the first sighting in each throttle window, sends SIGUSR1 to the
# dashboard backend, which appends an all-thread Python traceback to
# logs/gateway-stacks.log (see _install_stack_dump_diagnostics in
# hermes_cli/web_server.py). Turns the next production stall storm into
# named-frame evidence automatically — no root, no operator presence.
#
# Install (launchd or nohup):
#   nohup ~/.hermes/bin/gateway-stall-watchdog.sh >/dev/null 2>&1 &
# Stop: pkill -f gateway-stall-watchdog.sh

set -u

LOG="${HERMES_ERRORS_LOG:-$HOME/.hermes/logs/errors.log}"
STACKS_MARKER="event loop stalled"
STATE_FILE="${TMPDIR:-/tmp}/gateway-stall-watchdog.offset"
THROTTLE_S="${STALL_WATCHDOG_THROTTLE_S:-60}"

offset=0
last_fire=0

# Resume from end of file on start; only future stalls trigger dumps.
[ -f "$LOG" ] && offset=$(stat -f%z "$LOG" 2>/dev/null || echo 0)

while true; do
    sleep 5
    size=$(stat -f%z "$LOG" 2>/dev/null) || continue
    if [ "$size" -lt "$offset" ]; then
        offset=0  # log rotated/truncated
    fi
    new_bytes=$((size - offset))
    [ "$new_bytes" -le 0 ] && continue
    tail -c "$new_bytes" "$LOG" 2>/dev/null | grep -q "event loop stalled" || {
        offset=$size
        continue
    }
    offset=$size
    now=$(date +%s)
    if [ $((now - last_fire)) -lt "$THROTTLE_S" ]; then
        continue
    fi
    pid=$(lsof -nP -iTCP:9119 -sTCP:LISTEN 2>/dev/null | awk 'NR==2{print $2}')
    if [ -n "${pid:-}" ] && kill -0 "$pid" 2>/dev/null; then
        # Positive identity check (review-2 #10): a recycled PID must never
        # receive SIGUSR1 — default disposition terminates an unrelated
        # process. Only fire when the PID is actually the dashboard backend.
        CMD=$(ps -p "$pid" -o command= 2>/dev/null)
        case "$CMD" in
            *web_server*|*cmd_dashboard*|*hermes*dashboard*) ;;
            *)
                echo "$(date '+%Y-%m-%d %H:%M:%S') refusing SIGUSR1: pid $pid is not a hermes dashboard backend ($(echo "$CMD" | cut -c1-80))" \
                    >> "$HOME/.hermes/logs/gateway-stacks.log"
                last_fire=$now
                continue
                ;;
        esac
        # Rotate the stacks log when it grows past ~25MB (the C-level
        # faulthandler writes directly to it, so rotation happens here,
        # outside delivery). Post-rotation the gateway fd still writes to
        # the rotated .1 until the gateway restarts — accepted, documented.
        STACKS="$HOME/.hermes/logs/gateway-stacks.log"
        if [ -f "$STACKS" ] && [ "$(stat -f%z "$STACKS" 2>/dev/null || echo 0)" \
            -gt "${STACKS_ROTATE_BYTES:-26214400}" ]; then
            mv "$STACKS" "$STACKS.1" 2>/dev/null
            chmod 600 "$STACKS.1" 2>/dev/null
            touch "$STACKS" && chmod 600 "$STACKS"
        fi
        kill -USR1 "$pid" 2>/dev/null && \
            echo "$(date '+%Y-%m-%d %H:%M:%S') stall detected -> SIGUSR1 -> pid $pid" \
            >> "$HOME/.hermes/logs/gateway-stacks.log"
    fi
    last_fire=$now
done
