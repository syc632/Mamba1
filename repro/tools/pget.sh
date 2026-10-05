#!/usr/bin/env bash
# Parallel byte-range downloader. The HF mirror throttles single connections
# down to ~140 KB/s but aggregates fine across many (~200 KB/s each).
#
# usage: bash tools/pget.sh <url> <outfile> <total_bytes> <n_workers> [prefix_part]
set -u
URL="$1"
OUT="$2"
TOTAL="$3"
N="${4:-10}"
PREFIX_PART="${5:-}"

# If TOTAL is "auto", ask the mirror API for the real size.
# Hand-passing a size is how we ended up with two truncated parquet shards:
# every shard in sample/10BT has a DIFFERENT byte length.
auto_size() {
  # Prefer HTTP HEAD: it returns the exact object length and does not depend on
  # the (rate-limited) tree API. Every shard in sample/10BT has a different
  # length, so this must never be guessed.
  local n
  n=$(curl -sIL -m 60 "$1" | tr -d '\r' \
      | awk 'tolower($1)=="content-length:"{v=$2} END{print v}')
  if [ -n "$n" ] && [ "$n" -gt 0 ] 2>/dev/null; then echo "$n"; return 0; fi
  # fallback: tree API
  local repo fpath fname fdir
  repo=$(echo "$1" | sed -E 's#https?://[^/]+/datasets/([^/]+)/resolve/[^/]+/.*#\1#')
  fpath=$(echo "$1" | sed -E 's#https?://[^/]+/datasets/[^/]+/resolve/[^/]+/(.*)#\1#')
  fname=$(basename "$fpath"); fdir=$(dirname "$fpath")
  curl -s -m 45 "https://hf-mirror.com/api/datasets/${repo}/tree/main/${fdir}" \
    | "${PY:-python}" -c "
import sys, json, os
name = '${fname}'
try:
    d = json.loads(sys.stdin.read())
except Exception:
    sys.exit(0)
if not isinstance(d, list):
    sys.exit(0)
for e in d:
    if isinstance(e, dict) and os.path.basename(str(e.get('path',''))) == name:
        print(e.get('size','')); break
"
}

if [ "$TOTAL" = "auto" ] || [ -z "$TOTAL" ]; then
  TOTAL=$(auto_size "$URL")
  if [ -z "$TOTAL" ] || [ "$TOTAL" -eq 0 ] 2>/dev/null; then
    echo "[pget] could not auto-detect size for $URL"; exit 1
  fi
  echo "[pget] auto-detected size: $TOTAL"
fi

DIR="$(dirname "$OUT")"
mkdir -p "$DIR/parts"

# part_00 = the bytes we already have (from an interrupted single-stream curl)
if [ -n "$PREFIX_PART" ] && [ -f "$PREFIX_PART" ]; then
  [ -f "$DIR/parts/part_00" ] || cp -n "$PREFIX_PART" "$DIR/parts/part_00" || true
fi
HAVE=0
if [ -f "$DIR/parts/part_00" ]; then HAVE=$(stat -c%s "$DIR/parts/part_00"); fi
echo "[pget] total=$TOTAL already_have=$HAVE workers=$N"

REM=$((TOTAL - HAVE))
CH=$(( (REM + N - 1) / N ))

pids=()
for i in $(seq 1 "$N"); do
  S=$(( HAVE + (i - 1) * CH ))
  E=$(( S + CH - 1 ))
  [ "$S" -ge "$TOTAL" ] && break
  [ "$E" -ge "$TOTAL" ] && E=$((TOTAL - 1))
  F=$(printf "%s/parts/part_%02d" "$DIR" "$i")
  SZ=$((E - S + 1))
  if [ -f "$F" ] && [ "$(stat -c%s "$F")" -eq "$SZ" ]; then
    echo "[pget] part_$(printf %02d $i) already complete"
    continue
  fi
  (
    for try in 1 2 3 4 5 6 7 8; do
      cur=0; [ -f "$F" ] && cur=$(stat -c%s "$F")
      if [ "$cur" -eq "$SZ" ]; then break; fi
      curl -sL --retry 3 --retry-all-errors -m 1800 \
           -r $((S + cur))-$E -o "$F.tmp" "$URL" && \
        cat "$F.tmp" >> "$F" && rm -f "$F.tmp"
      cur=0; [ -f "$F" ] && cur=$(stat -c%s "$F")
      [ "$cur" -eq "$SZ" ] && break
      echo "[pget] part_$(printf %02d $i) retry $try ($cur/$SZ)"
    done
  ) &
  pids+=($!)
done

wait
echo "[pget] workers done; assembling"

# assemble in order
rm -f "$OUT.tmp"
cat "$DIR/parts/part_00" > "$OUT.tmp"
for f in $(ls "$DIR"/parts/part_* | grep -v part_00 | sort); do
  cat "$f" >> "$OUT.tmp"
done
mv "$OUT.tmp" "$OUT"
echo "[pget] final size: $(stat -c%s "$OUT") / $TOTAL"
