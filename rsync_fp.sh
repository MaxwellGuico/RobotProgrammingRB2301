#!/usr/bin/env bash
set -euo pipefail

robot="${1:-}"

if [[ ! "$robot" =~ ^[0-9]{1,2}$ ]]; then
    echo "Usage: $0 <robot number, 0–99>"
    echo "Example: $0 3"
    exit 1
fi

printf -v nn '%02d' "$((10#$robot))"
target="bingda@192.168.1.2${nn}"

rsync -auvx --delete \
    /home/maxwell-guico/rb2301/src/rb2301_fp \
    "${target}:~/Downloads/rb2301_fp_max/src"

rsync -auvx --delete \
    /home/maxwell-guico/rb2301/fp.sh \
    "${target}:~/Downloads/rb2301_fp_max"