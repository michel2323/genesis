#!/bin/bash
# Run probe.jl under several IGC_OverrideOCLMaxParamSize settings, each in a fresh process,
# with the persistent NEO compiler cache disabled and a distinct salt per run.
# Logs (log_*.txt) are written next to this script. Uses the repro/ environment.
cd "$(dirname "$0")"
for cfg in "1024 102" "4096 103" "8192 104" "4096 105"; do
  set -- $cfg
  log=log_override$1_salt$2.txt
  env IGC_OverrideOCLMaxParamSize=$1 NEO_CACHE_PERSISTENT=0 SALT=$2 julia +1.12 --project=.. probe.jl > $log 2>&1
  echo "== override $1 salt $2 exit $?"
  grep -E "^RESULT|!!|^# (setting|done)" $log | sed -E 's/:: ZeError.*//'
done
