#!/bin/bash

apt update && apt install time

set -euo pipefail

RUNS=5
OUTFILE="benchmark_results_gpu.json"
export OMP_NUM_THREADS=4

# Each entry: "executable arguments"
apps=(
    "./bin/sycl.x 5.0 1000000 add"
    "./bin/cuda.x 5.0 1000000 add"
)

# --- helpers ---

json_escape() {
  # Escapes a string for JSON (minimal but correct for typical stdout fields)
  local s="$1"
  s=${s//\\/\\\\}
  s=${s//\"/\\\"}
  s=${s//$'\n'/\\n}
  s=${s//$'\r'/\\r}
  s=${s//$'\t'/\\t}
  printf '%s' "$s"
}

is_number() {
  [[ "$1" =~ ^[+-]?([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][+-]?[0-9]+)?$ ]]
}

to_json_value() {
  # Print a JSON value (number/bool/null/string) from a raw token
  local v="$1"
  local vl="${v,,}"

  if [[ -z "$v" ]]; then
    printf '""'
  elif [[ "$vl" == "true" || "$vl" == "false" ]]; then
    printf '%s' "$vl"
  elif [[ "$vl" == "null" ]]; then
    printf 'null'
  elif is_number "$v"; then
    printf '%s' "$v"
  else
    printf '"%s"' "$(json_escape "$v")"
  fi
}

# Write opening JSON
printf '{\n  "meta": {\n    "runs_per_app": %d\n  },\n  "results": [\n' "$RUNS" > "$OUTFILE"

first_obj=1

for entry in "${apps[@]}"; do
  # Split entry into parts array
  read -r -a parts <<< "$entry"

  exe="${parts[0]}"
  wall_time="${parts[1]}"
  array_size="${parts[2]}"
  operation="${parts[3]}"
  app_name="$(basename "$exe")"

  echo "Benchmarking $app_name (wall_time=$wall_time, array_size=$array_size, op=$operation)"

  for run in $(seq 1 $RUNS); do
    # Capture GNU time metrics from stderr, program stdout separately.
    # NOTE: order of redirections matters.
    # We also prefix time line to parse reliably.
    time_prefix="__TIME__:"
    out="$(
      (/usr/bin/time -f "${time_prefix}%e,%M" "${parts[@]}" ) 2> >(cat >&2)   # keep stderr visible if you want
    )" || true

    # We need stderr too, but bash can't easily capture both stdout+stderr separately without temp files.
    # Use temp file for stderr.
    tmp_err="$(mktemp)"
    prog_out="$(
      /usr/bin/time -f "${time_prefix}%e,%M" "${parts[@]}" \
        2> "$tmp_err"
    )"
    time_line="$(grep -m1 "^${time_prefix}" "$tmp_err" || true)"
    rc=$?
    rm -f "$tmp_err"

    # Parse elapsed,maxrss from time_line
    tm="${time_line#${time_prefix}}"
    elapsed_s="${tm%%,*}"
    maxrss_kb="${tm#*,}"

    # Find last two non-empty lines of program stdout
    header_line="$(printf '%s\n' "$prog_out" | awk 'NF{a[NR]=$0} END{print a[NR-1]}' )"
    data_line="$(printf '%s\n' "$prog_out"   | awk 'NF{a[NR]=$0} END{print a[NR]}' )"

    # Split CSV header/data by commas (assumes no quoted commas in fields)
    IFS=',' read -r -a keys <<< "$header_line"
    IFS=',' read -r -a vals <<< "$data_line"

    # Build program_metrics JSON object
    program_metrics="null"
    if [[ "${#keys[@]}" -gt 0 && "${#keys[@]}" -eq "${#vals[@]}" ]]; then
      pm="{"
      for i in "${!keys[@]}"; do
        k="$(printf '%s' "${keys[$i]}" | xargs)"  # trim
        v="$(printf '%s' "${vals[$i]}" | xargs)"
        pm+="\"$(json_escape "$k")\":$(to_json_value "$v")"
        if [[ "$i" -lt $((${#keys[@]} - 1)) ]]; then
          pm+=","
        fi
      done
      pm+="}"
      program_metrics="$pm"
    fi

    # Comma between objects
    if [[ $first_obj -eq 1 ]]; then
      first_obj=0
    else
      printf ',\n' >> "$OUTFILE"
    fi

    # Write JSON object for this run
    printf '    {\n' >> "$OUTFILE"
    printf '      "app": "%s",\n' "$(json_escape "$app_name")" >> "$OUTFILE"
    printf '      "run": %d,\n' "$run" >> "$OUTFILE"
    printf '      "parameters": {\n' >> "$OUTFILE"
    printf '        "wall_time": %s,\n' "$(to_json_value "$wall_time")" >> "$OUTFILE"
    printf '        "array_size": %s,\n' "$(to_json_value "$array_size")" >> "$OUTFILE"
    printf '        "operation": "%s"\n' "$(json_escape "$operation")" >> "$OUTFILE"
    printf '      },\n' >> "$OUTFILE"
    printf '      "system": {\n' >> "$OUTFILE"
    printf '        "elapsed_seconds": %s,\n' "$(to_json_value "$elapsed_s")" >> "$OUTFILE"
    printf '        "max_rss_kb": %s\n' "$(to_json_value "$maxrss_kb")" >> "$OUTFILE"
    printf '      },\n' >> "$OUTFILE"
    printf '      "program_metrics": %s\n' "$program_metrics" >> "$OUTFILE"
    printf '    }' >> "$OUTFILE"
  done
done

# Close JSON
printf '\n  ]\n}\n' >> "$OUTFILE"

echo "Wrote $OUTFILE"

