#!/bin/bash
# Number of array tasks in a campaign file: the sum of the seed counts.
grep -v '^\s*#' "$1" | grep -v '^\s*$' | awk '{s += $2} END {print s}'
