#!/usr/bin/env bash
Parsynt $1 2>&1 | sed 's/\x1B\[[0-9;]\{1,\}[A-Za-z]//g' > $2
