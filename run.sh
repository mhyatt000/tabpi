#!/bin/bash

echo "Running LiberoObj Tests..."

for i in $(seq 0.2 0.2 .8); do
	uv run scripts/run.py --fit i
done

echo "Tests completed"
