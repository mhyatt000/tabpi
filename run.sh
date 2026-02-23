#!/bin/bash

echo "Running LiberoObj Tests..."

for i in $(seq 0.2 0.2 .8); do
	uv run experiments/demo_shuffle.py --task_suite "libero_object" --training $i
done

echo "Tests completed"
