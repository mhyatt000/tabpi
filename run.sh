#!/bin/bash

echo "Running LiberoObj Tests..."

for ((i=0; i<50; i+=5)); do
	uv run scripts/run.py --fit 1 --env.task
done

echo "Tests completed"
