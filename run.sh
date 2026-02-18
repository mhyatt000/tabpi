#!/bin/bash

echo "Running tests..."

for i in $(seq 0.2 0.2 .8); do
	pixi run libero --task_suite "libero_object" --training $i
done

echo "Libero Object done, moving to Spatial..."

for i in $(seq 0.2 0.2 .8); do
        pixi run libero --task_suite "libero_spatial" --training $i
done

echo "Libero Spatial done, moving to Goal..."

for i in $(seq 0.2 0.2 .8); do
        pixi run libero --task_suite "libero_goal" --training $i
done

echo "Tests completed"
