#bin/bash


./py/generate_all.sh

echo "Generated all necessary files."

echo "Starting dense reconstruction..."
./scripts/bounty_002_disabled_densification.sh | tee final_run.txt # Log the output to a file to check for results afterwards
echo "Dense reconstruction completed."