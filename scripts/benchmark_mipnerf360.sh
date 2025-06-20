#!/bin/bash

SCENE_DIR="data"
RESULT_DIR="results/benchmark"
SCENE_LIST="garden bicycle stump bonsai counter kitchen room" # treehill flowers

for SCENE in $SCENE_LIST;
do
    # Determine data factor based on scene type
    if [ "$SCENE" = "bonsai" ] || [ "$SCENE" = "counter" ] || [ "$SCENE" = "kitchen" ] || [ "$SCENE" = "room" ]; then
        DATA_FACTOR=2
    else
        DATA_FACTOR=4
    fi

    echo "========================================="
    echo "Running $SCENE with images_${DATA_FACTOR}"
    echo "========================================="

    # Run training with evaluation
    ./build/gaussian_splatting_cuda \
        -d $SCENE_DIR/$SCENE/ \
        -o $RESULT_DIR/$SCENE/ \
        --images images_${DATA_FACTOR} \
        --test-every 8 \
        --eval \
        --save-eval-images

    echo "Completed $SCENE"
    echo
done

# Print results summary
echo "========================================="
echo "BENCHMARK RESULTS SUMMARY"
echo "========================================="

for SCENE in $SCENE_LIST;
do
    echo "=== $SCENE ==="
    
    # Check if metrics files exist
    if [ -f "$RESULT_DIR/$SCENE/metrics_report.txt" ]; then
        echo "Metrics Report:"
        cat "$RESULT_DIR/$SCENE/metrics_report.txt"
    else
        echo "No metrics report found"
    fi
    
    # Also show the CSV for easy parsing
    if [ -f "$RESULT_DIR/$SCENE/metrics.csv" ]; then
        echo -e "\nCSV Data:"
        cat "$RESULT_DIR/$SCENE/metrics.csv"
    fi
    
    echo "----------------------------------------"
    echo
done

# Optional: Create a summary CSV combining all scenes
echo "Creating combined results..."
OUTPUT_CSV="$RESULT_DIR/combined_results.csv"
echo "scene,iteration,psnr,ssim,lpips,time_per_image,num_gaussians" > $OUTPUT_CSV

for SCENE in $SCENE_LIST;
do
    if [ -f "$RESULT_DIR/$SCENE/metrics.csv" ]; then
        # Skip header and add scene name to each row
        tail -n +2 "$RESULT_DIR/$SCENE/metrics.csv" | while read line; do
            echo "$SCENE,$line" >> $OUTPUT_CSV
        done
    fi
done

echo "Combined results saved to: $OUTPUT_CSV"
