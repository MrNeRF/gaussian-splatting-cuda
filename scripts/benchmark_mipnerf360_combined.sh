#!/bin/bash

SCENE_DIR="data"
RESULT_DIR="results/benchmark_combined"
SCENE_LIST="garden bicycle stump bonsai counter kitchen room" # treehill flowers

if [ -d "$RESULT_DIR" ]; then
    echo "Results directory '$RESULT_DIR' already exists."
    read -p "Do you want to delete it and start fresh? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing existing results directory..."
        rm -rf "$RESULT_DIR"
    else
        echo "Keeping existing results. New results will overwrite existing ones for each scene."
    fi
    echo
fi

for SCENE in $SCENE_LIST; do
    if [ "$SCENE" = "bonsai" ] || [ "$SCENE" = "counter" ] || [ "$SCENE" = "kitchen" ] || [ "$SCENE" = "room" ]; then
        DATA_FACTOR=2
    else
        DATA_FACTOR=4
    fi

    echo "========================================="
    echo "Running $SCENE with images_${DATA_FACTOR}"
    echo "========================================="

    ./build/LichtFeld-Studio \
        -d $SCENE_DIR/$SCENE/ \
        -o $RESULT_DIR/$SCENE/ \
        --images images_${DATA_FACTOR} \
        --test-every 8 \
        --eval \
        --headless \
        --save-eval-images \
        --num-workers 8 \
        --strategy combined

    echo "Completed $SCENE"
    echo
done

format_number() {
    local num=$1
    local decimals=$2
    printf "%.$decimalsf" $num
}

format_with_commas() {
    local num=$1
    echo $num | sed ':a;s/\B[0-9]\{3\}\>/,&/;ta'
}

echo
echo "=============================================================================="
echo "QUALITY METRICS SUMMARY (COMBINED)"
echo "=============================================================================="
printf "%-10s %-10s %-10s %-10s %-10s %-15s\n" "scene" "iteration" "psnr" "ssim" "lpips" "num_gaussians"
echo "------------------------------------------------------------------------------"

total_psnr=0
total_ssim=0
total_lpips=0
total_gaussians=0
valid_scenes=0

for SCENE in $SCENE_LIST; do
    csv_file="$RESULT_DIR/$SCENE/metrics.csv"
    if [ -f "$csv_file" ]; then
        final_metrics=$(tail -n 1 "$csv_file")
        IFS=',' read -r iteration psnr ssim lpips time_per_image num_gaussians <<< "$final_metrics"
        psnr_fmt=$(format_number $psnr 4)
        ssim_fmt=$(format_number $ssim 6)
        lpips_fmt=$(format_number $lpips 6)
        gaussians_fmt=$(format_with_commas $num_gaussians)
        printf "%-10s %-10s %-10s %-10s %-10s %-15s\n" "$SCENE" "$iteration" "$psnr_fmt" "$ssim_fmt" "$lpips_fmt" "$gaussians_fmt"
        echo "------------------------------------------------------------------------------"
        total_psnr=$(echo "$total_psnr + $psnr" | bc -l)
        total_ssim=$(echo "$total_ssim + $ssim" | bc -l)
        total_lpips=$(echo "$total_lpips + $lpips" | bc -l)
        total_gaussians=$((total_gaussians + num_gaussians))
        valid_scenes=$((valid_scenes + 1))
    fi
done

if [ $valid_scenes -gt 0 ]; then
    mean_psnr=$(echo "$total_psnr / $valid_scenes" | bc -l)
    mean_ssim=$(echo "$total_ssim / $valid_scenes" | bc -l)
    mean_lpips=$(echo "$total_lpips / $valid_scenes" | bc -l)
    mean_gaussians=$((total_gaussians / valid_scenes))
    mean_psnr_fmt=$(format_number $mean_psnr 4)
    mean_ssim_fmt=$(format_number $mean_ssim 6)
    mean_lpips_fmt=$(format_number $mean_lpips 6)
    mean_gaussians_fmt=$(format_with_commas $mean_gaussians)
    echo "=============================================================================="
    printf "%-10s %-10s %-10s %-10s %-10s %-15s\n" "mean" "30000" "$mean_psnr_fmt" "$mean_ssim_fmt" "$mean_lpips_fmt" "$mean_gaussians_fmt"
fi

echo "=============================================================================="
echo
echo


