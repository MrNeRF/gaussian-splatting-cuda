# Timelapse

If you want to see how your reconstruction is progressing during training, you can enable timelapse generation. This will save out images of the current reconstruction at regular intervals. Currently, you can only save renders from a camera pose that corresponds to one of the training images.

## How to Use
To enable timelapse generation, use the `--timelapse-images` flag for every image you save timelapse images for. You can also use the `--timelapse-interval` flag to set how often (in number of training steps) to save out images. The default is every 50 steps.

:::note
The saved images can take up a significant amount of disk space, especially if you are saving images frequently, for many different images, or training on high resolution images (saved image size depends on --resize_factor too). Make sure you have enough disk space available.
:::

### Example
```bash
--timelapse-images IMG_6672.JPG --timelapse-images IMG_6690.JPG --timelapse-interval 100
```

This will save out renders from the camera poses corresponding to `IMG_6672.JPG` and `IMG_6690.JPG` every 100 training steps. The images will be saved in subfolders that correspond to the image names (with the file extension truncated) in the `timelapse` folder inside your output directory (see visual structure below).

```
<output_directory>
├── project.ls
└── timelapse
    ├── IMG_6672
    │   ├── 000100.png
    │   ├── 000200.png
    │   └── ...
    └── IMG_6690
        ├── 000100.png
        ├── 000200.png
        └── ...
```