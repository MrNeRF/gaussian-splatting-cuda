# FAQ

### Where can I find a binary download or installer ?
LFS is not yet meant for a wide public release. There is no installer yet, because the software is not yet in a state that is stable enough for the support that a wide audience would require. Once development has progressed enough and we have a stabilization pass, there will be an installer.

### When I use LFS to train a scene, my scene "explodes". What could be the cause?
- Possible Causes:
	- You are using distorted images, where LFS expects undistorted images
	- You have done post processing on your images AFTER undistortion
	- Your alignment is having a too big of an error margin
-  Resolutions
    - Use undistorted images for training
	- Do not rescale/resample/sharpen/noise reduction/… your images
	- Check alignments of the cameras for errors
	- Adjust the parameter "position" from the default 0.00016 to 0.000016
		
### My system / training is slow, even less than 1 iteration / second.   What could be the cause?
- Possible Causes:
	-  Most likely you are hitting your VRAM limit.  
- Resolutions:
	- Decrease the number of gaussians for training
	- Use the parameter --num-workers to limit concurrent threads that use GPU for loading images in order to make more Vram available for gaussians
	- Train on lower resolution images.
	
### My CPU is heavily used and at 100% all the time, is this normal?
- Possible Causes
	- You used image rescaling factor
		- LFS is currently rescaling the image every time it loads it from disk.  This causes heavey CPU usage for the rescaling.   A caching mechanism (on memory or disk) would solve this, but this is currently not available
	- Your images are large
		- If an image has the largest size greater than 3840pixels, with default settings, LFS will rescale the image to 3840.  This scaling is happening every time the image is loaded from disk (see previous point)
- Resolutions
	- Export your undistorted image to the size that you want to train them
	- Export your images with a maximum of 3840pixels
	- Set the max-width to the width of  your images
		- Remark: max supoprted pixel size is currently 4096x4096 pixels
	
### How do I export the created splat
The gaussian splat is automatically created in the directory where you have saved your LFS project.  In case you did not save your project, you can do so after training.  If you did not save your project, the .ply is in a temporary directory
	
### What is the maximum number of images that LFS can train
LFS does not impose a maximum number of images. But more images require more training steps. You can play with the `steps-scaler` setting to change this ratio automatically if needed.  
Image count should not directly affect resource/VRAM usage, but image resolution will.
	
### Does LFS support multiple GPUs?
Not currently. PRs welcome.
