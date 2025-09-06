# LichtFeld Studio - Windows Buiild Instructions

Currently, LichtFeld Studio does not offer any pre-compiled binaries as the project is geared towards devevelopment and with the current speed of development it does not make sense for binaries to be distributed to the casual user. 

But if you are not a developer but you are keen on trying LichtFeld Studio, then this step by step guide is intented  is for you!  Follow the steps below and you will be creating gaussian splats with LichtFeld Studio soon!

Note: Installation of dependencies and compiling of LichtFeld Studio will take about 1 hour

# TLDR

- Install Visual Studio 2022
- Install Cuda Toolkit 12.8 after installing Visual Studio
- Install Python 3.13.X
- Install GIT
- Follow the [instructions for "Windows"](https://github.com/MrNeRF/gaussian-splatting-cuda/?tab=readme-ov-file#build-instructions) 

# Long Version:

## Step 1: Installation Dependencies <a name="step1"></a>

### Visual Studio 2022 <a name="vs2020"></a>
- Download installer from https://visualstudio.microsoft.com/vs/community/
	- **NOT** Visual Studio Code
		- this does not contain the required files for building LichtFeld Studio
	- **NOT** Visual Studio 2019
		- This contains old cmake version
- Run setup.exe
- Install the following packages:
	- Dekstop Development with C++

 <img width="600" alt="image" src="https://github.com/user-attachments/assets/095ed93e-1cb0-44c6-82c3-14fda647efe7" />
  
- After installation is complete, exit Visual Studio if it was started automatically

### Cuda Toolkit 12.8 <a name="cuda128"></a>
- <u>Important</u>:
	- Dont start installation until Visual Studio has completed installation
	- if you have another version of Cuda Toolkit, uninstall it and re-install Cuda Toolkit 12.
- Download from https://developer.nvidia.com/cuda-12-8-0-download-archive
- Select windows as your operation system, select x86_64 as architecture and select your windows version and select "exe (local)".
- Download the 3.2GB file

<img width="600" alt="image" src="https://github.com/user-attachments/assets/7c11556f-93d7-4a8a-8fab-7222c30e8c9c" />

- After download, execute the file and unpack the installation files in the proposed directory
- Use "express" installation during installation

<img width="400" alt="image" src="https://github.com/user-attachments/assets/f8262aaa-db90-47fa-b3bf-830c64edbb88" />
  
- After installation is complete, verify if "nsight for Visual Studio 2022" was installed

<img width="400" alt="image" src="https://github.com/user-attachments/assets/6663bfe7-456d-4262-aab7-cf6c25f77e83" />

- Press "next" and close the installation

### Python 3 <a name="python"></a>
- Download the Winsodws installer for the latest stable release from https://www.python.org/downloads/windows/
- Run the installer
- Choose "Customize the installation"
- In the first options screen, select all options

<img width="400" alt="image" src="https://github.com/user-attachments/assets/f6350398-43dc-485a-b0ae-e4ac81580eed" />

- In "Advanced options", make sure the following are selected:
	- Add Python to environment variables
 	- Download debug binaries

 <img width="400" alt="image" src="https://github.com/user-attachments/assets/7cb1a205-85cd-4a9e-b504-b630dcada95a" />

- Press "install" to start the installation
- Close the installer when completed

### Git <a name="git"></a>
- Download windows installer from https://git-scm.com/downloads
- Follow instructions, use default settings for all options (there are many)
- Close installation after completion

## Step 2: Verifying your prerequisites <a name="step2"></a>

- Press start and find "developer command prompt for VS 2022"
- Type the following commands to verify your installation:

		cmake --version
  
		nvcc --version
  
		git --version
  
 		Python --version
 
- Cmake: Must be 3.24 or higher
- nvcc: Verify that 12.8 is being used
- git: Verify that git shows version information
- python: Verify that 3.10 or above is used

<img width="1108" height="622" alt="image" src="https://github.com/user-attachments/assets/9cdee296-47f4-4cde-8072-03829ecc6342" />


## Step 3 : Downloading and building LichtFeld Studio <a name="step3"></a>

- Press start and find "x64 native tools command prompt for VS 2022"
- Create a directory "repos"
  
		mkdir repos

- Go to the directoy "repos"

		cd repos

- Set up vcpkg (one-time setup)

		git clone https://github.com/microsoft/vcpkg.git
  		cd vcpkg && .\bootstrap-vcpkg.bat -disableMetrics && cd ..
  
- Clone repository
  
		git clone https://github.com/MrNeRF/LichtFeld-Studio
  
- Create directories
  
		cd LichtFeld-Studio
		if not exist external mkdir external
		if not exist external\debug mkdir external\debug
		if not exist external\release mkdir external\release

- Download LibTorch (Debug)

		curl -L -o libtorch-debug.zip https://download.pytorch.org/libtorch/cu128/libtorch-win-shared-with-deps-debug-2.7.0%2Bcu128.zip
		tar -xf libtorch-debug.zip -C external\debug
		del libtorch-debug.zip

- Download LibTorch (Release)
  
		curl -L -o libtorch-release.zip https://download.pytorch.org/libtorch/cu128/libtorch-win-shared-with-deps-2.7.0%2Bcu128.zip
		tar -xf libtorch-release.zip -C external\release
		del libtorch-release.zip

- Build configuration files and download dependancies
 
		cmake -B build -DCMAKE_BUILD_TYPE=Release -G Ninja -DCMAKE_TOOLCHAIN_FILE="../vcpkg/scripts/buildsystems/vcpkg.cmake"

- build LichtFeld Studio
 
		cmake --build build -j

After the last step is complete, you should have a new directory "\build" where you can find "LichtFeld-Studio.exe" and you can execute that file to run LichtFeld Studio.

<img width="1110" height="683" alt="image" src="https://github.com/user-attachments/assets/605c5b3a-53b3-4f16-85e2-05e9af2327cd" />

<img width="1276" height="744" alt="image" src="https://github.com/user-attachments/assets/0daee03d-8aa5-4168-a5d0-c317491e21ff" />


## Troubleshooting <a name="troubleshooting"></a>

### Before anything else:
- Make sure you run all commands from the "developer command prompt for VS 2022" (notstandard command or cmd or powershell)
- Verify if you have the proper requirements installed -see Step 2
- Uninstall all Cuda Toolkit versions and re-install 12.
- Delete the gaussian splatting directory, and restart the instructions from Step 3

### Common issues
#### Release build works, but debug build fails with error "cannot open file 'python313_d.lib'"
- this could be missing python debug libraries
- Run the python setup again, choose "modify" and select "download debug binaries"
- copy the files python313_d.lib from your python installation directory to the build\debug directory
  	  
#### Cannot open include file

<img width="600" alt="image" src="https://github.com/user-attachments/assets/60eaae64-3b85-42af-9276-31550b5d3d33" />

- Run the visual studio installation and modify the installation. Verify you have the C++ package installed (see step 1)

#### Building does not generate the .exe, but only the lib file

<img width="600" alt="image" src="https://github.com/user-attachments/assets/28b2ed73-d0b1-492a-aa1d-7762391e94d1" />

- Possible cause: build files not up-to date with latest changes
- Solution: Re-generate the configuration files using in the command prompt and rebuild LichtFeld Studio
        
			cmake -B build -DCMAKE_BUILD_TYPE=Release -G Ninja -DCMAKE_TOOLCHAIN_FILE="../vcpkg/scripts/buildsystems/vcpkg.cmake"
			cmake --build build -j

#### Other things to check
- Type "set" in the console
- Verify the following environment variables
	- CUDA_ROOT -> must point to your cuda toolkit installation
   	- INCLUDE -> must point to your Visual Studio installation
   	- PATH -> must contain path to all binaries of the installed tools (Python, Visual Studio, Nvida Toolkit, Git)

#### Manual installation of Cuda in Visual Studio
- set CUDA_ROOT environment variable manually
- copy the files from
	`C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\extras\visual_studio_integration\MSBuildExtensions`
   	to
   	`C:\Program Files\Microsoft Visual Studio\ 2022 \Community\MSBuild\Microsoft\VC\v170\BuildCustomizations`

