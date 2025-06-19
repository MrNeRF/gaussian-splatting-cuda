mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

if [ $? -ne 0 ]; then
    echo "Build failed. Please check the output for errors."
    exit 1
fi

cd ..

# ./build/gaussian_splatting_cuda -d ./data/tandt/train -o ./output -i 10000 -v

./build/gaussian_splatting_cuda -d ./data/mipnerf360/kitchen -o ./output -i 10000 -v
