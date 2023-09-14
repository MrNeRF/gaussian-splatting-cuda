import subprocess
import os
import re
import argparse

def get_unique_folder_name(base_name):
    """
    This function generates a unique folder name by appending an incremental suffix.
    """
    counter = 1
    new_name = base_name

    while os.path.exists(new_name):
        new_name = f"{base_name}_{counter}"
        counter += 1

    return new_name

def run_command_and_rename(n_times, input_folder_path, iterations):
    cmd_base = ["./cmake-build-release/gaussian_splatting_cuda", "-r 4", "-f", "-d", input_folder_path, "-i", str(iterations)]
    output_folder_path = "output"  # Assuming the 'output' folder is generated in the current directory

    for i in range(n_times):
        print(f"Training on data folder: {input_folder_path}, Iteration: {i + 1}")

        # Run the command and capture the output
        result = subprocess.run(cmd_base, capture_output=True, text=True)

        # Check if the process was successful
        if result.returncode != 0:
            print(f"Command failed on iteration {i + 1} with return code: {result.returncode}")
            break

        # Ensure the output folder exists before renaming
        if not os.path.exists(output_folder_path):
            print(f"Output folder '{output_folder_path}' does not exist after iteration {i + 1}.")
            break

        # Extract number of splats from output using regex
        match = re.search(r"(\d+) splats", result.stdout)
        if match:
            num_splats = match.group(1)
            desired_name = f"{output_folder_path}_{num_splats}"

            # Get unique folder name
            new_folder_name = get_unique_folder_name(desired_name)

            os.rename(output_folder_path, new_folder_name)

            # Print the new folder name for reference
            print(f"Output folder renamed to: {new_folder_name}")
        else:
            print("Could not find the number of splats in the output.")

    # Print the final output line of the executable to stdout
    last_line = result.stdout.strip().split("\n")[-1]
    print(last_line)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the executable and rename the output folder.")
    parser.add_argument("n_times", type=int, help="Number of times to run the command.")
    parser.add_argument("input_folder_path", type=str, help="Path to the input folder.")
    parser.add_argument("iterations", type=int, help="Number of iterations for the executable.")

    args = parser.parse_args()
    run_command_and_rename(args.n_times, args.input_folder_path, args.iterations)

