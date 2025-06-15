import subprocess
import sys
import os

# Define the directory containing the scripts
scripts_dir = os.path.dirname(os.path.abspath(__file__))

# Define the scripts to run
scripts_to_run = [
    os.path.join(scripts_dir, 'B_pipeline.py'),
    os.path.join(scripts_dir, 'C_train_model.py'),
    os.path.join(scripts_dir, 'D_aggregate_predictions.py')
]

print("Starting the execution of the pipeline scripts...")

for script_path in scripts_to_run:
    script_name = os.path.basename(script_path)
    print(f"\nRunning script: {script_name}")
    try:
        # Use sys.executable to ensure the script is run with the same Python interpreter
        # that is running this orchestration script.
        # Use cwd to ensure the subprocess runs in the directory containing the scripts.
        result = subprocess.run([sys.executable, script_path], cwd=scripts_dir, check=True, capture_output=True, text=True)
        print(f"--- {script_name} Standard Output ---")
        print(result.stdout)
        print(f"--- End of {script_name} Standard Output ---")
        if result.stderr:
            print(f"--- {script_name} Standard Error ---")
            print(result.stderr)
            print(f"--- End of {script_name} Standard Error ---")
        print(f"Script {script_name} finished successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error running script {script_name}:")
        print(f"Command: {' '.join(e.cmd)}")
        print(f"Return Code: {e.returncode}")
        print(f"Standard Output:\n{e.stdout}")
        print(f"Standard Error:\n{e.stderr}")
        print("Stopping execution due to script failure.")
        sys.exit(1) # Exit with a non-zero code to indicate failure
    except FileNotFoundError:
        print(f"Error: Python interpreter '{sys.executable}' or script '{script_name}' not found.")
        print("Please ensure Python is installed and in your PATH, and the script file exists.")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred while running {script_name}: {e}")
        sys.exit(1)

print("All scripts finished successfully.") 