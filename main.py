import subprocess
import sys

def main():
    scripts = [
        "most_common_categories_network.py",
        "embedding_and_k_means.py",
        "id_category_data.py",
        "final_results.py",
        "graphs.py"
    ]
    
    for script in scripts:
        print(f"===== Running {script} =====")
        try:
            subprocess.run(["python", script], check=True)
            print(f"===== Finished {script} =====\n")
        except subprocess.CalledProcessError as e:
            print(f"Error running {script}")
            print(f"Exit code: {e.returncode}")
            sys.exit(1)
        except FileNotFoundError:
            print(f"Script not found: {script}")
            sys.exit(1)

if __name__ == "__main__":
    main()
