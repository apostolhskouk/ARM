from src.utils.sample_queries import subsample_json
import os
if __name__ == "__main__":
    input_file = "assets/feverous/benchmark.json"
    output_file = "assets/feverous/benchmark_subsampled.json"

    subsample_json(input_file, "claim", output_file, k=1000)
    
    input_folder = "assets/all_data/benchmarks"
    output_folder = "assets/all_data/benchmarks_subsampled"
    #create output folder if it does not exist
    os.makedirs(output_folder, exist_ok=True)
    #iterate over all files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".json"):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            subsample_json(input_path, "query", output_path, k=1000)