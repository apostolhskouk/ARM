#!/bin/bash

# --- Configuration ---
# The Hugging Face Hub repository ID (username/repo_name)
REPO_ID="ApostolosK/arm_reproduction_data_processed"

# List of all local files and folders to upload.
# The script will preserve these paths in the remote repository.
PATHS_TO_UPLOAD=(
    "assets/feverous/benchmark.json"
    "assets/feverous/serialized_output/serialized_cell_level.jsonl"
    "assets/feverous/serialized_output/serialized_row_level.jsonl"
    "assets/feverous/serialized_output/serialized_table_level.jsonl"
    "assets/all_data/benchmarks"
    "assets/all_data/benchmarks_subsampled"
    "assets/all_data/serialized_data"
    "assets/all_data/indexes/dense_bge_m3"
    "assets/all_data/indexes/bm25"
    "assets/feverous/faiss_indexes"
    "assets/feverous/pyserini_indexes"
    "assets/arm"
)
# --- End of Configuration ---


# --- Step 1: Calculate and Print Total Size ---
echo "Calculating total size of files to be uploaded..."

# Check if all paths exist before calculating size
for path in "${PATHS_TO_UPLOAD[@]}"; do
    if [ ! -e "$path" ]; then
        echo "Error: Path not found: $path"
        echo "Please make sure you are running this script from your repository's root directory."
        exit 1
    fi
done

# Use 'du' to get the total size in bytes
# -s: display only a total for each argument
# -c: produce a grand total
# -b: print sizes in bytes
TOTAL_BYTES=$(du -scb "${PATHS_TO_UPLOAD[@]}" | tail -n 1 | awk '{print $1}')

# Use 'numfmt' to convert bytes to a human-readable format (e.g., KB, MB, GB)
# If 'numfmt' is not available, we will just show bytes.
if command -v numfmt &> /dev/null; then
    HUMAN_READABLE_SIZE=$(numfmt --to=iec-i --suffix=B --format="%.2f" "$TOTAL_BYTES")
    echo "--------------------------------------------------"
    echo "Total upload size: $HUMAN_READABLE_SIZE ($TOTAL_BYTES bytes)"
    echo "--------------------------------------------------"
else
    echo "--------------------------------------------------"
    echo "Total upload size: $TOTAL_BYTES bytes"
    echo "('numfmt' command not found, showing size in bytes)"
    echo "--------------------------------------------------"
fi
echo "" # Newline for readability


# --- Step 2: Upload Files and Folders ---
echo "Starting upload to Hugging Face Hub repository: $REPO_ID"
echo "This may take a while depending on your connection speed and the data size."
echo ""

for path in "${PATHS_TO_UPLOAD[@]}"; do
    echo "--> Uploading: $path"
    
    # The core command:
    # huggingface-cli upload [repo_id] [local_path] [path_in_repo]
    # We use the same path for local and in-repo to preserve the structure.
    # --repo-type=dataset is important to ensure it goes to the right section on the Hub.
    huggingface-cli upload "$REPO_ID" "$path" "$path" --repo-type=dataset
    
    # Check if the upload was successful
    if [ $? -ne 0 ]; then
        echo "Error: Failed to upload '$path'. Aborting script."
        exit 1
    fi
    echo "--> Successfully uploaded: $path"
    echo ""
done

echo "=========================================="
echo "          All uploads complete!           "
echo "=========================================="
echo "You can view your files at: https://huggingface.co/datasets/$REPO_ID/tree/main"