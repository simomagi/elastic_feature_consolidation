import os
import pandas as pd
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregate summary.csv files from multiple experiment folders.")
    parser.add_argument("--base_path", type=str, required=True, help="Base path containing experiment folders.")
    args = parser.parse_args()

    BASE_PATH = args.base_path
    print(f"Aggregating summaries from: {BASE_PATH}")

    exp_folders = os.listdir(BASE_PATH)
    all_df = []

    for exp_folder in exp_folders:
        folder_path = os.path.join(BASE_PATH, exp_folder)

        if os.path.isdir(folder_path):
            summary_path = os.path.join(folder_path, "summary.csv")

            if os.path.exists(summary_path):
                print(f"Reading summary from: {summary_path}")
                df = pd.read_csv(summary_path) 
                all_df.append(df)
            else:
                print(f"No summary.csv found in {exp_folder}")

    if all_df:
        final_df = pd.concat(all_df, ignore_index=True)
        output_path = os.path.join(BASE_PATH, "all_summary.csv")
        final_df.to_csv(output_path, index=False)
        print(f"Saved all_summary.csv to {output_path}.")
    else:
        print("No summaries were found.")
