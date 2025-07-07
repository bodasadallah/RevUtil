import os
import pandas as pd
import subprocess
from tqdm import tqdm

# Path to the CSV
csv_path = "/home/abdelrahman.sadallah/mbzuai/review_rewrite/data/context_experiment.csv"

# Path to ScienceParse CLI JAR
science_parse_jar = "../Reviewer2/science-parse-cli-assembly-2.0.3.jar"

# Load the CSV
df = pd.read_csv(csv_path)

# Filter out rows without a valid paper_path or missing file
df = df[df['paper_path'].notnull() & df['paper_path'].str.strip().ne("")]

for paper_path in tqdm(df['paper_path']):
    paper_path = paper_path.strip()
    if not os.path.isfile(paper_path):
        continue

    # Build JSON path
    json_path = os.path.splitext(paper_path)[0] + ".json"
    
    # Skip if already parsed
    if os.path.isfile(json_path):
        continue

    # Run the ScienceParse command
    cmd = [
        "java", "-Xmx6g", "-jar",
        science_parse_jar,
        paper_path,
        "-o", os.path.dirname(json_path)
    ]

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"[!] Failed to parse {paper_path}: {e}")
