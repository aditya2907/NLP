import pandas as pd

# Sample data
data = {
    "comment": [
        "System crash occurred during test.",
        "Everything is running smoothly.",
        "Error 404 on page load.",
        "Performance seems fine.",
        "Memory usage spike detected.",
        "All systems operational.",
        "Unexpected shutdown occurred.",
        "Routine maintenance completed successfully."
    ],
    "date": [
        "2024-11-01",
        "2024-11-02",
        "2024-11-03",
        "2024-11-04",
        "2024-11-05",
        "2024-11-06",
        "2024-11-07",
        "2024-11-08"
    ],
    "failure_flag": [1, 0, 1, 0, 1, 0, 1, 0]
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Save as CSV
csv_file = "comments.csv"
df.to_csv(csv_file, index=False)
print(f"Sample dataset saved to {csv_file}")
