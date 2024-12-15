import os
import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import openai
from tenacity import retry, stop_after_attempt, wait_fixed

# Retry mechanism for LLM failures
@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def call_openai(prompt, model="gpt-4o-mini"):
    response = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )
    return response["choices"][0]["message"]["content"]

def analyze_data(csv_file):
    # Read data
    df = pd.read_csv(csv_file)
    
    # Perform basic analysis
    summary = {
        "columns": df.columns.tolist(),
        "missing_values": df.isnull().sum().to_dict(),
        "data_types": df.dtypes.to_dict(),
        "example_rows": df.head(3).to_dict(orient="records"),
    }
    
    # Use LLM to generate advanced analysis
    insights = call_openai(f"Here is a dataset summary: {summary}. Suggest analyses.")
    print(insights)
    
    # Generate visualizations
    create_visualizations(df)
    
    # Generate a narrative report
    generate_readme(df, summary, insights)

def create_visualizations(df):
    # Example visualization
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
    plt.title("Correlation Matrix")
    plt.savefig("correlation_matrix.png")
    plt.close()

def generate_readme(df, summary, insights):
    prompt = f"""
    Write a detailed story based on the following:
    - Dataset description: {summary}
    - Insights: {insights}
    - Visualizations: Include 'correlation_matrix.png'.

    Format the output as a Markdown file.
    """
    story = call_openai(prompt)
    with open("README.md", "w") as f:
        f.write(story)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: uv run autolysis.py <dataset.csv>")
        sys.exit(1)
    analyze_data(sys.argv[1])
