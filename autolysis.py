import os
import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from os.path import splitext, basename, join
from dotenv import load_dotenv
from tabulate import tabulate
import requests
import json
import matplotlib
import chardet

load_dotenv()
matplotlib.use('Agg')

def detect_encoding(file_path):
    with open(file_path, 'rb') as f:
        raw_data = f.read()
        result = chardet.detect(raw_data)
        return result['encoding']

def analyze_csv(file_path):
    base_name = splitext(basename(file_path))[0]
    output_dir = join(os.getcwd(), base_name)
    os.makedirs(output_dir, exist_ok=True)

    # Load the dataset
    try:
        encoding = detect_encoding(file_path)
        data = pd.read_csv(file_path, encoding=encoding)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return

    # Summary statistics
    summary = data.describe(include='all').transpose()
    missing_values = data.isnull().sum()

    # Correlation matrix
    numeric_data = data.select_dtypes(include=['number'])
    correlation_table = numeric_data.corr() if not numeric_data.empty else None
    correlation_image = None
    if correlation_table is not None:
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_table, annot=True, cmap='coolwarm')
        plt.title('Correlation Matrix Heatmap')
        correlation_image = join(output_dir, 'correlation_matrix.png')
        plt.savefig(correlation_image)
        plt.close()

    # Clustering analysis
    clustering_insights = "Not performed"
    pca_image, tsne_image = None, None
    if numeric_data.shape[1] > 1:
        scaled_data = StandardScaler().fit_transform(numeric_data.dropna())
        pca = PCA(n_components=2)
        reduced_data = pca.fit_transform(scaled_data)

        kmeans = KMeans(n_clusters=3, random_state=42)
        clusters = kmeans.fit_predict(reduced_data)

        plt.figure(figsize=(8, 6))
        plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=clusters, cmap='viridis')
        plt.title('Clustering Visualization (PCA)')
        pca_image = join(output_dir, 'pca_clustering.png')
        plt.savefig(pca_image)
        plt.close()

        tsne = TSNE(n_components=2, random_state=42)
        tsne_data = tsne.fit_transform(scaled_data)

        plt.figure(figsize=(8, 6))
        plt.scatter(tsne_data[:, 0], tsne_data[:, 1], c=clusters, cmap='viridis')
        plt.title('Clustering Visualization (t-SNE)')
        tsne_image = join(output_dir, 'tsne_clustering.png')
        plt.savefig(tsne_image)
        plt.close()

        clustering_insights = "Clustering performed using PCA and t-SNE."

    # Outlier detection
    outlier_insights = "Not performed"
    outliers_image = None
    if numeric_data.shape[1] > 1:
        isolation_forest = IsolationForest(random_state=42, contamination=0.05)
        outliers = isolation_forest.fit_predict(scaled_data)

        plt.figure(figsize=(8, 6))
        plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=outliers, cmap='coolwarm', alpha=0.6)
        plt.title('Outlier Detection (Isolation Forest)')
        outliers_image = join(output_dir, 'outliers.png')
        plt.savefig(outliers_image)
        plt.close()

        outlier_insights = "Outlier detection performed using Isolation Forest."

    # Regression analysis (Linear Regression)

    regression_metrics, regression_image = "Not performed", None
    if numeric_data.shape[1] > 1:
        X = numeric_data.iloc[:, 0].values.reshape(-1, 1)
        y = numeric_data.iloc[:, 1].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        regressor = LinearRegression()
        regressor.fit(X_train, y_train)
        y_pred = regressor.predict(X_test)

        regression_metrics = {
            "Mean Squared Error": mean_squared_error(y_test, y_pred),
            "R2 Score": r2_score(y_test, y_pred),
        }

        plt.figure(figsize=(8, 6))
        plt.scatter(X_test, y_test, color='blue')
        plt.plot(X_test, y_pred, color='red')
        plt.title('Linear Regression')
        regression_image = join(output_dir, 'regression.png')
        plt.savefig(regression_image)
        plt.close()

    # Save analysis context

    context = {
        'columns': data.columns.tolist(),
        'dtypes': data.dtypes.astype(str).tolist(),
        'missing_values': missing_values.to_frame(name="Missing Values"),
        'summary': summary,
        'correlation_table': correlation_table,
        'correlation_image': 'correlation_matrix.png' if correlation_image else None,
        'clustering_insights': clustering_insights,
        'pca_image': 'pca_clustering.png' if pca_image else None,
        'tsne_image': 'tsne_clustering.png' if tsne_image else None,
        'outlier_insights': outlier_insights,
        'outliers_image': 'outliers.png' if outliers_image else None,
        'regression_metrics': regression_metrics,
        'regression_image': 'regression.png' if regression_image else None,
    }

    return context

def narrate_analysis(context, file_path):
    base_name = splitext(basename(file_path))[0]
    output_dir = join(os.getcwd(), base_name)
    os.makedirs(output_dir, exist_ok=True)
    readme_file = join(output_dir, "README.md")

    url = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
    api_key = os.getenv("AIPROXY_TOKEN")

    prompt = (
        f"Analyze the dataset '{file_path}'. Here is the context:\n"
        f"Columns and types: {context['columns']}, {context['dtypes']}\n"
        f"Missing values: {context['missing_values'].to_markdown(index=True)}\n"
        f"Summary statistics:\n{context['summary'].to_markdown(index=True)}\n"
        f"Correlation matrix insights:\n{context['correlation_table'].to_markdown(index=True) if context.get('correlation_table') is not None else 'Not provided'}\n"
        f"Clustering results: {context.get('clustering_insights', 'Not performed')}\n"
        f"Outlier detection results: {context.get('outlier_insights', 'Not performed')}\n"
        f"Regression analysis metrics: {context.get('regression_metrics', 'Not performed')}\n"
        "Generate insights and provide a structured Markdown report, including:\n"
        "1. Dataset Overview: Highlight important features and missing values.\n"
        "2. Correlation Analysis: Include tables and insights.\n"
        "3. Clustering Analysis: Interpret PCA and t-SNE visualizations.\n"
        "4. Outlier Detection: Summarize anomalies and visualizations.\n"
        "5. Regression Analysis: Summarize performance metrics and provide interpretations.\n"
        "6. Conclusion: Provide a final summary of findings"
    )

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    data = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }

    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        story = response.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"Error generating narrative: {e}")
        story = "Error generating narrative."

    readme_content = "# Analysis Report\n\n"
    readme_content += "## Dataset Overview\n\n"
    readme_content += "### Summary Statistics\n\n"
    readme_content += f"{context['summary'].to_markdown(index=True)}\n\n"
    readme_content += "### Missing Values\n\n"
    readme_content += f"{context['missing_values'].to_markdown(index=True)}\n\n"

    if context.get('correlation_table') is not None:
        readme_content += "## Correlation Analysis\n\n"
        readme_content += "### Correlation Matrix\n\n"
        readme_content += f"{context['correlation_table'].to_markdown(index=True)}\n\n"
        if context.get('correlation_image'):
            readme_content += f"![Correlation Heatmap](./{context['correlation_image']})\n\n"

    if context.get('clustering_insights'):
        readme_content += "## Clustering Analysis\n\n"
        readme_content += f"{context['clustering_insights']}\n\n"
        if context.get('pca_image'):
            readme_content += f"![PCA Clustering](./{context['pca_image']})\n\n"
        if context.get('tsne_image'):
            readme_content += f"![t-SNE Clustering](./{context['tsne_image']})\n\n"

    if context.get('outlier_insights'):
        readme_content += "## Outlier Detection\n\n"
        readme_content += f"{context['outlier_insights']}\n\n"
        if context.get('outliers_image'):
            readme_content += f"![Outliers Visualization](./{context['outliers_image']})\n\n"

    if context.get('regression_metrics'):
        readme_content += "## Regression Analysis\n\n"
        readme_content += f"{pd.DataFrame(context['regression_metrics'], index=['Value']).to_markdown()}\n\n"
        if context.get('regression_image'):
            readme_content += f"![Regression Results](./{context['regression_image']})\n\n""

    readme_content += "## Conclusion\n\n"
    readme_content += story + "\n"

    try:
        with open(readme_file, 'w') as f:
            f.write(readme_content)
    except Exception as e:
        print(f"Error saving README: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: uv run autolysis.py <file1.csv> <file2.csv> ...")
        sys.exit(1)

    file_paths = sys.argv[1:]

    for file_path in file_paths:
        if os.path.exists(file_path):
            context = analyze_csv(file_path)
            if context:
                narrate_analysis(context, file_path)
        else:
            print(f"File not found: {file_path}")
