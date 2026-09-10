"""
part-2-preprocess-bigquery-results.py
Preprocesses, deduplicates, and cleans BigQuery PubMed search results
"""

import pandas as pd
import os
import re
from datetime import datetime


def load_data(filepath: str):
    """
    Load BigQuery CSV export.
    
    Args:
        filepath: Path to the CSV file
    
    Returns:
        DataFrame with raw results
    """
    print("=" * 70)
    print("LOADING DATA")
    print("=" * 70)
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} rows from {filepath}")
    print(f"Columns: {list(df.columns)}")
    
    return df


def basic_stats(df: pd.DataFrame):
    """
    Print basic statistics about the data.
    
    Args:
        df: DataFrame with results
    """
    print("\n" + "=" * 70)
    print("BASIC STATISTICS")
    print("=" * 70)
    
    print(f"\nTotal rows: {len(df)}")
    print(f"Unique papers (by pmc_id): {df['pmc_id'].nunique()}")
    
    print(f"\nPapers per cluster:")
    for cluster in sorted(df['query_cluster'].unique()):
        count = len(df[df['query_cluster'] == cluster])
        print(f"  {cluster}: {count}")
    
    print(f"\nSemantic distance range:")
    print(f"  Min: {df['semantic_distance'].min():.4f}")
    print(f"  Max: {df['semantic_distance'].max():.4f}")
    print(f"  Mean: {df['semantic_distance'].mean():.4f}")


def deduplicate_across_clusters(df: pd.DataFrame):
    """
    Remove duplicates across clusters, keeping the best match (lowest distance).
    
    Args:
        df: DataFrame with results
    
    Returns:
        Deduplicated DataFrame
    """
    print("\n" + "=" * 70)
    print("DEDUPLICATION ACROSS CLUSTERS")
    print("=" * 70)
    
    initial_count = len(df)
    
    # Find papers appearing in multiple clusters
    dup_papers = df.groupby('pmc_id').filter(lambda x: len(x) > 1)
    dup_count = dup_papers['pmc_id'].nunique()
    
    print(f"Papers appearing in multiple clusters: {dup_count}")
    
    if dup_count > 0:
        print("\nDuplicate papers:")
        for pmc_id in dup_papers['pmc_id'].unique()[:10]:
            paper = df[df['pmc_id'] == pmc_id]
            title = paper['title'].iloc[0][:60] if pd.notna(paper['title'].iloc[0]) else "No title"
            clusters = paper['query_cluster'].tolist()
            print(f"  {title}...")
            print(f"    Clusters: {clusters}")
        if dup_count > 10:
            print(f"  ... and {dup_count - 10} more")
    
    # Keep best match (lowest semantic distance) for each paper
    df_dedup = df.sort_values('semantic_distance').drop_duplicates(subset=['pmc_id'], keep='first')
    
    removed = initial_count - len(df_dedup)
    print(f"\nRemoved {removed} duplicate rows")
    print(f"Remaining unique papers: {len(df_dedup)}")
    
    return df_dedup


def clean_abstract(text):
    """
    Clean abstract text by removing artifacts and normalizing whitespace.
    
    Args:
        text: Raw abstract text
    
    Returns:
        Cleaned abstract
    """
    if pd.isna(text):
        return ""
    
    text = str(text)
    
    # Remove common artifacts
    text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
    text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)  # Remove control characters
    
    # Try to extract just the abstract if full text was captured
    abstract_match = re.search(r'(?i)abstract[:\s]*(.+?)(?:introduction|background|keywords|methods|1\.|$)', text, re.DOTALL)
    if abstract_match:
        text = abstract_match.group(1).strip()
    
    # Limit length
    if len(text) > 3000:
        text = text[:3000] + "..."
    
    return text.strip()


def clean_title(text):
    """
    Clean title text.
    
    Args:
        text: Raw title
    
    Returns:
        Cleaned title
    """
    if pd.isna(text):
        return ""
    
    text = str(text)
    text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
    text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)  # Remove control characters
    
    return text.strip()


def clean_authors(text):
    """
    Clean author text.
    
    Args:
        text: Raw author string
    
    Returns:
        Cleaned author string
    """
    if pd.isna(text):
        return ""
    
    text = str(text)
    text = re.sub(r'\s+', ' ', text)
    
    # Limit very long author lists
    if len(text) > 500:
        text = text[:500] + "..."
    
    return text.strip()


def clean_data(df: pd.DataFrame):
    """
    Clean all text fields in the DataFrame.
    
    Args:
        df: DataFrame with results
    
    Returns:
        Cleaned DataFrame
    """
    print("\n" + "=" * 70)
    print("CLEANING DATA")
    print("=" * 70)
    
    df = df.copy()
    
    # Clean text fields
    print("Cleaning titles...")
    df['title'] = df['title'].apply(clean_title)
    
    print("Cleaning abstracts...")
    df['abstract'] = df['abstract'].apply(clean_abstract)
    
    print("Cleaning authors...")
    df['author'] = df['author'].apply(clean_authors)
    
    # Ensure URLs are complete
    print("Validating URLs...")
    df['full_article_url'] = df['pmc_link'].apply(
        lambda x: x if pd.notna(x) and x.startswith('http') else ''
    )
    
    # Remove rows with missing essential data
    before = len(df)
    df = df[df['title'].str.len() > 0]
    df = df[df['pmc_id'].notna()]
    after = len(df)
    
    if before > after:
        print(f"Removed {before - after} rows with missing title or pmc_id")
    
    print("Data cleaning complete")
    
    return df


def add_review_columns(df: pd.DataFrame):
    """
    Add columns for manual review process.
    
    Args:
        df: DataFrame with results
    
    Returns:
        DataFrame with review columns
    """
    print("\n" + "=" * 70)
    print("ADDING REVIEW COLUMNS")
    print("=" * 70)
    
    df = df.copy()
    
    df['include'] = ''
    df['exclude_reason'] = ''
    df['relevance_notes'] = ''
    df['reviewer'] = ''
    
    print("Added columns: include, exclude_reason, relevance_notes, reviewer")
    
    return df


def reorder_columns(df: pd.DataFrame):
    """
    Reorder columns for easier review.
    
    Args:
        df: DataFrame with results
    
    Returns:
        DataFrame with reordered columns
    """
    column_order = [
        'query_cluster',
        'pmc_id',
        'title',
        'author',
        'abstract',
        'full_article_url',
        'semantic_distance',
        'include',
        'exclude_reason',
        'relevance_notes',
        'reviewer'
    ]
    
    # Only include columns that exist
    existing_cols = [c for c in column_order if c in df.columns]
    other_cols = [c for c in df.columns if c not in column_order]
    
    return df[existing_cols + other_cols]


def generate_summary_by_cluster(df: pd.DataFrame):
    """
    Print summary statistics by cluster.
    
    Args:
        df: DataFrame with results
    """
    print("\n" + "=" * 70)
    print("SUMMARY BY CLUSTER")
    print("=" * 70)
    
    for cluster in sorted(df['query_cluster'].unique()):
        cluster_df = df[df['query_cluster'] == cluster]
        print(f"\n{cluster}:")
        print(f"  Papers: {len(cluster_df)}")
        print(f"  Distance range: {cluster_df['semantic_distance'].min():.4f} - {cluster_df['semantic_distance'].max():.4f}")
        print(f"  Top 3 titles:")
        for _, row in cluster_df.head(3).iterrows():
            title = row['title'][:70] if len(row['title']) > 70 else row['title']
            print(f"    [{row['semantic_distance']:.3f}] {title}")


def save_outputs(df: pd.DataFrame, output_dir: str):
    """
    Save processed data to files.
    
    Args:
        df: Processed DataFrame
        output_dir: Output directory
    """
    print("\n" + "=" * 70)
    print("SAVING OUTPUTS")
    print("=" * 70)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save full processed file
    output_file = os.path.join(output_dir, "papers_for_review.csv")
    df.to_csv(output_file, index=False, encoding='utf-8')
    print(f"Saved: {output_file} ({len(df)} papers)")
    
    # Save per-cluster files
    for cluster in df['query_cluster'].unique():
        cluster_df = df[df['query_cluster'] == cluster]
        safe_name = cluster.replace(' ', '_').replace('/', '_')
        cluster_file = os.path.join(output_dir, f"cluster_{safe_name}.csv")
        cluster_df.to_csv(cluster_file, index=False, encoding='utf-8')
        print(f"Saved: {cluster_file} ({len(cluster_df)} papers)")
    
    # Save statistics summary
    stats = {
        'total_papers': len(df),
        'unique_clusters': df['query_cluster'].nunique(),
        'papers_per_cluster': df['query_cluster'].value_counts().to_dict(),
        'processed_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    stats_file = os.path.join(output_dir, "processing_stats.txt")
    with open(stats_file, 'w') as f:
        f.write("BigQuery PubMed Search Results - Processing Statistics\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Processed: {stats['processed_date']}\n")
        f.write(f"Total unique papers: {stats['total_papers']}\n")
        f.write(f"Number of clusters: {stats['unique_clusters']}\n\n")
        f.write("Papers per cluster:\n")
        for cluster, count in stats['papers_per_cluster'].items():
            f.write(f"  {cluster}: {count}\n")
    print(f"Saved: {stats_file}")


def main():
    """Main function to preprocess BigQuery results."""
    
    print("=" * 70)
    print("PART 2: PREPROCESS BIGQUERY PUBMED RESULTS")
    print("=" * 70)
    print()
    
    # Configuration
    input_file = "/home/sebasmos/Desktop/AnpassenNN/mit-projects/creative-ai/code-pubmed-vertex/part-1-get-vertexpubmed/data.csv"
    output_dir = "/home/sebasmos/Desktop/AnpassenNN/mit-projects/creative-ai/code-pubmed-vertex/part-2-preprocessong"
    
    # Load data
    df = load_data(input_file)
    
    # Basic stats
    basic_stats(df)
    
    # Deduplicate across clusters
    df = deduplicate_across_clusters(df)
    
    # Clean data
    df = clean_data(df)
    
    # Add review columns
    df = add_review_columns(df)
    
    # Reorder columns
    df = reorder_columns(df)
    
    # Summary by cluster
    generate_summary_by_cluster(df)
    
    # Save outputs
    save_outputs(df, output_dir)
    
    # Final summary
    print("\n" + "=" * 70)
    print("PROCESSING COMPLETE")
    print("=" * 70)
    print(f"\nTotal unique papers for review: {len(df)}")
    print(f"\nOutput files saved to: {output_dir}")
    print("\nNext steps:")
    print("1. Open 'papers_for_review.csv' in Excel/Google Sheets")
    print("2. Review each paper and mark 'include' column as 'yes' or 'no'")
    print("3. Add notes in 'relevance_notes' for borderline cases")
    print("4. Run part-3 to apply filtering criteria and generate final list")


if __name__ == "__main__":
    main()