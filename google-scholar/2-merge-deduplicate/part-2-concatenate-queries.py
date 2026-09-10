"""
Concatenate Query Results
Combines all individual query CSV files into a single summary CSV
"""

import os
import pandas as pd
from datetime import datetime
import glob


def load_queries_from_file(filepath: str = "queries.txt"):
    """
    Load queries from a Python file containing a queries list.

    Args:
        filepath: Path to the queries file

    Returns:
        List of (query, year) tuples
    """
    queries = []

    try:
        with open(filepath, 'r') as f:
            content = f.read()

        # Execute the file content to get the queries variable
        exec_globals = {}
        exec(content, exec_globals)
        queries = exec_globals.get('queries', [])

        print(f"Loaded {len(queries)} queries from {filepath}")
        return queries

    except Exception as e:
        print(f"Error loading queries file: {e}")
        return []


def concatenate_query_results(queries_dir: str = "queries", output_file: str = "summary_all_queries.csv", queries_file: str = "queries.txt"):
    """
    Concatenate all query CSV files into a single summary file.

    Args:
        queries_dir: Directory containing individual query CSV files
        output_file: Output filename for the concatenated results
        queries_file: Path to the queries.txt file
    """

    if not os.path.exists(queries_dir):
        print(f"Error: Directory '{queries_dir}' not found!")
        return

    # Find all CSV files in the queries directory
    csv_files = glob.glob(os.path.join(queries_dir, "*.csv"))

    if not csv_files:
        print(f"No CSV files found in '{queries_dir}'")
        return

    print(f"Found {len(csv_files)} CSV files to process\n")

    # Load original queries to map filenames to queries
    queries_list = load_queries_from_file(queries_file)

    # Create a more flexible query mapping
    query_map = {}
    for query, year in queries_list:
        # Create multiple possible filename formats
        safe_query = "".join(c if c.isalnum() or c == ' ' else "_" for c in query)
        safe_query_normalized = "_".join(safe_query.split())
        query_map[safe_query_normalized.lower()] = (query, year)

    all_papers = []
    file_stats = []

    for csv_file in sorted(csv_files):
        try:
            # Read CSV
            df = pd.read_csv(csv_file)

            if df.empty:
                print(f"⚠ Skipping empty file: {os.path.basename(csv_file)}")
                continue

            # Extract query name from filename
            filename = os.path.basename(csv_file).replace('.csv', '').lower()

            # Try to match with original query using normalized filename
            matched_query = "Unknown"
            year_filter = "Unknown"

            # Direct match
            if filename in query_map:
                matched_query, year_filter = query_map[filename]
            else:
                # Fuzzy match - check if filename contains query key or vice versa
                for query_key, (original_query, year) in query_map.items():
                    if query_key in filename or filename in query_key:
                        matched_query = original_query
                        year_filter = year
                        break

            if matched_query == "Unknown":
                print(f"⚠ Warning: Could not match query for file: {os.path.basename(csv_file)}")

            # Add metadata columns
            df['query'] = matched_query
            df['year_filter'] = year_filter
            df['source_file'] = os.path.basename(csv_file)

            all_papers.append(df)

            # Track statistics
            file_stats.append({
                'filename': os.path.basename(csv_file),
                'query': matched_query,
                'year_filter': year_filter,
                'num_papers': len(df),
                'date_processed': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })

            print(f"✓ Loaded {len(df):3d} papers from: {os.path.basename(csv_file)}")

        except Exception as e:
            print(f"⚠ Error reading {csv_file}: {e}")
            continue

    if not all_papers:
        print("\nNo papers to concatenate!")
        return

    # Concatenate all dataframes
    print("\nCombining all results...")
    combined_df = pd.concat(all_papers, ignore_index=True)

    # Reorder columns to put metadata first
    column_order = ['query', 'year_filter', 'title', 'authors', 'year', 'publisher', 'abstract', 'url', 'source_file']
    combined_df = combined_df[column_order]

    # Save combined results
    combined_df.to_csv(output_file, index=False, encoding='utf-8')
    print(f"\n✓ Combined results saved to: {output_file}")
    print(f"  Total papers: {len(combined_df)}")
    print(f"  Total queries: {len(all_papers)}")

    # Create statistics summary
    stats_df = pd.DataFrame(file_stats)
    stats_file = os.path.join(os.path.dirname(output_file), "search_statistics.csv")
    stats_df.to_csv(stats_file, index=False, encoding='utf-8')
    print(f"✓ Statistics saved to: {stats_file}")

    # Print summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)
    print(f"Total unique queries processed: {len(stats_df)}")
    print(f"Total papers collected: {len(combined_df)}")
    print(f"Average papers per query: {len(combined_df) / len(stats_df):.1f}")
    print(f"Min papers in a query: {stats_df['num_papers'].min()}")
    print(f"Max papers in a query: {stats_df['num_papers'].max()}")

    # Year distribution
    print("\n" + "=" * 70)
    print("YEAR DISTRIBUTION")
    print("=" * 70)
    year_counts = combined_df['year'].value_counts().sort_index(ascending=False)
    print(year_counts.head(10))

    # Publisher distribution
    print("\n" + "=" * 70)
    print("TOP 10 PUBLISHERS")
    print("=" * 70)
    publisher_counts = combined_df['publisher'].value_counts()
    print(publisher_counts.head(10))

    # Remove duplicates based on title (case-insensitive)
    print("\n" + "=" * 70)
    print("DUPLICATE DETECTION")
    print("=" * 70)

    # Create lowercase title for comparison
    combined_df['title_lower'] = combined_df['title'].str.lower().str.strip()
    duplicates = combined_df[combined_df.duplicated(subset=['title_lower'], keep=False)]

    print(f"Papers appearing in multiple queries: {len(duplicates)}")
    print(f"Unique duplicate titles: {duplicates['title_lower'].nunique()}")

    # Save deduplicated version
    deduplicated_df = combined_df.drop_duplicates(subset=['title_lower'], keep='first')
    deduplicated_df = deduplicated_df.drop(columns=['title_lower'])

    dedup_file = os.path.join(os.path.dirname(output_file), "summary_all_queries_deduplicated.csv")
    deduplicated_df.to_csv(dedup_file, index=False, encoding='utf-8')
    print(f"\n✓ Deduplicated results saved to: {dedup_file}")
    print(f"  Unique papers: {len(deduplicated_df)}")
    print(f"  Removed duplicates: {len(combined_df) - len(deduplicated_df)}")

    return combined_df, stats_df


def main():
    """Main function to concatenate query results."""

    print("=" * 70)
    print("Query Results Concatenation Tool")
    print("=" * 70)
    print()

    # Get the script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Check if queries directory exists (in part-1-apply-criteria)
    queries_dir = os.path.join(script_dir, "../part-1-apply-criteria/queries")
    queries_file = os.path.join(script_dir, "../part-1-apply-criteria/queries.txt")

    if not os.path.exists(queries_dir):
        print(f"Error: '{queries_dir}' directory not found!")
        print("Please run the scraper first to generate query results.")
        return

    if not os.path.exists(queries_file):
        print(f"Error: '{queries_file}' not found!")
        return

    # Process and concatenate
    concatenate_query_results(
        queries_dir=queries_dir,
        output_file=os.path.join(script_dir, "summary_all_queries.csv"),
        queries_file=queries_file
    )

    print("\n" + "=" * 70)
    print("Processing complete!")
    print("=" * 70)
    print("\nGenerated files:")
    print("  1. summary_all_queries.csv - All papers with query metadata")
    print("  2. summary_all_queries_deduplicated.csv - Unique papers only")
    print("  3. search_statistics.csv - Statistics for each query")


if __name__ == "__main__":
    main()
