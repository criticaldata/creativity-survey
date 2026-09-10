"""
part-3-apply-criteria.py
Apply Inclusion/Exclusion Criteria to BigQuery PubMed Results (STRICT)

Input: /home/sebasmos/Desktop/AnpassenNN/mit-projects/creative-ai/code-pubmed-vertex/part-2-preprocessong/papers_for_review.csv
Output: /home/sebasmos/Desktop/AnpassenNN/mit-projects/creative-ai/code-pubmed-vertex/part-3-excl-criteria/
"""

import pandas as pd
import os
import re
from datetime import datetime


def clean_abstract(text):
    """
    Clean abstract text by removing leading artifacts like '====', '----', etc.
    Ensures abstracts start with letters.
    """
    if pd.isna(text):
        return ""

    text = str(text)

    # Remove leading non-letter characters (=, -, *, spaces, etc.)
    text = re.sub(r'^[^a-zA-Z]+', '', text)

    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def clean_abstracts_in_df(df):
    """
    Clean all abstracts in the DataFrame.
    """
    print("\n" + "=" * 70)
    print("CLEANING ABSTRACTS")
    print("=" * 70)

    df = df.copy()
    df['abstract'] = df['abstract'].apply(clean_abstract)

    # Count how many abstracts were cleaned
    print(f"Cleaned {len(df)} abstracts")

    return df


def check_relevance_keywords(df):
    """
    Check for relevance based on keywords in title/abstract.
    STRICT: Requires >= 3 keywords.
    """
    print("\n" + "=" * 70)
    print("CHECKING KEYWORD RELEVANCE (STRICT)")
    print("=" * 70)

    relevance_keywords = [
        # Neurosymbolic & reasoning
        'neurosymbolic', 'symbolic reasoning', 'neural-symbolic', 'neuro-symbolic',

        # Creativity terms (specific)
        'creativity', 'creative', 'divergent thinking', 'convergent thinking',

        # Cognitive architecture
        'dual process', 'system 1', 'system 2', 'cognitive flexibility',

        # Ensemble & composition
        'mixture of experts', 'compositional', 'moe',

        # Language models
        'language model', 'llm', 'large language model',

        # Multimodal
        'multimodal', 'vision-language', 'cross-modal',

        # Neuroscience (psychedelics)
        'psilocybin', 'psychedelic', 'default mode network', 'dmn',
        '5-ht2a', 'serotonin', 'lsd',

        # AI specific
        'artificial intelligence', 'generative ai', 'neural network'
    ]

    def count_keywords(row):
        title = str(row['title']).lower() if pd.notna(row['title']) else ''
        abstract = str(row['abstract']).lower() if pd.notna(row['abstract']) else ''
        text = f"{title} {abstract}"
        count = sum(1 for keyword in relevance_keywords if keyword in text)
        return count

    df['keyword_count'] = df.apply(count_keywords, axis=1)
    df['relevant_content'] = df['keyword_count'] >= 3  # STRICT: 3+ keywords

    relevant_count = df['relevant_content'].sum()
    print(f"Papers with >=3 keywords: {relevant_count}")
    print(f"Papers with <3 keywords: {len(df) - relevant_count}")

    return df


def check_exclusion_patterns(df):
    """
    Check for exclusion patterns in titles.
    """
    print("\n" + "=" * 70)
    print("CHECKING EXCLUSION PATTERNS")
    print("=" * 70)

    exclusion_patterns = [
        'book review', 'editorial', 'corrigendum', 'retraction',
        'phd thesis', 'dissertation', 'patent', 'poster presentation',
        'erratum', 'corrigenda', 'book chapter', 'letter to editor',
        'commentary', 'correction', 'withdrawn', 'protocol',
        'case report', 'case series', 'author correction', 'reply to',
        'response to', 'erratum', 'addendum', 'supplement'
    ]

    def check_exclusions(title):
        title_lower = str(title).lower() if pd.notna(title) else ''
        return not any(pattern in title_lower for pattern in exclusion_patterns)

    df['passes_exclusion'] = df['title'].apply(check_exclusions)

    passes = df['passes_exclusion'].sum()
    excluded = len(df) - passes

    print(f"Papers passing exclusion check: {passes}")
    print(f"Papers with exclusion patterns: {excluded}")

    return df


def prioritize_by_semantic_distance(df, max_per_cluster: int = 5):
    """
    Prioritize papers within each cluster by semantic distance.
    STRICT: Only top 5 per cluster.
    """
    print("\n" + "=" * 70)
    print(f"PRIORITIZING PAPERS (TOP {max_per_cluster} PER CLUSTER)")
    print("=" * 70)

    df['priority_rank'] = df.groupby('query_cluster')['semantic_distance'].rank(
        method='first', ascending=True
    )

    df['high_priority'] = df['priority_rank'] <= max_per_cluster

    print("\nPapers per cluster (total / high priority):")
    for cluster in sorted(df['query_cluster'].unique()):
        cluster_df = df[df['query_cluster'] == cluster]
        total = len(cluster_df)
        high_pri = cluster_df['high_priority'].sum()
        print(f"  {cluster[:40]:40s}: {total:3d} total / {high_pri:2d} high priority")

    print(f"\nTotal high priority papers: {df['high_priority'].sum()}")

    return df


def apply_screening_decision(df):
    """
    Make final screening decision based on all criteria.
    STRICT MODE.
    """
    print("\n" + "=" * 70)
    print("APPLYING SCREENING DECISIONS (STRICT)")
    print("=" * 70)

    # Auto-include: high priority AND passes exclusion AND 3+ keywords AND low semantic distance
    df['auto_include'] = (
        df['high_priority'] &
        df['passes_exclusion'] &
        df['relevant_content'] &
        (df['semantic_distance'] < 0.82)  # STRICT: Only very close matches
    )

    # Auto-exclude: fails exclusion OR <2 keywords OR high semantic distance
    df['auto_exclude'] = (
        ~df['passes_exclusion'] |
        (df['keyword_count'] < 2) |
        (df['semantic_distance'] > 0.88) |
        (~df['high_priority'] & (df['keyword_count'] < 4))  # Low priority needs 4+ keywords
    )

    # Maybe: everything else
    df['needs_review'] = ~(df['auto_include'] | df['auto_exclude'])

    def get_decision(row):
        if row['auto_include']:
            return 'include'
        elif row['auto_exclude']:
            return 'exclude'
        else:
            return 'needs_review'

    df['screening_decision'] = df.apply(get_decision, axis=1)

    include_count = (df['screening_decision'] == 'include').sum()
    exclude_count = (df['screening_decision'] == 'exclude').sum()
    review_count = (df['screening_decision'] == 'needs_review').sum()

    print(f"\nScreening Summary:")
    print(f"  INCLUDE:      {include_count} papers ({include_count/len(df)*100:.1f}%)")
    print(f"  EXCLUDE:      {exclude_count} papers ({exclude_count/len(df)*100:.1f}%)")
    print(f"  NEEDS REVIEW: {review_count} papers ({review_count/len(df)*100:.1f}%)")

    return df


def save_outputs(df, output_dir: str):
    """
    Save 3 output files: included, excluded, needs_review.
    """
    print("\n" + "=" * 70)
    print("SAVING OUTPUTS")
    print("=" * 70)

    os.makedirs(output_dir, exist_ok=True)

    # Select key columns
    output_columns = [
        'query_cluster',
        'pmc_id',
        'title',
        'author',
        'abstract',
        'full_article_url',
        'semantic_distance',
        'keyword_count',
        'screening_decision'
    ]
    output_columns = [c for c in output_columns if c in df.columns]

    # Save included papers
    df_included = df[df['screening_decision'] == 'include'][output_columns].copy()
    included_file = os.path.join(output_dir, "papers-included.csv")
    df_included.to_csv(included_file, index=False, encoding='utf-8')
    print(f"Saved: {included_file} ({len(df_included)} papers)")

    # Save papers needing review
    df_review = df[df['screening_decision'] == 'needs_review'][output_columns].copy()
    review_file = os.path.join(output_dir, "papers-needs-review.csv")
    df_review.to_csv(review_file, index=False, encoding='utf-8')
    print(f"Saved: {review_file} ({len(df_review)} papers)")

    # Save excluded papers
    df_excluded = df[df['screening_decision'] == 'exclude'][output_columns].copy()
    excluded_file = os.path.join(output_dir, "papers-excluded.csv")
    df_excluded.to_csv(excluded_file, index=False, encoding='utf-8')
    print(f"Saved: {excluded_file} ({len(df_excluded)} papers)")

    return df_included, df_review, df_excluded


def main():
    """Main function to apply screening criteria."""

    print("=" * 70)
    print("PART 3: APPLY SCREENING CRITERIA (STRICT MODE)")
    print("=" * 70)

    input_file = "/home/sebasmos/Desktop/AnpassenNN/mit-projects/creative-ai/code-pubmed-vertex/part-2-preprocessong/papers_for_review.csv"
    output_dir = "/home/sebasmos/Desktop/AnpassenNN/mit-projects/creative-ai/code-pubmed-vertex/part-3-excl-criteria"

    if not os.path.exists(input_file):
        print(f"\nError: '{input_file}' not found!")
        return

    print(f"\nLoading: {input_file}")
    df = pd.read_csv(input_file)
    print(f"Total papers to screen: {len(df)}")

    # Clean abstracts (remove leading "====" and other artifacts)
    df = clean_abstracts_in_df(df)

    # Apply filters
    df = check_relevance_keywords(df)
    df = check_exclusion_patterns(df)
    df = prioritize_by_semantic_distance(df, max_per_cluster=5)
    df = apply_screening_decision(df)

    # Save outputs
    df_included, df_review, df_excluded = save_outputs(df, output_dir)

    # Final summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"Total screened: {len(df)}")
    print(f"  Included:     {len(df_included)}")
    print(f"  Needs review: {len(df_review)}")
    print(f"  Excluded:     {len(df_excluded)}")
    print(f"\nTotal for manual review: {len(df_included) + len(df_review)}")

    print("\nScreening complete!")


if __name__ == "__main__":
    main()