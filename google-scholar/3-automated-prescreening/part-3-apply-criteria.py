"""
Apply Inclusion/Exclusion Criteria
Rule-based filtering for systematic review screening
"""

import pandas as pd
import os
from datetime import datetime


def filter_grey_literature_and_preprints(df):
    """
    Remove grey literature and preprint sources using rule-based filtering.

    Args:
        df: DataFrame with papers

    Returns:
        Filtered DataFrame
    """
    # Grey literature sources
    grey_sources = [
        'researchgate',
        'academia.edu',
        'ssrn.com',  # Social Science Research Network
    ]

    # Preprint servers (excluding peer-reviewed content)
    preprint_sources = [
        'arxiv',
        'biorxiv',
        'medrxiv',
        'psyarxiv',
        'chemrxiv',
        'preprints.org',
    ]

    print("\n" + "=" * 70)
    print("FILTERING GREY LITERATURE AND PREPRINTS")
    print("=" * 70)

    initial_count = len(df)

    # Filter out grey literature
    df['is_grey_literature'] = df['publisher'].apply(
        lambda x: any(source in str(x).lower() for source in grey_sources)
    )

    # Filter out preprints
    df['is_preprint'] = df['publisher'].apply(
        lambda x: any(source in str(x).lower() for source in preprint_sources)
    )

    # Combined exclusion
    df['exclude_grey_or_preprint'] = df['is_grey_literature'] | df['is_preprint']

    grey_count = df['is_grey_literature'].sum()
    preprint_count = df['is_preprint'].sum()
    total_excluded = df['exclude_grey_or_preprint'].sum()

    print(f"Grey literature found: {grey_count} papers")
    print(f"Preprints found: {preprint_count} papers")
    print(f"Total to exclude: {total_excluded} papers")

    if grey_count > 0:
        print("\nGrey literature sources:")
        grey_publishers = df[df['is_grey_literature']]['publisher'].value_counts()
        for publisher, count in grey_publishers.items():
            print(f"  - {publisher}: {count} papers")

    if preprint_count > 0:
        print("\nPreprint sources:")
        preprint_publishers = df[df['is_preprint']]['publisher'].value_counts()
        for publisher, count in preprint_publishers.items():
            print(f"  - {publisher}: {count} papers")

    # Remove grey literature and preprints
    df_filtered = df[~df['exclude_grey_or_preprint']].copy()

    print(f"\nRemoved: {initial_count - len(df_filtered)} papers")
    print(f"Remaining: {len(df_filtered)} papers")

    return df_filtered


def check_year_criteria(df):
    """
    Apply year inclusion criteria.

    Args:
        df: DataFrame with papers

    Returns:
        DataFrame with year_criteria column
    """
    print("\n" + "=" * 70)
    print("CHECKING YEAR CRITERIA")
    print("=" * 70)

    # Convert year to numeric
    df['year_numeric'] = pd.to_numeric(df['year'], errors='coerce')

    # Year criteria: 2020+ for AI papers, 2015+ for neuroscience
    # Check if query is neuroscience-related
    neuroscience_keywords = ['psilocybin', 'psychedelic', 'default mode', 'brain', 'neural']

    def check_year_threshold(row):
        query_lower = str(row.get('query', '')).lower()
        year = row['year_numeric']

        if pd.isna(year):
            return True  # Keep papers with unknown year for manual review

        # Neuroscience papers: 2015+
        if any(kw in query_lower for kw in neuroscience_keywords):
            return year >= 2015

        # AI papers: 2020+
        return year >= 2020

    df['meets_year_criteria'] = df.apply(check_year_threshold, axis=1)

    passes = df['meets_year_criteria'].sum()
    fails = len(df) - passes

    print(f"Papers meeting year criteria: {passes}")
    print(f"Papers failing year criteria: {fails}")

    if fails > 0:
        print("\nYear distribution of excluded papers:")
        excluded_years = df[~df['meets_year_criteria']]['year_numeric'].value_counts().sort_index()
        print(excluded_years)

    return df


def check_quality_venues(df):
    """
    Check if papers are from TOP-TIER quality venues only (STRICT).
    Peer-reviewed only, no preprints.

    Args:
        df: DataFrame with papers

    Returns:
        DataFrame with quality_venue column
    """
    print("\n" + "=" * 70)
    print("CHECKING VENUE QUALITY (TOP-TIER ONLY - STRICT)")
    print("=" * 70)

    # STRICT: Only top-tier venues
    quality_venues = [
        # Top journals only
        'nature', 'science', 'pnas', 'jmlr',

        # Top AI/ML conferences
        'neurips', 'icml', 'iclr', 'aaai', 'cvpr',
        'pmlr', 'proceedings.mlr.press',

        # Top NLP conferences
        'acl anthology', 'aclweb',

        # Top cognitive science
        'cognitive science',

        # Selective high-quality journals
        'frontiers in', 'ieee transactions', 'acm transactions',

        # Note: Removed broader publishers (Springer, Elsevier, Wiley, MDPI)
        # to only keep top-tier venues
    ]

    df['quality_venue'] = df['publisher'].apply(
        lambda x: any(venue in str(x).lower() for venue in quality_venues)
    )

    quality_count = df['quality_venue'].sum()
    non_quality_count = len(df) - quality_count

    print(f"Papers from TOP-TIER venues: {quality_count}")
    print(f"Papers from other venues: {non_quality_count}")

    if non_quality_count > 0:
        print("\nNon-top-tier venue publishers:")
        non_quality_pubs = df[~df['quality_venue']]['publisher'].value_counts().head(20)
        for pub, count in non_quality_pubs.items():
            print(f"  - {pub}: {count} papers")

    return df


def check_relevance_keywords(df):
    """
    Check for relevance based on keywords in title/abstract.
    STRICT CRITERIA: Requires ≥3 keywords for inclusion.

    Args:
        df: DataFrame with papers

    Returns:
        DataFrame with relevant_content column
    """
    print("\n" + "=" * 70)
    print("CHECKING KEYWORD RELEVANCE (STRICT)")
    print("=" * 70)

    # Core relevance keywords - STRICTER: more specific terms
    # Removed generic terms like 'creative', 'neural network', 'machine learning'
    relevance_keywords = [
        # Neurosymbolic & reasoning (core to your research)
        'neurosymbolic', 'symbolic reasoning', 'knowledge integration',
        'hybrid ai', 'neural-symbolic', 'neuro-symbolic',

        # Specific creativity terms (not just 'creative')
        'creativity', 'creative system', 'creative process',
        'divergent thinking', 'convergent thinking',

        # Thinking processes
        'critical thinking', 'analogical reasoning', 'analogy',

        # Cognitive architecture
        'dual process', 'system 1', 'system 2', 'cognitive architecture',
        'executive control', 'cognitive control',

        # Uncertainty & exploration (specific)
        'epistemic uncertainty', 'aleatoric uncertainty',
        'curiosity driven', 'intrinsic motivation',

        # Ensemble & composition
        'mixture of experts', 'compositional generation', 'moe',

        # Language models (specific)
        'language model', 'llm', 'large language model',

        # Multimodal (specific)
        'multimodal reasoning', 'vision-language',

        # Neuroscience (for foundational papers)
        'psilocybin', 'psychedelic', 'default mode network',
        '5-ht2a', 'serotonin receptor'
    ]

    def count_keywords(row):
        text = f"{row['title']} {row['abstract']}".lower()
        count = sum(1 for keyword in relevance_keywords if keyword in text)
        return count

    df['keyword_count'] = df.apply(count_keywords, axis=1)
    # STRICT: Require ≥3 keywords instead of 2
    df['relevant_content'] = df['keyword_count'] >= 3

    relevant_count = df['relevant_content'].sum()
    irrelevant_count = len(df) - relevant_count

    print(f"Papers with ≥3 keywords: {relevant_count}")
    print(f"Papers with <3 keywords: {irrelevant_count}")

    print("\nKeyword count distribution:")
    print(df['keyword_count'].value_counts().sort_index(ascending=False))

    return df


def check_exclusion_patterns(df):
    """
    Check for exclusion patterns in titles.

    Args:
        df: DataFrame with papers

    Returns:
        DataFrame with passes_exclusion column
    """
    print("\n" + "=" * 70)
    print("CHECKING EXCLUSION PATTERNS")
    print("=" * 70)

    exclusion_patterns = [
        'book review', 'editorial', 'corrigendum', 'retraction',
        'phd thesis', 'dissertation', 'patent', 'poster presentation',
        'erratum', 'corrigenda', 'book chapter'
    ]

    def check_exclusions(title):
        title_lower = str(title).lower()
        return not any(pattern in title_lower for pattern in exclusion_patterns)

    df['passes_exclusion'] = df['title'].apply(check_exclusions)

    passes = df['passes_exclusion'].sum()
    excluded = len(df) - passes

    print(f"Papers passing exclusion check: {passes}")
    print(f"Papers with exclusion patterns: {excluded}")

    if excluded > 0:
        print("\nExcluded titles:")
        excluded_df = df[~df['passes_exclusion']][['title', 'publisher']]
        for idx, row in excluded_df.head(10).iterrows():
            print(f"  - {row['title'][:80]}...")

    return df


def prioritize_papers_per_query(df, max_per_query: int = 3):
    """
    Prioritize and limit papers per query cluster.
    Keeps only the top N papers per query based on relevance score.

    Args:
        df: DataFrame with papers
        max_per_query: Maximum papers to keep per query (default 3)

    Returns:
        DataFrame with priority_rank column
    """
    print("\n" + "=" * 70)
    print(f"PRIORITIZING PAPERS (MAX {max_per_query} PER QUERY)")
    print("=" * 70)

    # Calculate relevance score
    df['relevance_score'] = df['keyword_count']

    # Add priority rank within each query
    df['priority_rank'] = df.groupby('query')['relevance_score'].rank(
        method='first', ascending=False
    )

    # Mark as high priority if within top N per query
    df['high_priority'] = df['priority_rank'] <= max_per_query

    # Print statistics per query
    print("\nPapers per query (before/after prioritization):")
    for query in df['query'].unique():
        query_df = df[df['query'] == query]
        before = len(query_df)
        after = query_df['high_priority'].sum()
        print(f"  {query[:50]:50s}: {before:3d} → {after:2d}")

    high_priority_count = df['high_priority'].sum()
    print(f"\nTotal high-priority papers: {high_priority_count}")

    return df


def apply_screening_decision(df):
    """
    Make final screening decision based on all criteria.
    STRICT MODE: Only includes high-priority papers.

    Args:
        df: DataFrame with papers and screening columns

    Returns:
        DataFrame with screening decision
    """
    print("\n" + "=" * 70)
    print("APPLYING SCREENING DECISIONS (STRICT MODE)")
    print("=" * 70)

    # Auto-include: meets all criteria AND is high priority
    df['auto_include'] = (
        df['meets_year_criteria'] &
        df['quality_venue'] &
        df['relevant_content'] &
        df['passes_exclusion'] &
        df['high_priority']  # NEW: Must be high priority
    )

    # Auto-exclude: fails critical criteria OR low priority
    df['auto_exclude'] = (
        ~df['meets_year_criteria'] |
        ~df['passes_exclusion'] |
        (df['keyword_count'] < 3) |  # STRICT: <3 keywords
        ~df['high_priority']  # NEW: Low priority = exclude
    )

    # Maybe: requires manual review (meets criteria but edge cases)
    df['needs_review'] = ~(df['auto_include'] | df['auto_exclude'])

    # Set screening status
    def get_decision(row):
        if row['auto_include']:
            return 'include'
        elif row['auto_exclude']:
            return 'exclude'
        else:
            return 'maybe'

    df['screening_decision'] = df.apply(get_decision, axis=1)

    # Add notes for manual reviewers
    def get_notes(row):
        notes = []
        if not row['meets_year_criteria']:
            notes.append(f"Year {row['year_numeric']} outside criteria")
        if not row['quality_venue']:
            notes.append(f"Non-top-tier venue: {row['publisher']}")
        if not row['relevant_content']:
            notes.append(f"Low keyword match ({row['keyword_count']} keywords)")
        if not row['passes_exclusion']:
            notes.append("Exclusion pattern in title")
        if not row['high_priority']:
            notes.append(f"Low priority (rank {int(row['priority_rank'])} in query)")
        return "; ".join(notes) if notes else "Meets all criteria"

    df['screening_notes'] = df.apply(get_notes, axis=1)

    # Summary statistics
    include_count = (df['screening_decision'] == 'include').sum()
    exclude_count = (df['screening_decision'] == 'exclude').sum()
    maybe_count = (df['screening_decision'] == 'maybe').sum()

    print(f"\nScreening Summary:")
    print(f"  AUTO-INCLUDE: {include_count} papers ({include_count/len(df)*100:.1f}%)")
    print(f"  AUTO-EXCLUDE: {exclude_count} papers ({exclude_count/len(df)*100:.1f}%)")
    print(f"  NEEDS REVIEW: {maybe_count} papers ({maybe_count/len(df)*100:.1f}%)")

    return df


def main():
    """Main function to apply screening criteria."""

    print("=" * 70)
    print("SYSTEMATIC REVIEW SCREENING - RULE-BASED FILTERING")
    print("=" * 70)

    # Get the script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Check for input file from part-2
    input_file = os.path.join(script_dir, "../part-2-concatenate-queries/summary_all_queries.csv")

    if not os.path.exists(input_file):
        print(f"\nError: '{input_file}' not found!")
        print("Please run part-2-concatenate-queries.py first.")
        return

    # Load data
    print(f"\nLoading: {input_file}")
    df = pd.read_csv(input_file)
    print(f"Total papers to screen: {len(df)}")

    # Apply filtering steps
    print("\n" + "=" * 70)
    print("STEP 1: REMOVE GREY LITERATURE AND PREPRINTS")
    print("=" * 70)
    df_filtered = filter_grey_literature_and_preprints(df)

    print("\n" + "=" * 70)
    print("STEP 2: APPLY INCLUSION CRITERIA")
    print("=" * 70)
    df_filtered = check_year_criteria(df_filtered)
    df_filtered = check_quality_venues(df_filtered)
    df_filtered = check_relevance_keywords(df_filtered)
    df_filtered = check_exclusion_patterns(df_filtered)

    print("\n" + "=" * 70)
    print("STEP 3: PRIORITIZE PAPERS PER QUERY")
    print("=" * 70)
    # Limit to top 2-3 papers per query for ~40 total (20 queries × 2)
    df_filtered = prioritize_papers_per_query(df_filtered, max_per_query=3)

    print("\n" + "=" * 70)
    print("STEP 4: MAKE SCREENING DECISIONS")
    print("=" * 70)
    df_filtered = apply_screening_decision(df_filtered)

    # Save filtered results to script directory
    output_file = os.path.join(script_dir, "summary-filtered.csv")
    df_filtered.to_csv(output_file, index=False, encoding='utf-8')
    print(f"\n✓ Filtered results saved to: {output_file}")

    # Save only included papers
    df_included = df_filtered[df_filtered['screening_decision'] == 'include'].copy()
    included_file = os.path.join(script_dir, "summary-included.csv")
    df_included.to_csv(included_file, index=False, encoding='utf-8')
    print(f"✓ Included papers saved to: {included_file}")
    print(f"  Total included: {len(df_included)}")

    # Save papers needing review
    df_review = df_filtered[df_filtered['screening_decision'] == 'maybe'].copy()
    review_file = os.path.join(script_dir, "summary-needs-review.csv")
    df_review.to_csv(review_file, index=False, encoding='utf-8')
    print(f"✓ Papers needing review saved to: {review_file}")
    print(f"  Total needing review: {len(df_review)}")

    # Save excluded papers
    df_excluded = df_filtered[df_filtered['screening_decision'] == 'exclude'].copy()
    excluded_file = os.path.join(script_dir, "summary-excluded.csv")
    df_excluded.to_csv(excluded_file, index=False, encoding='utf-8')
    print(f"✓ Excluded papers saved to: {excluded_file}")
    print(f"  Total excluded: {len(df_excluded)}")

    # Final summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"Initial papers (after deduplication): {len(df)}")
    print(f"After grey literature removal: {len(df_filtered)}")
    print(f"  → Auto-included: {len(df_included)}")
    print(f"  → Needs manual review: {len(df_review)}")
    print(f"  → Auto-excluded: {len(df_excluded)}")

    print("\n" + "=" * 70)
    print("NEXT STEPS")
    print("=" * 70)
    print("1. Review papers in 'summary-needs-review.csv'")
    print("2. Make final include/exclude decisions")
    print("3. Update 'screening_decision' column in 'summary-filtered.csv'")
    print("4. Proceed with full-text review of included papers")

    print("\n✓ Screening complete!")


if __name__ == "__main__":
    main()
