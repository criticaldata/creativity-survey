"""
Part 4: ChatGPT API-Based Review of Creative AI Papers (Google Scholar)

This script uses the OpenAI API to review papers (based on abstracts) for inclusion
in a narrative review about Creative AI and Neurosymbolic approaches.

Decisions: Accept, Reject, Maybe
"""

import pandas as pd
import os
import json
import time
from openai import OpenAI
from pathlib import Path
from dotenv import load_dotenv

# =============================================================================
# CONFIGURATION
# =============================================================================

# Load API key from vibe-science .env file
ENV_PATH = Path("/home/sebasmos/Desktop/AnpassenNN/mit-projects/vibe-science/code copy/part-3-chatgpt-review/.env")
load_dotenv(ENV_PATH)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = "gpt-5.2"

# Input/Output paths
BASE_DIR = Path("/home/sebasmos/Desktop/AnpassenNN/mit-projects/creative-ai/code-google-scholar/part-4-review-chatgpt")
INPUT_CSV = BASE_DIR / "summary-needs-review - summary-needs-review.csv"
OUTPUT_CSV = BASE_DIR / "summary-reviewed-chatgpt.csv"

# Creative AI context for the prompt
CREATIVE_AI_CONTEXT = """
This narrative review focuses on Creative AI and Neurosymbolic approaches to human-AI systems.

5 THEMATIC CLUSTERS OF INTEREST:
1. Neurosymbolic AI & Creativity - Hybrid systems combining neural networks with symbolic reasoning
2. Cognitive Architecture - Dual process theory, divergent/convergent thinking, System 1/System 2
3. Uncertainty & Exploration - How uncertainty drives exploration and creativity
4. Ensemble & Compositional Methods - Mixture of Experts (MoE), compositional approaches
5. Psychedelic Neuroscience - Research on altered states, creativity, and cognitive flexibility

We are interested in papers that:
- Study creativity (human or AI)
- Propose cognitive architectures or frameworks
- Explore neurosymbolic approaches
- Investigate divergent/convergent thinking
- Study psychedelics and creativity/cognition
- Discuss AI systems that generate novel outputs
- Provide foundational/background knowledge for understanding creativity
"""


def create_review_prompt(title: str, authors: str, year: str, abstract: str, publisher: str) -> str:
    """Create the prompt for ChatGPT to review a paper based on abstract."""
    return f"""You are reviewing papers for a narrative review on Creative AI and Neurosymbolic approaches.

{CREATIVE_AI_CONTEXT}

PAPER TO REVIEW:
Title: {title}
Authors: {authors}
Year: {year}
Publisher: {publisher}

Abstract:
{abstract}

TASK:
1. Read the abstract carefully
2. Determine if it relates to any of our 5 thematic clusters
3. Make a decision: Accept, Reject, or Maybe

RESPOND IN THIS EXACT JSON FORMAT:
{{
    "decision": "Accept" or "Reject" or "Maybe",
    "comment": "<ONE sentence explaining your decision, max 100 characters>",
    "primary_theme": "<which of the 5 clusters it fits, or 'None'>"
}}

DECISION GUIDANCE:
- ACCEPT: Relevant to any of the 5 thematic clusters, studies creativity/cognition, proposes frameworks, or provides useful foundational insights
- MAYBE: Partially relevant, might provide useful background or tangential insights
- REJECT: Completely off-topic (e.g., unrelated technical methods, purely clinical with no creativity/cognition focus)

Be inclusive - we want papers that can inform our understanding of creativity, cognition, and AI systems. When in doubt, lean toward Accept or Maybe.
"""


def review_paper_with_gpt(client: OpenAI, title: str, authors: str, year: str, abstract: str, publisher: str) -> dict:
    """Use GPT to review a single paper based on abstract."""
    prompt = create_review_prompt(title, authors, year, abstract, publisher)

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are an expert in AI, cognitive science, and creativity research. Respond only with valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_completion_tokens=500
        )

        content = response.choices[0].message.content.strip()

        # Handle markdown code blocks
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]

        result = json.loads(content)
        return result

    except json.JSONDecodeError as e:
        print(f"  JSON parse error: {e}")
        return {
            "decision": "Maybe",
            "comment": f"JSON parse error: {str(e)[:50]}",
            "primary_theme": "None"
        }
    except Exception as e:
        print(f"  API error: {e}")
        return {
            "decision": "Maybe",
            "comment": f"API error: {str(e)[:50]}",
            "primary_theme": "None"
        }


def main():
    print("=" * 70)
    print("Part 4: ChatGPT API-Based Review of Creative AI Papers (Google Scholar)")
    print(f"Model: {MODEL}")
    print("=" * 70)

    # Initialize OpenAI client
    if not OPENAI_API_KEY:
        print("ERROR: OPENAI_API_KEY not found in .env file")
        return None

    client = OpenAI(api_key=OPENAI_API_KEY)

    # Load input CSV
    if not INPUT_CSV.exists():
        print(f"ERROR: Input file not found: {INPUT_CSV}")
        return None

    df = pd.read_csv(INPUT_CSV)
    print(f"\nLoaded {len(df)} papers from CSV")

    # Initialize result columns
    df['Decision-chatgpt'] = ''
    df['comment-chatgpt'] = ''
    df['theme-chatgpt'] = ''

    # Track statistics
    stats = {"Accept": 0, "Reject": 0, "Maybe": 0}

    print(f"\nReviewing papers with {MODEL}...")
    print("-" * 70)

    for idx, row in df.iterrows():
        title = str(row.get('title', ''))
        authors = str(row.get('authors', ''))
        year = str(row.get('year', ''))
        abstract = str(row.get('abstract', ''))
        publisher = str(row.get('publisher', ''))

        print(f"\n[{idx+1}/{len(df)}] {title[:60]}...")

        # Skip if no abstract
        if not abstract or abstract == 'nan' or len(abstract.strip()) < 50:
            df.at[idx, 'Decision-chatgpt'] = 'Maybe'
            df.at[idx, 'comment-chatgpt'] = 'Abstract too short or missing - needs manual review'
            df.at[idx, 'theme-chatgpt'] = 'None'
            stats["Maybe"] += 1
            print(f"  -> MAYBE (No abstract)")
            continue

        # Send to GPT for review
        print(f"  Sending to {MODEL} for review...")
        result = review_paper_with_gpt(client, title, authors, year, abstract, publisher)

        decision = result.get('decision', 'Maybe')
        comment = result.get('comment', '')[:100]  # Limit to 100 chars
        theme = result.get('primary_theme', 'None')

        df.at[idx, 'Decision-chatgpt'] = decision
        df.at[idx, 'comment-chatgpt'] = comment
        df.at[idx, 'theme-chatgpt'] = theme
        stats[decision] += 1

        print(f"  -> {decision}: {comment}")
        print(f"     Theme: {theme}")

        # Rate limiting
        time.sleep(1.0)

    # Save results
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n{'=' * 70}")
    print("REVIEW COMPLETE")
    print(f"{'=' * 70}")
    print(f"Total papers reviewed: {len(df)}")
    print(f"Accept: {stats['Accept']}")
    print(f"Maybe: {stats['Maybe']}")
    print(f"Reject: {stats['Reject']}")
    print(f"\nResults saved to: {OUTPUT_CSV}")

    # Also update the original CSV with the new columns
    df.to_csv(INPUT_CSV, index=False)
    print(f"Original CSV updated: {INPUT_CSV}")

    return df


if __name__ == "__main__":
    result_df = main()
