"""
Part 5: Gemini API-Based Review of Creative AI Papers

This script uses the Google Gemini API to review PDF papers for inclusion in a
narrative review about Creative AI and Neurosymbolic approaches.

Decisions: Accept, Reject, Maybe
Exclusion criteria:
- MDPI publications
- Papers prior to 2020
- Editorials
- Background material

Focus: Papers relevant to Creative AI frameworks that operationalize,
measure, or enforce creativity within AI systems.
"""

import pandas as pd
import os
import json
import time
import fitz  # PyMuPDF
import google.generativeai as genai
from pathlib import Path

# =============================================================================
# CONFIGURATION
# =============================================================================

# Gemini API Key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")  # set via environment; never hardcode

# Model configuration - Gemini 3.0 Flash
MODEL = "models/gemini-2.0-flash"  # Using 2.0-flash which has higher rate limits

# Input/Output paths
BASE_DIR = Path("/home/sebasmos/Desktop/AnpassenNN/mit-projects/creative-ai/code-pubmed-vertex/part-5-gemini")
PART4_DIR = Path("/home/sebasmos/Desktop/AnpassenNN/mit-projects/creative-ai/code-pubmed-vertex/part-4-review-chatgpt")
INPUT_CSV = PART4_DIR / "papers-needs-review-blinded.csv"
OUTPUT_CSV = BASE_DIR / "papers-reviewed-gemini.csv"
PDF_DIR = PART4_DIR / "papers-20251229T104717Z-1-001" / "papers"

# MDPI journals to exclude
MDPI_JOURNALS = [
    "brainsci", "sensors", "entropy", "jintelligence", "ijms", "algorithms",
    "applsci", "biomedicines", "diagnostics", "electronics", "healthcare",
    "jcm", "jpm", "life", "mathematics", "medicina", "molecules", "nutrients"
]

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
"""


def extract_pdf_text(pdf_path: Path, max_pages: int = None) -> str:
    """Extract full text from a PDF file using PyMuPDF."""
    try:
        doc = fitz.open(str(pdf_path))
        text_parts = []

        num_pages = len(doc) if max_pages is None else min(len(doc), max_pages)

        for page_num in range(num_pages):
            page = doc[page_num]
            text_parts.append(page.get_text())

        doc.close()
        return "\n".join(text_parts)
    except Exception as e:
        return f"ERROR extracting PDF: {str(e)}"


def is_mdpi_paper(pdf_filename: str) -> bool:
    """Check if paper is from MDPI based on filename pattern."""
    pdf_lower = pdf_filename.lower()
    for journal in MDPI_JOURNALS:
        if journal in pdf_lower:
            return True
    return False


def find_pdf_file(pdf_filename: str, pdf_dir: Path) -> Path:
    """Find PDF file with fuzzy matching for typos in filenames."""
    import re
    from difflib import SequenceMatcher

    # Try exact match first
    exact_path = pdf_dir / pdf_filename
    if exact_path.exists():
        return exact_path

    # Try with .pdf extension
    with_ext = pdf_dir / f"{pdf_filename}.pdf"
    if with_ext.exists():
        return with_ext

    # Get all PDFs in directory
    all_pdfs = list(pdf_dir.glob("*.pdf"))
    pdf_basename = pdf_filename.replace('.pdf', '').lower()

    # Try fuzzy matching - look for similar filenames
    best_match = None
    best_score = 0.0

    for pdf_path in all_pdfs:
        candidate = pdf_path.stem.lower()

        # Use SequenceMatcher for fuzzy string comparison
        ratio = SequenceMatcher(None, pdf_basename, candidate).ratio()

        # High similarity threshold (0.75+)
        if ratio > best_score and ratio > 0.75:
            best_score = ratio
            best_match = pdf_path

        # Also check pattern-based matching for journal article names
        csv_match = re.match(r'([a-z]+)-?(\d+)-?(\d+)', pdf_basename)
        file_match = re.match(r'([a-z]+)-?(\d+)-?(\d+)', candidate)

        if csv_match and file_match:
            csv_prefix, csv_vol, csv_num = csv_match.groups()
            file_prefix, file_vol, file_num = file_match.groups()

            if csv_vol == file_vol:
                num_ratio = SequenceMatcher(None, csv_num, file_num).ratio()
                if num_ratio > 0.8:
                    combined_score = (ratio + num_ratio) / 2
                    if combined_score > best_score:
                        best_score = combined_score
                        best_match = pdf_path

    return best_match


def extract_year_from_text(text: str) -> int:
    """Try to extract publication year from paper text."""
    import re
    patterns = [
        r'©\s*(\d{4})',
        r'Copyright\s*©?\s*(\d{4})',
        r'Published:?\s*\w+\s+\d{1,2},?\s*(\d{4})',
        r'Received:?\s*\w+\s+\d{1,2},?\s*(\d{4})',
        r'\b(20[12][0-9])\b'
    ]

    for pattern in patterns:
        matches = re.findall(pattern, text[:5000])
        if matches:
            years = [int(y) for y in matches if 2010 <= int(y) <= 2030]
            if years:
                return max(years)
    return None


def create_review_prompt(title: str, abstract: str, full_text: str) -> str:
    """Create the prompt for Gemini to review a paper."""
    truncated_text = full_text[:15000] if len(full_text) > 15000 else full_text

    return f"""You are reviewing papers for a narrative review on Creative AI and Neurosymbolic approaches.

{CREATIVE_AI_CONTEXT}

PAPER TO REVIEW:
Title: {title}

Abstract: {abstract}

Full Text (may be truncated):
{truncated_text}

TASK:
1. Read the paper carefully
2. Determine if it relates to any of our 5 thematic clusters
3. Make a decision: Accept, Reject, or Maybe (lean toward Accept/Maybe when relevant)

RESPOND IN THIS EXACT JSON FORMAT (no markdown, just raw JSON):
{{
    "decision": "Accept" or "Reject" or "Maybe",
    "comment": "<ONE sentence explaining your decision, max 100 characters>",
    "is_editorial": true/false,
    "is_background_material": true/false,
    "estimated_year": <year or null>,
    "primary_theme": "<which of the 5 clusters it fits, or 'None'>"
}}

DECISION GUIDANCE:
- ACCEPT: Relevant to any of the 5 thematic clusters, studies creativity/cognition, proposes frameworks
- MAYBE: Partially relevant, might provide useful background or tangential insights
- REJECT: Completely off-topic (e.g., purely clinical drug trials, unrelated technical methods)

Be inclusive - when in doubt, lean toward Accept or Maybe.
"""


def review_paper_with_gemini(model, title: str, abstract: str, full_text: str, max_retries: int = 3) -> dict:
    """Use Gemini to review a single paper with retry logic for rate limits."""
    prompt = create_review_prompt(title, abstract, full_text)

    for attempt in range(max_retries):
        try:
            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.3,
                    max_output_tokens=1000,
                    response_mime_type="application/json"
                )
            )

            content = response.text.strip()

            # Handle markdown code blocks
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]

            if "```" in content:
                content = content.split("```")[0]

            result = json.loads(content.strip())
            return result

        except json.JSONDecodeError as e:
            print(f"  JSON parse error: {e}")
            decision = "Maybe"
            comment = "JSON parse error"
            if '"decision": "Accept"' in content or '"decision":"Accept"' in content:
                decision = "Accept"
                comment = "Parsed from partial response"
            elif '"decision": "Reject"' in content or '"decision":"Reject"' in content:
                decision = "Reject"
                comment = "Parsed from partial response"
            return {
                "decision": decision,
                "comment": comment,
                "is_editorial": False,
                "estimated_year": None,
                "primary_theme": "None"
            }
        except Exception as e:
            error_str = str(e)
            if "429" in error_str and attempt < max_retries - 1:
                wait_time = 65
                print(f"  Rate limited, waiting {wait_time}s before retry {attempt + 2}/{max_retries}...")
                time.sleep(wait_time)
                continue
            print(f"  API error: {e}")
            return {
                "decision": "Maybe",
                "comment": f"API error: {str(e)[:50]}",
                "is_editorial": False,
                "estimated_year": None,
                "primary_theme": "None"
            }

    return {
        "decision": "Maybe",
        "comment": "Max retries exceeded",
        "is_editorial": False,
        "estimated_year": None,
        "primary_theme": "None"
    }


def main():
    print("=" * 70)
    print("Part 5: Gemini API-Based Review of Creative AI Papers")
    print(f"Model: {MODEL}")
    print("=" * 70)

    # Configure Gemini API
    genai.configure(api_key=GEMINI_API_KEY)

    # Initialize Gemini model
    model = genai.GenerativeModel(MODEL)

    # Load input CSV
    if not INPUT_CSV.exists():
        print(f"ERROR: Input file not found: {INPUT_CSV}")
        return None

    df = pd.read_csv(INPUT_CSV)
    print(f"\nLoaded {len(df)} papers from CSV")

    # Initialize result columns
    df['Decision-gemini'] = ''
    df['comment-gemini'] = ''

    # Track statistics
    stats = {"Accept": 0, "Reject": 0, "Maybe": 0, "MDPI_excluded": 0, "Pre2020_excluded": 0, "PDF_not_found": 0}

    print(f"\nReviewing papers with {MODEL}...")
    print("-" * 70)

    for idx, row in df.iterrows():
        title = str(row['title'])
        abstract = str(row.get('abstract', ''))
        pdf_filename = str(row.get('PDF file name', ''))

        print(f"\n[{idx+1}/{len(df)}] {title[:60]}...")
        print(f"  PDF: {pdf_filename}")

        # Check for MDPI papers first
        if is_mdpi_paper(pdf_filename):
            df.at[idx, 'Decision-gemini'] = 'Reject'
            df.at[idx, 'comment-gemini'] = 'MDPI publication - excluded per criteria'
            stats["MDPI_excluded"] += 1
            stats["Reject"] += 1
            print(f"  -> REJECTED (MDPI publication)")
            continue

        # Find and read PDF
        pdf_path = find_pdf_file(pdf_filename, PDF_DIR)

        if pdf_path is None:
            df.at[idx, 'Decision-gemini'] = 'Maybe'
            df.at[idx, 'comment-gemini'] = 'PDF file not found - needs manual review'
            stats["PDF_not_found"] += 1
            stats["Maybe"] += 1
            print(f"  -> MAYBE (PDF not found)")
            continue

        if pdf_path.stem != pdf_filename.replace('.pdf', ''):
            print(f"  Fuzzy match: {pdf_filename} -> {pdf_path.name}")

        # Extract PDF text
        print(f"  Extracting text from PDF...")
        full_text = extract_pdf_text(pdf_path)

        if full_text.startswith("ERROR"):
            df.at[idx, 'Decision-gemini'] = 'Maybe'
            df.at[idx, 'comment-gemini'] = f'PDF extraction error - needs manual review'
            stats["Maybe"] += 1
            print(f"  -> MAYBE (PDF extraction error)")
            continue

        # Check year from text (pre-2020 auto-exclusion)
        year = extract_year_from_text(full_text)
        if year and year < 2020:
            df.at[idx, 'Decision-gemini'] = 'Reject'
            df.at[idx, 'comment-gemini'] = f'Pre-2020 publication ({year}) - excluded per criteria'
            stats["Pre2020_excluded"] += 1
            stats["Reject"] += 1
            print(f"  -> REJECTED (Pre-2020: {year})")
            continue

        # Send to Gemini for review
        print(f"  Sending to {MODEL} for review...")
        result = review_paper_with_gemini(model, title, abstract, full_text)

        decision = result.get('decision', 'Maybe')
        comment = result.get('comment', '')[:100]

        flags = []
        if result.get('is_editorial', False):
            flags.append('[EDITORIAL]')
        if result.get('is_background_material', False):
            flags.append('[BACKGROUND]')

        if flags:
            flag_str = ' '.join(flags)
            comment = f"{flag_str} {comment}"[:100]

        df.at[idx, 'Decision-gemini'] = decision
        df.at[idx, 'comment-gemini'] = comment
        stats[decision] += 1

        print(f"  -> {decision}: {comment}")

        # Rate limiting - Gemini free tier has rate limits
        time.sleep(4)

    # Save results
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n{'=' * 70}")
    print("REVIEW COMPLETE")
    print(f"{'=' * 70}")
    print(f"Total papers reviewed: {len(df)}")
    print(f"Accept: {stats['Accept']}")
    print(f"Maybe: {stats['Maybe']}")
    print(f"Reject: {stats['Reject']}")
    print(f"  - MDPI excluded: {stats['MDPI_excluded']}")
    print(f"  - Pre-2020 excluded: {stats['Pre2020_excluded']}")
    print(f"  - PDF not found: {stats['PDF_not_found']}")
    print(f"\nResults saved to: {OUTPUT_CSV}")

    return df


if __name__ == "__main__":
    result_df = main()
