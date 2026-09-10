"""
Google Scholar Scraper using Selenium
Scrapes paper information from Google Scholar and saves to CSV
Attempts to fetch full abstracts from paper URLs
"""

import csv
import time
import random
from typing import List, Dict
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from webdriver_manager.chrome import ChromeDriverManager
import re
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse


def setup_driver():
    """
    Set up Chrome driver with options to avoid detection.

    Returns:
        WebDriver instance
    """
    chrome_options = Options()

    # Add arguments to make it look like a real browser
    chrome_options.add_argument('--disable-blink-features=AutomationControlled')
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    chrome_options.add_argument('--start-maximized')
    chrome_options.add_argument('--disable-gpu')

    # Add user agent
    chrome_options.add_argument('user-agent=Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')

    # Exclude automation flags
    chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
    chrome_options.add_experimental_option('useAutomationExtension', False)

    # Uncomment for headless mode (no browser window)
    # chrome_options.add_argument('--headless')

    # Use webdriver-manager to automatically handle driver installation
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)

    # Override navigator.webdriver flag
    driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")

    return driver


def extract_publisher_from_url(url: str) -> str:
    """
    Extract publisher/source from URL.

    Args:
        url: URL string

    Returns:
        Publisher name
    """
    if url == 'N/A' or not url:
        return 'N/A'

    # Common publisher domains
    publishers = {
        'arxiv.org': 'arXiv',
        'ieee.org': 'IEEE',
        'acm.org': 'ACM',
        'springer.com': 'Springer',
        'sciencedirect.com': 'Elsevier/ScienceDirect',
        'nature.com': 'Nature',
        'science.org': 'Science',
        'wiley.com': 'Wiley',
        'tandfonline.com': 'Taylor & Francis',
        'cambridge.org': 'Cambridge University Press',
        'oxford.com': 'Oxford University Press',
        'mit.edu': 'MIT Press',
        'pnas.org': 'PNAS',
        'nih.gov': 'NIH/PubMed',
        'ncbi.nlm.nih.gov': 'PubMed Central',
        'researchgate.net': 'ResearchGate',
        'academia.edu': 'Academia.edu',
        'semanticscholar.org': 'Semantic Scholar',
        'openreview.net': 'OpenReview',
        'jmlr.org': 'JMLR',
        'aaai.org': 'AAAI',
        'neurips.cc': 'NeurIPS',
        'proceedings.mlr.press': 'PMLR',
        'aclweb.org': 'ACL Anthology',
    }

    url_lower = url.lower()
    for domain, publisher in publishers.items():
        if domain in url_lower:
            return publisher

    # Extract domain as fallback
    try:
        from urllib.parse import urlparse
        parsed = urlparse(url)
        domain = parsed.netloc
        # Remove www. prefix
        domain = domain.replace('www.', '')
        return domain if domain else 'Unknown'
    except:
        return 'Unknown'


def fetch_full_abstract_from_url(url: str, snippet: str) -> str:
    """
    Attempt to fetch full abstract from the paper's URL.
    Uses requests + BeautifulSoup for fast scraping.

    Args:
        url: Paper URL
        snippet: Fallback snippet if fetching fails

    Returns:
        Full abstract or snippet
    """
    if not url or url == 'N/A' or len(snippet) > 300:
        return snippet

    try:
        # Common abstract selectors for different publishers
        abstract_selectors = {
            'arxiv.org': ['blockquote.abstract', 'meta[name="citation_abstract"]'],
            'aclanthology.org': ['div.acl-abstract', 'div.card-body', 'meta[name="citation_abstract"]'],
            'aclweb.org': ['div.acl-abstract', 'div.card-body', 'meta[name="citation_abstract"]'],
            'nature.com': ['div#Abs1-content', 'div.c-article-section__content', 'meta[name="dc.description"]'],
            'science.org': ['div.abstract', 'div.section.abstract', 'meta[name="description"]'],
            'ieee': ['div.abstract-text', 'meta[property="og:description"]'],
            'springer': ['div#Abs1-content', 'section.Abstract', 'meta[name="dc.description"]'],
            'elsevier': ['div.abstract', 'div#abstracts', 'meta[name="description"]'],
            'sciencedirect': ['div.abstract', 'div#abstracts', 'meta[name="description"]'],
            'frontiersin.org': ['div.JournalAbstract', 'meta[name="description"]'],
            'pnas.org': ['div.abstract', 'meta[name="description"]'],
            'neurips.cc': ['div.abstract', 'meta[name="description"]'],
            'proceedings.mlr.press': ['div.abstract', 'div#abstract'],
            'pmlr': ['div.abstract', 'div#abstract'],
        }

        # Determine selectors from URL
        url_lower = url.lower()
        selectors = ['meta[name="citation_abstract"]', 'meta[name="description"]', 'div.abstract']

        for publisher, pub_selectors in abstract_selectors.items():
            if publisher in url_lower:
                selectors = pub_selectors + selectors
                break

        # Fetch with requests
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)

        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')

            # Try each selector
            for selector in selectors:
                try:
                    if selector.startswith('meta'):
                        elem = soup.select_one(selector)
                        if elem:
                            abstract = elem.get('content', '').strip()
                            if abstract and len(abstract) > 100:
                                # Clean up
                                abstract = re.sub(r'\s+', ' ', abstract)
                                return abstract
                    else:
                        elem = soup.select_one(selector)
                        if elem:
                            abstract = elem.get_text().strip()
                            # Clean up
                            abstract = re.sub(r'^abstract[:\s]*', '', abstract, flags=re.IGNORECASE)
                            abstract = re.sub(r'\s+', ' ', abstract)
                            if len(abstract) > 100:
                                return abstract
                except:
                    continue

    except:
        pass

    return snippet


def extract_paper_info_selenium(driver, article_element) -> Dict:
    """
    Extract paper information from a Selenium WebElement.
    Attempts to get full abstract when available.

    Args:
        driver: Selenium WebDriver instance
        article_element: WebElement containing paper information

    Returns:
        Dictionary with paper information
    """
    try:
        # Extract title and URL
        try:
            title_elem = article_element.find_element(By.CSS_SELECTOR, 'h3.gs_rt')
            title = title_elem.text.strip()

            # Try to get the URL from the link
            try:
                link_elem = title_elem.find_element(By.TAG_NAME, 'a')
                url = link_elem.get_attribute('href')
            except NoSuchElementException:
                url = 'N/A'
        except NoSuchElementException:
            title = 'N/A'
            url = 'N/A'

        # Extract publisher from URL
        publisher = extract_publisher_from_url(url)

        # Extract authors, year, and publisher info
        try:
            authors_elem = article_element.find_element(By.CSS_SELECTOR, 'div.gs_a')
            authors_text = authors_elem.text
            parts = authors_text.split(' - ')

            authors = parts[0].strip() if len(parts) > 0 else 'N/A'

            # Extract year
            year = 'N/A'
            for part in parts:
                year_match = re.search(r'\b(19|20)\d{2}\b', part)
                if year_match:
                    year = year_match.group(0)
                    break

            # Try to get publisher from metadata if not found in URL
            if publisher == 'Unknown' and len(parts) > 1:
                # Often the publisher is in the second part
                pub_part = parts[1].strip()
                # Remove year if present
                pub_part = re.sub(r'\b(19|20)\d{2}\b', '', pub_part).strip()
                if pub_part and len(pub_part) > 0:
                    publisher = pub_part

        except NoSuchElementException:
            authors = 'N/A'
            year = 'N/A'

        # Extract abstract/snippet from Google Scholar
        snippet = 'N/A'
        try:
            abstract_elem = article_element.find_element(By.CSS_SELECTOR, 'div.gs_rs')
            snippet = abstract_elem.text.strip()
        except NoSuchElementException:
            pass

        # Try to fetch full abstract from the paper's URL
        abstract = fetch_full_abstract_from_url(url, snippet)

        # Print status
        if len(abstract) > len(snippet) and abstract != snippet:
            print(f"      ✓ Fetched full abstract ({len(abstract)} chars)")
        else:
            print(f"      → Using snippet ({len(snippet)} chars)")

        return {
            'title': title,
            'authors': authors,
            'year': year,
            'publisher': publisher,
            'abstract': abstract,
            'url': url
        }
    except Exception as e:
        print(f"    Error extracting paper info: {e}")
        return None


def search_google_scholar_selenium(query: str, max_results: int = 50, year_from: int = 2015) -> List[Dict]:
    """
    Search Google Scholar using Selenium and extract paper information.

    Args:
        query: Search query string
        max_results: Maximum number of results to return (default 50)
        year_from: Filter papers from this year onwards (default 2015)

    Returns:
        List of dictionaries containing paper information
    """
    papers = []
    driver = None

    print(f"Searching Google Scholar for: '{query}'")
    print(f"Year filter: {year_from} onwards")
    print(f"Fetching up to {max_results} results...")
    print("Opening browser...\n")

    try:
        driver = setup_driver()
        base_url = "https://scholar.google.com/scholar"
        results_per_page = 10

        num_pages = min((max_results + results_per_page - 1) // results_per_page, 10)

        for page in range(num_pages):
            if len(papers) >= max_results:
                break

            start = page * results_per_page
            # Add year filter to URL
            url = f"{base_url}?q={query}&start={start}&hl=en&as_ylo={year_from}"

            print(f"Fetching page {page + 1}/{num_pages}...")

            try:
                driver.get(url)

                # Random human-like delay
                time.sleep(random.uniform(3, 6))

                # Wait for results to load
                try:
                    WebDriverWait(driver, 10).until(
                        EC.presence_of_element_located((By.CSS_SELECTOR, 'div.gs_ri'))
                    )
                except TimeoutException:
                    print(f"  ⚠ Timeout waiting for results on page {page + 1}")

                    # Check if CAPTCHA or blocked
                    if "unusual traffic" in driver.page_source.lower() or "captcha" in driver.page_source.lower():
                        print("\n  ⚠ Google Scholar detected automated access!")
                        print("  The browser window will stay open so you can:")
                        print("    1. Solve the CAPTCHA manually")
                        print("    2. Then press Enter here to continue...")
                        input()
                        continue
                    else:
                        break

                # Find all paper articles
                articles = driver.find_elements(By.CSS_SELECTOR, 'div.gs_ri')

                if not articles:
                    print(f"  ⚠ No articles found on page {page + 1}")
                    break

                for article in articles:
                    if len(papers) >= max_results:
                        break

                    paper_info = extract_paper_info_selenium(driver, article)
                    if paper_info and paper_info['title'] != 'N/A':
                        papers.append(paper_info)
                        print(f"  [{len(papers)}] {paper_info['title'][:80]}... ({paper_info['year']})")

                # Scroll down to mimic human behavior
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(random.uniform(1, 2))

            except Exception as e:
                print(f"  ⚠ Error on page {page + 1}: {e}")
                continue

        print(f"\nSuccessfully collected {len(papers)} papers")

    except Exception as e:
        print(f"Error during search: {e}")

    finally:
        if driver:
            print("\nClosing browser...")
            driver.quit()

    return papers


def save_to_csv(papers: List[Dict], filename: str = "scholar_results.csv"):
    """
    Save papers to CSV file.

    Args:
        papers: List of paper dictionaries
        filename: Output CSV filename
    """
    if not papers:
        print("No papers to save!")
        return

    # Create DataFrame
    df = pd.DataFrame(papers)

    # Reorder columns
    column_order = ['title', 'authors', 'year', 'publisher', 'abstract', 'url']
    df = df[column_order]

    # Save to CSV
    df.to_csv(filename, index=False, encoding='utf-8', quoting=csv.QUOTE_ALL)
    print(f"\nResults saved to: {filename}")
    print(f"Total papers saved: {len(papers)}")


def load_queries_from_file(filepath: str) -> List[tuple]:
    """
    Load queries from a Python file containing a queries list.

    Args:
        filepath: Path to the queries file

    Returns:
        List of (query, year) tuples
    """
    queries = []

    try:
        # Read the file and extract queries
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


def process_multiple_queries(queries_file: str = "queries.txt", output_dir: str = "queries", max_results: int = 50):
    """
    Process multiple queries from a file and save each to a separate CSV.

    Args:
        queries_file: Path to file containing queries
        output_dir: Directory to save CSV files
        max_results: Maximum results per query
    """
    import os

    # Load queries
    queries = load_queries_from_file(queries_file)

    if not queries:
        print("No queries to process!")
        return

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}\n")

    # Process each query
    total_queries = len(queries)
    driver = None

    try:
        # Initialize driver once for all queries
        print("Initializing browser...\n")
        driver = setup_driver()

        for idx, (query, year_from) in enumerate(queries, 1):
            print("=" * 70)
            print(f"Processing query {idx}/{total_queries}")
            print("=" * 70)
            print(f"Query: {query}")
            print(f"Year filter: {year_from} onwards")
            print(f"Max results: {max_results}\n")

            try:
                # Search for papers
                papers = search_with_existing_driver(driver, query, max_results, year_from)

                # Save to CSV
                if papers:
                    # Create safe filename
                    safe_query = "".join(c if c.isalnum() or c == ' ' else "_" for c in query)
                    safe_query = "_".join(safe_query.split())[:50]
                    filename = os.path.join(output_dir, f"{safe_query}.csv")
                    save_to_csv(papers, filename)
                else:
                    print(f"⚠ No papers found for query: {query}\n")

                # Add delay between queries to be polite
                if idx < total_queries:
                    print(f"\nWaiting 10 seconds before next query...\n")
                    time.sleep(10)

            except Exception as e:
                print(f"⚠ Error processing query '{query}': {e}\n")
                continue

        print("\n" + "=" * 70)
        print("All queries processed!")
        print("=" * 70)

    finally:
        if driver:
            print("\nClosing browser...")
            driver.quit()


def search_with_existing_driver(driver, query: str, max_results: int = 50, year_from: int = 2015) -> List[Dict]:
    """
    Search Google Scholar using an existing Selenium driver.

    Args:
        driver: Existing Selenium WebDriver instance
        query: Search query string
        max_results: Maximum number of results to return
        year_from: Filter papers from this year onwards

    Returns:
        List of dictionaries containing paper information
    """
    papers = []
    base_url = "https://scholar.google.com/scholar"
    results_per_page = 10

    num_pages = min((max_results + results_per_page - 1) // results_per_page, 10)

    for page in range(num_pages):
        if len(papers) >= max_results:
            break

        start = page * results_per_page
        url = f"{base_url}?q={query}&start={start}&hl=en&as_ylo={year_from}"

        print(f"Fetching page {page + 1}/{num_pages}...")

        try:
            driver.get(url)

            # Random human-like delay
            time.sleep(random.uniform(3, 6))

            # Wait for results to load
            try:
                WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, 'div.gs_ri'))
                )
            except TimeoutException:
                print(f"  ⚠ Timeout waiting for results on page {page + 1}")

                # Check if CAPTCHA or blocked
                if "unusual traffic" in driver.page_source.lower() or "captcha" in driver.page_source.lower():
                    print("\n  ⚠ Google Scholar detected automated access!")
                    print("  The browser window will stay open so you can:")
                    print("    1. Solve the CAPTCHA manually")
                    print("    2. Then press Enter here to continue...")
                    input()
                    continue
                else:
                    break

            # Find all paper articles
            articles = driver.find_elements(By.CSS_SELECTOR, 'div.gs_ri')

            if not articles:
                print(f"  ⚠ No articles found on page {page + 1}")
                break

            for article in articles:
                if len(papers) >= max_results:
                    break

                paper_info = extract_paper_info_selenium(driver, article)
                if paper_info and paper_info['title'] != 'N/A':
                    papers.append(paper_info)
                    print(f"  [{len(papers)}] {paper_info['title'][:80]}... ({paper_info['year']})")

            # Scroll down to mimic human behavior
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(random.uniform(1, 2))

        except Exception as e:
            print(f"  ⚠ Error on page {page + 1}: {e}")
            continue

    print(f"Collected {len(papers)} papers\n")
    return papers


def main():
    """Main function to run the scraper."""
    import os

    print("=" * 70)
    print("Google Scholar Scraper (Selenium)")
    print("=" * 70)
    print("\nNote: This will open a Chrome browser window.")
    print("If CAPTCHA appears, solve it and press Enter to continue.\n")

    # Check if queries.txt exists
    queries_file = "queries.txt"

    if os.path.exists(queries_file):
        print(f"Found {queries_file}")
        print("Processing all queries from queries.txt with default settings (50 results per query)")

        # Auto-process with defaults
        max_results = 50
        process_multiple_queries(queries_file, "queries", max_results)
        return

    # Single query mode (original functionality)
    query = input("Enter your search query: ").strip()

    if not query:
        print("Error: Search query cannot be empty!")
        return

    # Optional: customize year filter
    year_input = input("Filter papers from year (default 2015, press Enter to use default): ").strip()
    year_from = int(year_input) if year_input.isdigit() and 1900 <= int(year_input) <= 2100 else 2015

    # Optional: customize max results
    max_results_input = input("Maximum results (default 50, press Enter to use default): ").strip()
    max_results = int(max_results_input) if max_results_input.isdigit() else 50

    # Perform search
    papers = search_google_scholar_selenium(query, max_results, year_from)

    # Save to CSV
    if papers:
        # Create filename based on query
        safe_query = "".join(c if c.isalnum() else "_" for c in query)[:30]
        filename = f"scholar_{safe_query}.csv"
        save_to_csv(papers, filename)
    else:
        print("\nNo papers found or error occurred during scraping.")


if __name__ == "__main__":
    main()
