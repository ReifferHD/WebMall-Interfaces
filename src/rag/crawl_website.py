import os
import sys
import json
import asyncio
import requests
import io
from xml.etree import ElementTree
from typing import List, Dict, Any, Set
from dataclasses import dataclass
from datetime import datetime, timezone
from urllib.parse import urlparse, urljoin
from dotenv import load_dotenv
from bs4 import BeautifulSoup
import argparse

from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from openai import AsyncOpenAI
from rag.elasticsearch_client import ElasticsearchRAGClient
from unstructured.partition.html import partition_html

load_dotenv()

# Initialize OpenAI and Elasticsearch clients
openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
elasticsearch_client = ElasticsearchRAGClient()


# MODIFIED: Added summary and separate embeddings for V2 Elasticsearch schema
@dataclass
class ProcessedChunk:
    url: str
    chunk_number: int
    title: str
    summary: str  # MODIFIED: Added summary field
    content: str
    metadata: Dict[str, Any]
    title_embedding: List[float]  # MODIFIED: Separate embedding for title
    summary_embedding: List[float]  # MODIFIED: Separate embedding for summary
    content_embedding: List[float]  # MODIFIED: Separate embedding for content


def chunk_text(text: str, chunk_size: int = 5000) -> List[str]:
    """Split text into chunks, respecting code blocks and paragraphs."""
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        # Calculate end position
        end = start + chunk_size

        # If we're at the end of the text, just take what's left
        if end >= text_length:
            chunks.append(text[start:].strip())
            break

        # Try to find a code block boundary first (```)
        chunk = text[start:end]
        code_block = chunk.rfind('```')
        if code_block != -1 and code_block > chunk_size * 0.3:
            end = start + code_block

        # If no code block, try to break at a paragraph
        elif '\n\n' in chunk:
            # Find the last paragraph break
            last_break = chunk.rfind('\n\n')
            if last_break > chunk_size * 0.3:  # Only break if we're past 30% of chunk_size
                end = start + last_break

        # If no paragraph break, try to break at a sentence
        elif '. ' in chunk:
            # Find the last sentence break
            last_period = chunk.rfind('. ')
            if last_period > chunk_size * 0.3:  # Only break if we're past 30% of chunk_size
                end = start + last_period + 1

        # Extract chunk and clean it up
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        # Move start position for next chunk
        start = max(start + 1, end)

    return chunks


# MODIFIED: Changed from get_title to get_title_and_summary to generate both title and summary
async def get_title_and_summary(chunk: str, url: str, content_type: str = "webpage") -> Dict[str, str]:
    """Extract title and summary using GPT-4."""
    system_prompt = f"""You are an AI that extracts titles and summaries from {content_type} content chunks.
    Return a JSON object with 'title' and 'summary' keys.
    For the title: If this seems like the start of a document, extract its title. If it's a middle chunk, derive a descriptive title.
    For the summary: Create a concise summary of the main points in this chunk. Please highlight the price of the product in the summary. As well as any highly relevant product details and technical specifications. Do not include any marketing or promotional language. The title should be as specific as possible. For example, AMD Ryzen 9 5900X - 3.7 GHz - 12 Cores - 24 Threads is great; AMD Ryzen 9 5900X Overview IS BAD.
    Keep both title and summary concise but informative."""

    try:
        response = await openai_client.chat.completions.create(
            model=os.getenv("LLM_MODEL", "gpt-4.1-mini"),
            messages=[
                {"role": "system", "content": system_prompt},
                # Send first 1000 chars for context
                {"role": "user",
                    "content": f"URL: {url}\n\nContent Type: {content_type}\n\nContent:\n{chunk[:1000]}..."}
            ],
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"Error getting title and summary: {e}")
        return {"title": "Error processing title", "summary": "Error processing summary"}


async def get_embedding(text: str) -> List[float]:
    """Get embedding vector from OpenAI."""
    try:
        response = await openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return [0] * 1536  # Return zero vector on error


async def process_chunk(chunk: str, chunk_number: int, url: str, content_type: str = "webpage") -> ProcessedChunk:
    """Process a single chunk of text."""
    # MODIFIED: Generate title and summary together instead of just title
    extracted = await get_title_and_summary(chunk, url, content_type)

    # MODIFIED: Generate 3 separate embeddings for V2 schema (title, summary, content)
    title_embedding, summary_embedding, content_embedding = await asyncio.gather(
        get_embedding(extracted['title']),
        get_embedding(extracted['summary']),
        get_embedding(chunk)
    )

    # Create metadata
    parsed_url = urlparse(url)
    domain = parsed_url.netloc.lower()

    domain = domain.replace("https://", "")
    domain = domain.replace("http://", "")
    domain = domain.replace("www.", "")

    metadata = {
        "source": domain,
        "content_type": content_type,
        "chunk_size": len(chunk),
        "crawled_at": datetime.now(timezone.utc).isoformat(),
        "url_path": parsed_url.path
    }

    return ProcessedChunk(
        url=url,
        chunk_number=chunk_number,
        title=extracted['title'],
        summary=extracted['summary'],  # MODIFIED: Added summary
        content=chunk,
        metadata=metadata,
        title_embedding=title_embedding,  # MODIFIED: Separate embeddings
        summary_embedding=summary_embedding,
        content_embedding=content_embedding
    )


async def insert_chunk(chunk: ProcessedChunk):
    """Insert a processed chunk into Elasticsearch."""

    try:
        # MODIFIED: Send all 3 embeddings + summary to match V2 Elasticsearch schema
        data = {
            "url": chunk.url,
            "chunk_number": chunk.chunk_number,
            "title": chunk.title,
            "summary": chunk.summary,
            "content": chunk.content,
            "metadata": chunk.metadata,
            "title_embedding": chunk.title_embedding,
            "summary_embedding": chunk.summary_embedding,
            "content_embedding": chunk.content_embedding
        }

        result = await elasticsearch_client.insert_chunk(data)
        return result
    except Exception as e:
        print(f"Error inserting chunk: {e}")
        return None


# NEW FUNCTION: Check if a URL is already stored in the database

async def url_already_in_db(url: str) -> bool:
    """Return True if the exact URL already exists in the Elasticsearch index."""
    try:
        # Check if URL exists in Elasticsearch
        return await elasticsearch_client.check_url_exists(url)
    except Exception as e:
        print(f"Error checking if URL exists in DB: {e}")
        # Fail-open (treat as not present) so the pipeline can continue
        return False


async def process_and_store_document(url: str, content: str, content_type: str = "webpage"):
    """Process a document and store its chunks in parallel."""
    if "/cart" in url or "/checkout" in url:
        print(f"⏭️ Skipping URL: {url}")
        return

    # skip add to cart pages
    if "add-to-cart" in url:
        print(f"⏭️ Skipping add to cart URL: {url}")
        return

    # skip images
    if url.endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp')):
        print(f"⏭️ Skipping image URL: {url}")
        return

    # skip videos
    if url.endswith(('.mp4', '.mov', '.avi', '.wmv', '.flv', '.mpeg', '.mpg', '.m4v', '.webm', '.mkv')):
        print(f"⏭️ Skipping video URL: {url}")
        return

    # skip urls that are anchor links
    if "#" in url:
        print(f"⏭️ Skipping anchor link URL: {url}")
        return

    # Only index product pages
    if "/product/" not in url:
        print(f"⏭️ Skipping non-product URL: {url}")
        return

    # Skip processing if we've already stored this exact URL
    if await url_already_in_db(url):
        print(f"Skipping URL already in DB: {url}")
        return

    # For HTML content, use unstructured to extract text
    if content_type == "webpage" and content.strip().startswith('<'):
        # Only use unstructured if content is actual HTML
        try:
            elements = partition_html(text=content)
            unstruct_content = ""
            for element in elements:
                unstruct_content += element.text + "\n"
            content = unstruct_content.strip()
        except Exception as e:
            print(
                f"Error processing with unstructured: {e}, using original content")
            # Fall back to original content if unstructured fails

    # Split into chunks
    chunks = chunk_text(content, chunk_size=1_000_000)

    # Process chunks in parallel
    tasks = [
        process_chunk(chunk, i, url, content_type)
        for i, chunk in enumerate(chunks)
    ]
    processed_chunks = await asyncio.gather(*tasks)

    # Store chunks in parallel
    insert_tasks = [
        insert_chunk(chunk)
        for chunk in processed_chunks
    ]
    await asyncio.gather(*insert_tasks)


async def crawl_website(urls: List[str], max_concurrent: int = 5, reset_index: bool = False):
    """Crawl multiple URLs in parallel and process both page content."""

    # Reset index if requested
    if reset_index:
        print("Resetting Elasticsearch index...")
        await elasticsearch_client.reset_index()

    browser_config = BrowserConfig(
        headless=True,
        verbose=False,
        ignore_https_errors=True,
        extra_args=["--disable-gpu",
                    "--disable-dev-shm-usage", "--no-sandbox"],
    )
    crawl_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS)

    # Create the crawler instance
    crawler = AsyncWebCrawler(config=browser_config)
    await crawler.start()

    try:
        # Create a semaphore to limit concurrency
        semaphore = asyncio.Semaphore(max_concurrent)

        async def process_url(url: str):
            async with semaphore:
                try:
                    # MODIFIED: Removed shared session_id to avoid deadlocks with concurrent crawling
                    result = await crawler.arun(
                        url=url,
                        config=crawl_config
                    )

                    if result.success:
                        print(f"Successfully crawled: {url}")

                        # Process the page content
                        await process_and_store_document(url, result.html, "webpage")
                    else:
                        print(
                            f"Failed to crawl: {url} - Error: {result.error_message}")

                except Exception as e:
                    print(f"Error processing {url}: {e}")

        # Process all URLs in parallel with limited concurrency
        await asyncio.gather(*[process_url(url) for url in urls])

    finally:
        await crawler.close()
        await elasticsearch_client.close()


def get_urls_from_sitemap(sitemap_url: str) -> List[str]:
    """Extract URLs from a sitemap XML."""
    try:
        response = requests.get(sitemap_url, timeout=30)
        response.raise_for_status()

        # Parse the XML
        root = ElementTree.fromstring(response.content)

        # Extract all URLs from the sitemap
        namespace = {'ns': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
        urls = [loc.text for loc in root.findall('.//ns:loc', namespace)]

        return urls
    except Exception as e:
        print(f"Error fetching sitemap: {e}")
        return []


def get_urls_from_rss(rss_url: str) -> List[str]:
    """Extract product URLs from an RSS feed."""
    try:
        response = requests.get(rss_url, timeout=30)
        response.raise_for_status()

        # Parse the XML
        root = ElementTree.fromstring(response.content)

        # Extract all g:link elements from RSS items
        namespace = {'g': 'http://base.google.com/ns/1.0'}
        urls = [link.text for link in root.findall('.//g:link', namespace)]

        return urls
    except Exception as e:
        print(f"Error fetching RSS feed: {e}")
        return []


def get_urls_from_rss_list(json_file: str) -> List[str]:
    """Extract URLs from multiple RSS feeds listed in a JSON file."""
    all_urls = []
    try:
        with open(json_file, 'r') as f:
            rss_feeds = json.load(f)

        for rss_url in rss_feeds:
            print(f"Processing RSS feed: {rss_url}")
            urls = get_urls_from_rss(rss_url)
            all_urls.extend(urls)
            print(f"Found {len(urls)} URLs from {rss_url}")

        return all_urls
    except Exception as e:
        print(f"Error processing RSS list file: {e}")
        return []


def discover_urls_from_page(base_url: str, max_depth: int = 2, allow_domain_hopping: bool = False) -> List[str]:
    """Discover URLs from a base page by following links.

    Args:
        base_url: The starting URL.
        max_depth: How deep to traverse links.
        allow_domain_hopping: If True, follow links to external domains; otherwise restrict to the base domain.
    """
    discovered_urls = set()
    to_process = [(base_url, 0)]
    processed = set()

    # Limit to prevent infinite crawling
    while to_process and len(discovered_urls) < 5_000:
        current_url, depth = to_process.pop(0)

        if current_url in processed or depth > max_depth:
            continue

        processed.add(current_url)

        try:
            response = requests.get(current_url, timeout=10)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, 'html.parser')
            discovered_urls.add(current_url)

            # Find more links if we haven't reached max depth
            if depth < max_depth:
                base_domain = urlparse(base_url).netloc
                for link in soup.find_all('a', href=True):
                    href = link['href']
                    full_url = urljoin(current_url, href)

                    # Only follow links on the same domain
                    if allow_domain_hopping or urlparse(full_url).netloc == base_domain:
                        to_process.append((full_url, depth + 1))

        except Exception as e:
            print(f"Error discovering URLs from {current_url}: {e}")

    return list(discovered_urls)


async def main():
    parser = argparse.ArgumentParser(
        description='Crawl websites and index content including PDFs')
    parser.add_argument('url', nargs='?', help='Starting URL to crawl')
    parser.add_argument('--sitemap', action='store_true',
                        help='Treat the URL as a sitemap')
    parser.add_argument('--discover', action='store_true',
                        help='Discover URLs by following links')
    parser.add_argument('--rss', action='store_true',
                        help='Treat the URL as an RSS feed')
    parser.add_argument('--rss-list', type=str,
                        help='JSON file containing list of RSS feed URLs')
    parser.add_argument('--reset-index', action='store_true',
                        help='Reset Elasticsearch index before crawling')
    parser.add_argument('--max-depth', type=int, default=2,
                        help='Maximum depth for URL discovery (default: 2)')
    parser.add_argument('--max-concurrent', type=int, default=5,
                        help='Maximum concurrent requests (default: 5)')
    parser.add_argument('--allow-domain-hopping', action='store_true', default=False,
                        help='Allow crawler to follow links to external domains during discovery (default: disabled)')

    args = parser.parse_args()

    urls = []

    if args.rss_list:
        print(f"Processing RSS feeds from file: {args.rss_list}")
        urls = get_urls_from_rss_list(args.rss_list)
    elif args.sitemap:
        print(f"Fetching URLs from sitemap: {args.url}")
        urls = get_urls_from_sitemap(args.url)
    elif args.rss:
        print(f"Fetching URLs from RSS feed: {args.url}")
        urls = get_urls_from_rss(args.url)
    elif args.discover:
        print(f"Discovering URLs from: {args.url}")
        urls = discover_urls_from_page(
            args.url, args.max_depth, args.allow_domain_hopping)
    elif args.url:
        # Single URL
        urls = [args.url]
    else:
        print("No URL or RSS list provided")
        parser.print_help()
        return

    if not urls:
        print("No URLs found to crawl")
        return

    print(f"Found {len(urls)} URL(s) to crawl")
    print("Starting crawl process...")

    await crawl_website(urls, args.max_concurrent, args.reset_index)

    print("Crawling completed!")


if __name__ == "__main__":
    asyncio.run(main())
