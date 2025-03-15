import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import requests
from io import StringIO
import random
from typing import List, Dict, Any, Tuple
import re
import json
import os
from datetime import datetime, timedelta

# Set page configuration
st.set_page_config(
    page_title="German Startup Funding Analyzer",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #0D47A1;
    }
    .insight-box {
        background-color: #E3F2FD;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #1E88E5;
    }
    /* Key Metrics Cards Styling */
    .metric-card {
        background-color: #ffffff !important;
        border: 1px solid #e0e0e0 !important;
        border-radius: 10px !important;
        padding: 20px 15px !important;
        margin: 5px !important;
        min-width: 200px !important;
        height: 120px !important;
        display: flex !important;
        flex-direction: column !important;
        justify-content: center !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important;
    }
    .metric-card h3 {
        color: #1E88E5 !important;  /* Blue color for titles */
        font-size: 1.1rem !important;
        margin-bottom: 8px !important;
    }
    .metric-card p {
        color: #2d3436 !important;  /* Dark gray for values */
        font-size: 24px !important;
        font-weight: 700 !important;
        margin: 0 !important;
    }
    /* Equal column widths */
    [data-testid="column"] {
        flex: 1 1 0 !important;
        min-width: 200px !important;
        max-width: 25% !important;
    }
</style>
""", unsafe_allow_html=True)

# Function to fetch German startup data from Groq API
def fetch_from_groq(start_date, end_date, api_key):
    """
    Uses Groq API to generate German startup funding data
    
    Parameters:
    start_date (datetime): Start date for data collection
    end_date (datetime): End date for data collection
    api_key (str): Groq API key
    
    Returns:
    pd.DataFrame: German startup funding data
    """
    if not api_key:
        st.error("Groq API key is required")
        return None
        
    try:
        st.info("Connecting to Groq API to generate German startup funding data...")
        
        # Format dates for API
        start_date_str = start_date.strftime("%Y-%m-%d")
        end_date_str = end_date.strftime("%Y-%m-%d")
        
        # Define API endpoint
        base_url = "https://api.groq.com/openai/v1/chat/completions"
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        # Prepare the prompt for Groq
        system_prompt = """You are a German startup funding data expert. Generate realistic funding data for German startups between the specified dates. 
        Include realistic startup names, industries, cities, funding rounds, funding amounts, dates, and investors. 
        Make the data as realistic as possible based on the German startup ecosystem."""
        
        user_prompt = f"""Generate realistic German startup funding data between {start_date_str} and {end_date_str}.
        Return the data as a JSON array of objects, with each object having these fields:
        - startup_name: Name of a real German startup
        - industry: Industry sector (Fintech, AI/ML, Healthtech, etc.)
        - city: German city where the startup is based
        - funding_round: Type of funding round (Seed, Series A, B, C, etc.)
        - funding_amount_millions: Amount raised in millions of euros (realistic for the stage)
        - funding_date: A date between {start_date_str} and {end_date_str} in YYYY-MM-DD format
        - investors: Comma-separated list of real investors active in Germany
        - valuation_millions: Post-money valuation in millions of euros (or 0 if unknown)
        
        Generate data for approximately 200 different funding events. It's okay to have multiple funding rounds for the same startup as long as they're realistically spaced in time. Make sure to:
        1. Use real German startup names like N26, Celonis, GetYourGuide, etc.
        2. Match startups with their likely industries (e.g., N26 is Fintech)
        3. Match startups with their actual locations (e.g., N26 is in Berlin)
        4. Use realistic funding amounts for each stage
        5. Include real investors active in Germany
        
        Return ONLY the JSON array with no other text."""
        
        payload = {
            "model": "llama3-70b-8192",  # Using Groq's Llama 3 70B model
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.3,  # Lower temperature for more factual responses
            "max_tokens": 8000
        }
        
        # Make API request
        st.text("Sending request to Groq API...")
        response = requests.post(base_url, json=payload, headers=headers)
        
        if response.status_code != 200:
            st.error(f"API error: {response.status_code} - {response.text}")
            return None
        
        # Parse response
        data = response.json()
        
        # Extract the JSON content from the response
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        
        # Find and extract the JSON array from the content
        # This handles cases where the model includes explanatory text
        json_start = content.find("[")
        json_end = content.rfind("]") + 1
        
        if json_start == -1 or json_end == 0:
            st.error("Could not find valid JSON data in response")
            return None
            
        json_str = content[json_start:json_end]
        
        try:
            funding_data = json.loads(json_str)
        except json.JSONDecodeError:
            st.error("Failed to parse JSON data from response")
            return None
        
        # Process the data
        processed_data = []
        
        for entry in funding_data:
            # Ensure funding_date is in the correct format
            try:
                funding_date = datetime.strptime(entry.get("funding_date", ""), "%Y-%m-%d")
            except ValueError:
                # If date parsing fails, generate a random date in the range
                days_range = (end_date - start_date).days
                random_days = random.randint(0, max(1, days_range))
                funding_date = start_date + timedelta(days=random_days)
            
            # Add computed fields
            processed_entry = {
                'startup_name': entry.get('startup_name', ''),
                'industry': entry.get('industry', ''),
                'country': 'Germany',
                'city': entry.get('city', ''),
                'funding_round': entry.get('funding_round', ''),
                'funding_amount_millions': float(entry.get('funding_amount_millions', 0)),
                'funding_date': funding_date,
                'investors': entry.get('investors', ''),
                'valuation_millions': float(entry.get('valuation_millions', 0)),
                'year': funding_date.year,
                'month': funding_date.month,
                'quarter': (funding_date.month - 1) // 3 + 1
            }
            
            processed_data.append(processed_entry)
        
        # If we got fewer than expected entries, make a second request to get more
        if len(processed_data) < 100:
            st.info(f"Only received {len(processed_data)} entries, requesting more...")
            
            # Make a second request with a different prompt
            payload["messages"][1]["content"] = user_prompt + "\nPlease generate different startups than in your previous response."
            response = requests.post(base_url, json=payload, headers=headers)
            
            if response.status_code == 200:
                data = response.json()
                content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                json_start = content.find("[")
                json_end = content.rfind("]") + 1
                
                if json_start != -1 and json_end != 0:
                    json_str = content[json_start:json_end]
                    try:
                        more_funding_data = json.loads(json_str)
                        
                        for entry in more_funding_data:
                            try:
                                funding_date = datetime.strptime(entry.get("funding_date", ""), "%Y-%m-%d")
                            except ValueError:
                                days_range = (end_date - start_date).days
                                random_days = random.randint(0, max(1, days_range))
                                funding_date = start_date + timedelta(days=random_days)
                            
                            processed_entry = {
                                'startup_name': entry.get('startup_name', ''),
                                'industry': entry.get('industry', ''),
                                'country': 'Germany',
                                'city': entry.get('city', ''),
                                'funding_round': entry.get('funding_round', ''),
                                'funding_amount_millions': float(entry.get('funding_amount_millions', 0)),
                                'funding_date': funding_date,
                                'investors': entry.get('investors', ''),
                                'valuation_millions': float(entry.get('valuation_millions', 0)),
                                'year': funding_date.year,
                                'month': funding_date.month,
                                'quarter': (funding_date.month - 1) // 3 + 1
                            }
                            
                            processed_data.append(processed_entry)
                    except json.JSONDecodeError:
                        st.warning("Could not parse additional data")
        
        # Convert to DataFrame
        df = pd.DataFrame(processed_data)
        st.success(f"Successfully generated {len(df)} funding rounds from Groq API.")
        return df
    
    except Exception as e:
        st.error(f"Error fetching data from Groq API: {str(e)}")
        st.exception(e)
        return None

# Enhanced web scraping function for real German startup funding data
def scrape_startup_funding_news():
    """
    Scrapes real German startup funding news from multiple tech news sites
    using BeautifulSoup for proper HTML parsing
    """
    try:
        st.info("Scraping real German startup funding data from multiple sources...")
        
        # Import BeautifulSoup for proper HTML parsing
        from bs4 import BeautifulSoup
        import re
        import time
        
        # List of sources to check with their specific parsing strategies
        sources = [
            {
                "url": "https://www.eu-startups.com/category/germany-startups/",
                "name": "EU-Startups",
                "pages": 5  # Scrape up to 5 pages
            },
            {
                "url": "https://sifted.eu/articles/tag/germany",
                "name": "Sifted",
                "pages": 3
            },
            {
                "url": "https://techcrunch.com/tag/germany/",
                "name": "TechCrunch",
                "pages": 3
            },
            {
                "url": "https://tech.eu/category/germany/",
                "name": "Tech.eu",
                "pages": 3
            }
        ]
        
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        
        all_articles = []
        
        # Step 1: Get all articles from all sources
        for source in sources:
            st.text(f"Collecting articles from {source['name']}...")
            base_url = source['url']
            
            for page in range(1, source['pages'] + 1):
                try:
                    # Handle pagination differently based on the source
                    if page > 1:
                        if "eu-startups" in base_url:
                            url = f"{base_url}page/{page}/"
                        elif "sifted" in base_url:
                            url = f"{base_url}/page/{page}"
                        elif "techcrunch" in base_url:
                            url = f"{base_url}/page/{page}/"
                        elif "tech.eu" in base_url:
                            url = f"{base_url}/page/{page}"
                        else:
                            url = base_url
                    else:
                        url = base_url
                    
                    st.text(f"  Scraping page {page}: {url}")
                    response = requests.get(url, headers=headers, timeout=15)
                    
                    if response.status_code != 200:
                        st.warning(f"Failed to fetch page {page} from {source['name']}: {response.status_code}")
                        continue
                    
                    soup = BeautifulSoup(response.content, 'html.parser')
                    
                    # Different parsing for different sources
                    if "eu-startups" in url:
                        articles = soup.find_all('article')
                        for article in articles:
                            title_elem = article.find('h2', class_='entry-title')
                            if not title_elem:
                                continue
                                
                            title = title_elem.text.strip()
                            link = title_elem.find('a')['href'] if title_elem.find('a') else None
                            date_elem = article.find('time', class_='entry-date')
                            date_str = date_elem['datetime'].split('T')[0] if date_elem else None
                            
                            if title and link and date_str:
                                all_articles.append({
                                    'title': title,
                                    'url': link,
                                    'date': date_str,
                                    'source': source['name']
                                })
                    
                    elif "sifted" in url:
                        articles = soup.find_all('div', class_='articleCard')
                        for article in articles:
                            title_elem = article.find('h3')
                            if not title_elem:
                                continue
                                
                            title = title_elem.text.strip()
                            link_elem = article.find('a')
                            link = f"https://sifted.eu{link_elem['href']}" if link_elem else None
                            date_elem = article.find('span', class_='date')
                            date_str = parse_date_text(date_elem.text.strip()) if date_elem else None
                            
                            if title and link:
                                all_articles.append({
                                    'title': title,
                                    'url': link,
                                    'date': date_str,
                                    'source': source['name']
                                })
                    
                    elif "techcrunch" in url:
                        articles = soup.find_all('div', class_='post-block')
                        for article in articles:
                            title_elem = article.find('h2')
                            if not title_elem:
                                continue
                                
                            title = title_elem.text.strip()
                            link_elem = article.find('a', class_='post-block__title__link')
                            link = link_elem['href'] if link_elem else None
                            date_elem = article.find('time')
                            date_str = date_elem['datetime'].split('T')[0] if date_elem else None
                            
                            if title and link and date_str:
                                all_articles.append({
                                    'title': title,
                                    'url': link,
                                    'date': date_str,
                                    'source': source['name']
                                })
                    
                    elif "tech.eu" in url:
                        articles = soup.find_all('article')
                        for article in articles:
                            title_elem = article.find('h3', class_='entry-title')
                            if not title_elem:
                                continue
                                
                            title = title_elem.text.strip()
                            link = title_elem.find('a')['href'] if title_elem.find('a') else None
                            date_elem = article.find('time', class_='entry-date')
                            date_str = date_elem['datetime'].split('T')[0] if date_elem else None
                            
                            if title and link and date_str:
                                all_articles.append({
                                    'title': title,
                                    'url': link,
                                    'date': date_str,
                                    'source': source['name']
                                })
                    
                    # Respect websites by adding a small delay between requests
                    time.sleep(1)
                    
                except Exception as e:
                    st.warning(f"Error scraping page {page} from {source['name']}: {str(e)}")
                    continue
        
        st.text(f"Collected {len(all_articles)} articles from all sources")
        
        # Step 2: Filter for funding-related articles
        funding_keywords = [
            'raises', 'raised', 'secures', 'secured', 'funding', 'investment',
            'series', 'round', 'million', 'financing', 'seed', 'venture',
            'capital', 'VC ', 'investors', 'euros', '€', 'IPO'
        ]
        
        funding_articles = []
        for article in all_articles:
            title = article['title'].lower()
            if any(keyword.lower() in title for keyword in funding_keywords):
                funding_articles.append(article)
        
        st.text(f"Found {len(funding_articles)} articles about funding")
        
        # Step 3: Extract detailed information from each article
        funding_data = []
        for idx, article in enumerate(funding_articles[:200]):  # Limiting to 200 to avoid excessive scraping
            try:
                title = article['title']
                url = article['url']
                date_str = article['date']
                
                # Skip if we don't have a date
                if not date_str:
                    continue
                
                # Try to convert string to date
                try:
                    funding_date = datetime.strptime(date_str, '%Y-%m-%d').date()
                except:
                    # If we couldn't parse the date, skip this article
                    continue
                
                # Extract startup name, funding amount from title
                startup_name = extract_startup_name(title)
                funding_amount = extract_funding_amount(title)
                funding_round = extract_funding_round(title)
                
                # Only continue if we have the key information
                if not (startup_name and funding_amount > 0):
                    continue
                
                # Try to get more details from the article itself
                try:
                    article_response = requests.get(url, headers=headers, timeout=10)
                    if article_response.status_code == 200:
                        article_soup = BeautifulSoup(article_response.content, 'html.parser')
                        article_text = article_soup.get_text()
                        
                        # Try to extract city from article text
                        city = extract_city(article_text)
                        industry = extract_industry(article_text)
                        investors = extract_investors(article_text)
                        
                        # If we couldn't extract these, use some defaults
                        if not city:
                            city = "Berlin"  # Default to most common German startup hub
                        if not industry:
                            industry = "Tech"  # Default industry
                        if not investors:
                            investors = "Undisclosed"
                        
                        # Skip non-German articles
                        if not is_german_startup(article_text, title, startup_name):
                            continue
                            
                    else:
                        # If we can't get the article, use some defaults
                        city = "Berlin"
                        industry = "Tech"
                        investors = "Undisclosed"
                except:
                    # If there's an error, use defaults
                    city = "Berlin"
                    industry = "Tech" 
                    investors = "Undisclosed"
                
                # Calculate quarter from the date
                year = funding_date.year
                month = funding_date.month
                quarter = (month - 1) // 3 + 1
                
                # Estimate valuation (if not found, typically 4-5x the funding amount)
                valuation_millions = funding_amount * random.uniform(4, 5)
                
                # Add the entry to our funding data
                funding_data.append({
                    'startup_name': startup_name,
                    'industry': industry,
                    'country': 'Germany',
                    'city': city,
                    'funding_round': funding_round,
                    'funding_amount_millions': funding_amount,
                    'funding_date': funding_date,
                    'investors': investors,
                    'valuation_millions': round(valuation_millions, 2),
                    'year': year,
                    'month': month,
                    'quarter': quarter,
                    'source': article['source'],
                    'source_url': url
                })
                
                # Log progress every 10 articles
                if (idx + 1) % 10 == 0:
                    st.text(f"Processed {idx + 1}/{len(funding_articles)} funding articles")
                
                # Add a short delay to avoid overwhelming the servers
                time.sleep(0.5)
                
            except Exception as e:
                st.warning(f"Error processing article {article['title']}: {str(e)}")
                continue
        
        # Remove duplicates (same startup with same funding round within 30 days)
        unique_funding_data = []
        seen_startups = {}
        
        for entry in funding_data:
            startup = entry['startup_name']
            funding_round = entry['funding_round']
            funding_date = entry['funding_date']
            
            key = f"{startup}|{funding_round}"
            
            if key in seen_startups:
                # If we've seen this startup+round before, check if it's within 30 days
                prev_date = seen_startups[key]
                if abs((funding_date - prev_date).days) <= 30:
                    # This is likely a duplicate, skip it
                    continue
            
            seen_startups[key] = funding_date
            unique_funding_data.append(entry)
        
        if not unique_funding_data:
            st.warning("No funding data could be extracted. Falling back to demo data.")
            return None
            
        st.success(f"Successfully extracted {len(unique_funding_data)} real German startup funding events")
        return unique_funding_data
    
    except Exception as e:
        st.error(f"Error in web scraping process: {str(e)}")
        st.exception(e)
        return None

# Helper functions for extracting information from articles
def parse_date_text(date_text):
    """Convert text dates like '5 days ago' to YYYY-MM-DD format"""
    today = datetime.now().date()
    
    if 'day' in date_text.lower():
        # Extract number of days
        days = re.search(r'(\d+)', date_text)
        if days:
            days = int(days.group(1))
            date = today - timedelta(days=days)
            return date.strftime('%Y-%m-%d')
    
    if 'week' in date_text.lower():
        # Extract number of weeks
        weeks = re.search(r'(\d+)', date_text)
        if weeks:
            weeks = int(weeks.group(1))
            date = today - timedelta(days=weeks*7)
            return date.strftime('%Y-%m-%d')
    
    if 'month' in date_text.lower():
        # Extract number of months (approximation)
        months = re.search(r'(\d+)', date_text)
        if months:
            months = int(months.group(1))
            # Approximate a month as 30 days
            date = today - timedelta(days=months*30)
            return date.strftime('%Y-%m-%d')
    
    # Try to parse as a standard date
    try:
        date = datetime.strptime(date_text, '%B %d, %Y').date()
        return date.strftime('%Y-%m-%d')
    except:
        pass
    
    try:
        date = datetime.strptime(date_text, '%d %B %Y').date()
        return date.strftime('%Y-%m-%d')
    except:
        pass
    
    # Return None if we couldn't parse the date
    return None

def extract_startup_name(title):
    """Extract startup name from article title"""
    # Common patterns like "Startup X raises €Y million"
    patterns = [
        r'([A-Z][A-Za-z0-9\-\.]+)(?:\s[A-Z][A-Za-z0-9\-\.]+)? raises',
        r'([A-Z][A-Za-z0-9\-\.]+)(?:\s[A-Z][A-Za-z0-9\-\.]+)? secures',
        r'([A-Z][A-Za-z0-9\-\.]+)(?:\s[A-Z][A-Za-z0-9\-\.]+)? gets',
        r'([A-Z][A-Za-z0-9\-\.]+)(?:\s[A-Z][A-Za-z0-9\-\.]+)? closes',
        r'([A-Z][A-Za-z0-9\-\.]+)(?:\s[A-Z][A-Za-z0-9\-\.]+)? lands',
        r'([A-Z][A-Za-z0-9\-\.]+)(?:\s[A-Z][A-Za-z0-9\-\.]+)? announces',
        r'([A-Z][A-Za-z0-9\-\.]+)(?:\s[A-Z][A-Za-z0-9\-\.]+)? bags',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, title)
        if match:
            return match.group(1).strip()
    
    # If no match is found, try to find the first proper noun 
    words = title.split()
    for word in words:
        if word[0].isupper() and len(word) > 1 and word.lower() not in ['german', 'berlin', 'munich', 'germany', 'funding', 'raises', 'secures', 'series']:
            return word.strip()
    
    return "Unknown Startup"

def extract_funding_amount(title):
    """Extract funding amount in millions of euros from article title"""
    # Look for patterns like €10 million or $10M or 10 million euros
    amount_patterns = [
        r'€(\d+(?:\.\d+)?)\s*[Mm]illion',
        r'€(\d+(?:\.\d+)?)\s*[Mm]',
        r'\$(\d+(?:\.\d+)?)\s*[Mm]illion',
        r'\$(\d+(?:\.\d+)?)\s*[Mm]',
        r'(\d+(?:\.\d+)?)\s*[Mm]illion\s*€',
        r'(\d+(?:\.\d+)?)\s*[Mm]illion\s*[Ee]uro',
        r'(\d+(?:\.\d+)?)\s*[Mm]illion',  # Fixed line
        r'(\d+(?:\.\d+)?)\s*[Bb]illion\s*€',
        r'(\d+(?:\.\d+)?)\s*[Bb]',
    ]
    
    for pattern in amount_patterns:
        match = re.search(pattern, title)
        if match:
            amount = float(match.group(1))
            
            # Convert billions to millions
            if 'billion' in pattern.lower():
                amount *= 1000
                
            return round(amount, 2) if 'amount' in locals() else None  # Ensure this is inside a function
    
    # Look for patterns like €10M or $10M
    amount_patterns_short = [
        r'€(\d+(?:\.\d+)?)[Mm]',
        r'\$(\d+(?:\.\d+)?)[Mm]',
        r'(\d+(?:\.\d+)?)[Mm]€',
        r'(\d+(?:\.\d+)?)[Mm]'
    ]

    for pattern in amount_patterns_short:
        match = re.search(pattern, title)
        if match:
            return round(float(match.group(1)), 2)
        r'€(\d+(?:\.\d+)?)[Mm]',
        r'\$(\d+(?:\.\d+)?)[Mm]',
        r'(\d+(?:\.\d+)?)[Mm]€',
        r'(\d+(?:\.\d+)?)[Mm]',  # Fixed line
    
    
    for pattern in amount_patterns_short:
        match = re.search(pattern, title)
        if match:
            return round(float(match.group(1)), 2)
    
    # Look for patterns like €10 (implied millions)
    amount_patterns_euro = [
        r'€(\d+(?:\.\d+)?)\s[Mm]',
        r'€(\d+(?:\.\d+)?)'
    ]
    
    for pattern in amount_patterns_euro:
        match = re.search(pattern, title)
        if match:
            amount = float(match.group(1))
            return round(amount, 2) if amount < 100 else round(amount/1000, 2)
    
    return 0  # Default if no amount found

# Function to fetch German startup funding data
def fetch_german_startup_data(start_date, end_date, num_entries=200):
    """
    Generates realistic demo data for German startups within the specified date range.
    
    Parameters:
    start_date (datetime): Start date for data generation
    end_date (datetime): End date for data generation
    num_entries (int): Number of entries to generate
    
    Returns:
    pd.DataFrame: Generated startup funding data
    """
    st.info("Generating demo data with real German startup names...")
    
    # German cities with startup ecosystems
    german_cities = [
        "Berlin", "Munich", "Hamburg", "Frankfurt", "Cologne", 
        "Stuttgart", "Düsseldorf", "Dresden", "Leipzig", "Karlsruhe",
        "Hannover", "Nuremberg", "Heidelberg", "Dortmund", "Essen"
    ]
    
    # Industry sectors prominent in Germany
    industries = [
        'Fintech', 'Healthtech', 'Edtech', 'AI/ML', 'E-commerce', 
        'Cybersecurity', 'Clean Energy', 'Biotech', 'SaaS', 'Mobility',
        'Logistics', 'Proptech', 'Foodtech', 'Insurtech', 'Manufacturing',
        'Agritech', 'Blockchain', 'IoT', 'B2B Software', 'Consumer Apps'
    ]
    
    # Common funding rounds
    funding_rounds = ['Seed', 'Series A', 'Series B', 'Series C', 'Series D+', 'Growth', 'IPO']
    
    # Real German startup names
    german_startup_names = [
        "N26", "Celonis", "Personio", "Lilium", "GetYourGuide", "Wefox", 
        "Trade Republic", "Solarisbank", "Mambu", "FlixBus", "HelloFresh", 
        "Infarm", "Tier Mobility", "Auto1", "Adjust", "Contentful", 
        "Taxfix", "Omio", "Emma", "Raisin", "Spryker", "Forto", "Enpal",
        "Choco", "Sennder", "Grover", "DeepL", "Pitch", "Razor Group", 
        "GoStudent", "Flink", "Vivid Money", "Fraugster", "Everphone",
        "Flix", "Volocopter", "The Climate Choice", "Dance", "Finleap", 
        "BOOM", "DyeMansion", "Gorillas", "Babbel", "SumUp", "Wunderflats",
        # Additional startups to reach higher count
        "Scalable Capital", "Joblift", "Blinkist", "McMakler", "Flaschenpost",
        "Tourlane", "Clark", "Billie", "Signavio", "Stocard", "IDnow",
        "CrossEngage", "Simplesurance", "Smacc", "Cluno", "Homeday", "Catchys",
        "Freighthub", "Medbelle", "Amboss", "Kaia Health", "EyeEm", "Zenjob",
        "CareerFoundry", "Bliq", "Klarx", "Doctolib", "Isar Aerospace",
        "Wandelbots", "CoachHub", "Xentral", "YFood", "Finanzguru", "MagForce",
        "Moberries", "Shyftplan", "Hometogo", "Solaris", "Thermondo", "Elinvar",
        "COMATCH", "Kreditech", "MEDWING", "Agricool", "Exporo", "Sono Motors",
        "Urban Sports Club", "Horizn Studios", "Qunomedical", "Jobpal",
        "FreightHub", "Campanda", "Sharpist", "Temedica", "Horizn Studios",
        "TIER Mobility", "Getyourguide", "Staffbase", "Signavio", "Quentic",
        "Talentspace", "Planetly", "Lendstar", "Zeitgold"
    ]
    
    # Real investor names
    german_investors = [
        "High-Tech Gründerfonds", "Earlybird Venture Capital", "Project A", 
        "Rocket Internet", "Point Nine Capital", "HV Capital", "Target Global", 
        "Cherry Ventures", "Speedinvest", "Acton Capital", "Cavalry Ventures", 
        "Creandum", "Northzone", "Atlantic Labs", "Amplifier", "Btov Partners", 
        "Capnamic Ventures", "Heartcore Capital", "La Famiglia", "Lakestar",
        "BlueYard Capital", "Picus Capital", "Global Founders Capital", 
        "Holtzbrinck Ventures", "EQT Ventures", "Balderton Capital", "Sequoia Capital",
        "Accel", "Benchmark", "Index Ventures", "Lightspeed Venture Partners",
        # Additional investors for more variety
        "Partech", "Atomico", "Headline", "DN Capital", "Insight Partners",
        "Tiger Global", "SoftBank Vision Fund", "Bain Capital Ventures", 
        "Eight Roads Ventures", "True Ventures", "Union Square Ventures",
        "Kizoo Technology", "Seventure Partners", "Round Hill Capital",
        "AP Ventures", "Impact Ventures", "Redstone", "Coparion",
        "Bayern Kapital", "Wachstumsfonds Bayern", "KfW Capital",
        "IBB Ventures", "NRW.Bank", "Mittelständische Beteiligungsgesellschaft", 
        "Rheingau Founders", "Signals Venture Capital", "Gründerfonds Ruhr"
    ]
    
    # Map startups to most likely industries (for more realistic data)
    startup_industries = {
        "N26": "Fintech",
        "Celonis": "AI/ML",
        "Personio": "SaaS",
        "Lilium": "Mobility",
        "GetYourGuide": "E-commerce",
        "Wefox": "Insurtech",
        "Trade Republic": "Fintech",
        "Solarisbank": "Fintech",
        "Mambu": "Fintech",
        "FlixBus": "Mobility",
        "HelloFresh": "Foodtech",
        "Infarm": "Agritech",
        "Tier Mobility": "Mobility",
        "Auto1": "E-commerce",
        "Adjust": "SaaS",
        "Contentful": "SaaS",
        "Taxfix": "Fintech",
        "Omio": "Mobility",
        "Emma": "E-commerce",
        "Raisin": "Fintech",
        "Spryker": "SaaS",
        "Forto": "Logistics",
        "Enpal": "Clean Energy",
        "Choco": "Foodtech",
        "Sennder": "Logistics",
        "Grover": "E-commerce",
        "DeepL": "AI/ML",
        "Pitch": "SaaS",
        "Razor Group": "E-commerce",
        "GoStudent": "Edtech",
        "Flink": "E-commerce",
        "Vivid Money": "Fintech",
        "Fraugster": "Fintech",
        "Everphone": "SaaS",
        "Flix": "Mobility",
        "Volocopter": "Mobility",
        "The Climate Choice": "Clean Energy",
        "Dance": "Mobility",
        "Finleap": "Fintech",
        "BOOM": "AI/ML",
        "DyeMansion": "Manufacturing",
        "Gorillas": "E-commerce",
        "Babbel": "Edtech",
        "SumUp": "Fintech",
        "Wunderflats": "Proptech",
        "Scalable Capital": "Fintech",
        "Blinkist": "Edtech",
        "Billie": "Fintech",
        "Tourlane": "E-commerce",
        "Clark": "Insurtech",
        "Simplesurance": "Insurtech",
        "Kaia Health": "Healthtech",
        "Doctolib": "Healthtech",
        "YFood": "Foodtech",
        "Sono Motors": "Mobility",
        "CoachHub": "SaaS",
        "Isar Aerospace": "Manufacturing"
    }
    
    # Map startups to most likely cities (for more realistic data)
    startup_cities = {
        "N26": "Berlin",
        "Celonis": "Munich",
        "Personio": "Munich",
        "Lilium": "Munich",
        "GetYourGuide": "Berlin",
        "Wefox": "Berlin",
        "Trade Republic": "Berlin",
        "Solarisbank": "Berlin",
        "Mambu": "Berlin",
        "FlixBus": "Munich",
        "HelloFresh": "Berlin",
        "Infarm": "Berlin",
        "Tier Mobility": "Berlin",
        "Auto1": "Berlin",
        "Adjust": "Berlin",
        "Contentful": "Berlin",
        "Taxfix": "Berlin",
        "Omio": "Berlin",
        "Emma": "Frankfurt",
        "Raisin": "Berlin",
        "Spryker": "Berlin",
        "Forto": "Berlin",
        "Enpal": "Berlin",
        "Choco": "Berlin",
        "Sennder": "Berlin",
        "Grover": "Berlin",
        "DeepL": "Cologne",
        "Pitch": "Berlin",
        "Razor Group": "Berlin",
        "GoStudent": "Munich",
        "Flink": "Berlin",
        "Vivid Money": "Berlin",
        "Fraugster": "Berlin",
        "Everphone": "Berlin",
        "Flix": "Munich",
        "Volocopter": "Stuttgart",
        "The Climate Choice": "Berlin",
        "Dance": "Berlin",
        "Finleap": "Berlin",
        "BOOM": "Munich",
        "DyeMansion": "Munich",
        "Gorillas": "Berlin",
        "Babbel": "Berlin",
        "SumUp": "Berlin",
        "Wunderflats": "Berlin",
        "Scalable Capital": "Munich",
        "Blinkist": "Berlin",
        "Clark": "Frankfurt",
        "Billie": "Berlin",
        "Tourlane": "Berlin",
        "Kaia Health": "Munich",
        "Doctolib": "Berlin",
        "Isar Aerospace": "Munich",
        "YFood": "Munich",
        "Sono Motors": "Munich",
        "CoachHub": "Berlin"
    }
    
    # Date range for data generation
    days_range = (end_date - start_date).days
    if days_range <= 0:
        st.error("End date must be after start date")
        return pd.DataFrame()
    
    data = []
    
    # Allow multiple rounds per startup to reach higher count
    # Calculate how many rounds per startup we need
    available_startups = len(german_startup_names)
    rounds_per_startup = max(1, min(3, num_entries // available_startups + 1))
    
    # Distribute startups evenly across the time period
    for startup_name in german_startup_names:
        # Generate up to rounds_per_startup funding rounds for each startup
        # But stop if we've reached our target
        if len(data) >= num_entries:
            break
            
        # Get most likely industry for this startup, or random if not mapped
        industry = startup_industries.get(startup_name, random.choice(industries))
        
        # Get most likely city for this startup, or random if not mapped
        city = startup_cities.get(startup_name, random.choice(german_cities))
        
        for i in range(rounds_per_startup):
            # Stop if we've reached our target
            if len(data) >= num_entries:
                break
                
            # Space out rounds for the same startup
            segment_size = days_range // rounds_per_startup
            segment_start = i * segment_size
            segment_end = (i + 1) * segment_size - 1
            
            random_days = random.randint(segment_start, segment_end)
            funding_date = start_date + timedelta(days=random_days)
            
            # Choose appropriate funding round based on sequence
            if i == 0:
                possible_rounds = ['Seed', 'Series A']
            elif i == 1:
                possible_rounds = ['Series A', 'Series B']
            else:
                possible_rounds = ['Series B', 'Series C', 'Series D+', 'Growth']
                
            round_type = random.choice(possible_rounds)
            
            # Generate appropriate funding amount based on startup profile and round
            if startup_name in ["N26", "Celonis", "Lilium", "GetYourGuide", "HelloFresh", "Auto1"]:
                # Well-known larger startups
                if round_type == 'Seed':
                    amount = random.uniform(1.5, 4.0)  # Higher seed for known companies
                elif round_type == 'Series A':
                    amount = random.uniform(10.0, 25.0)  # Higher Series A
                elif round_type == 'Series B':
                    amount = random.uniform(30.0, 80.0)
                elif round_type == 'Series C':
                    amount = random.uniform(80.0, 150.0)
                elif round_type == 'Series D+':
                    amount = random.uniform(150.0, 300.0)
                elif round_type == 'Growth':
                    amount = random.uniform(100.0, 500.0)
                else:  # IPO
                    amount = random.uniform(300.0, 1000.0)
            elif startup_name in ["Personio", "Wefox", "Trade Republic", "Tier Mobility", "Gorillas"]:
                # Medium-sized well-known startups
                if round_type == 'Seed':
                    amount = random.uniform(1.0, 3.0)
                elif round_type == 'Series A':
                    amount = random.uniform(8.0, 20.0)
                elif round_type == 'Series B':
                    amount = random.uniform(20.0, 60.0)
                elif round_type == 'Series C':
                    amount = random.uniform(60.0, 120.0)
                else:
                    amount = random.uniform(100.0, 200.0)
            else:
                # Other startups
                if round_type == 'Seed':
                    amount = random.uniform(0.5, 3.0)  # €500K to €3M
                elif round_type == 'Series A':
                    amount = random.uniform(3.0, 15.0)  # €3M to €15M
                elif round_type == 'Series B':
                    amount = random.uniform(15.0, 50.0)  # €15M to €50M
                elif round_type == 'Series C':
                    amount = random.uniform(50.0, 100.0)  # €50M to €100M
                elif round_type == 'Series D+':
                    amount = random.uniform(100.0, 300.0)  # €100M to €300M
                elif round_type == 'Growth':
                    amount = random.uniform(50.0, 500.0)  # €50M to €500M
                else:  # IPO
                    amount = random.uniform(200.0, 1000.0)  # €200M to €1B
            
            # Adjust real investors count based on round size
            if amount < 5:  # Small rounds have fewer investors
                num_investors = random.randint(1, 2)
            elif amount < 20:  # Medium rounds
                num_investors = random.randint(2, 3)
            elif amount < 100:  # Larger rounds
                num_investors = random.randint(3, 5)
            else:  # Very large rounds
                num_investors = random.randint(4, 7)
                
            investors = random.sample(german_investors, num_investors)
            
            # Valuation (typically a multiple of the funding amount)
            valuation_multiple = random.uniform(3, 10)
            valuation = amount * valuation_multiple
            
            data.append({
                'startup_name': startup_name,
                'industry': industry,
                'country': 'Germany',
                'city': city,
                'funding_round': round_type,
                'funding_amount_millions': round(amount, 2),
                'funding_date': funding_date,
                'investors': ', '.join(investors),
                'valuation_millions': round(valuation, 2),
                'year': funding_date.year,
                'month': funding_date.month,
                'quarter': (funding_date.month - 1) // 3 + 1
            })
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    st.success(f"Generated {len(df)} demo funding rounds with real German startup names.")
    return df

# Function to save data as CSV
def save_data_as_csv(df):
    """Save dataframe as CSV with timestamp in filename"""
    timestamp = datetime.now().strftime("%Y%m%dT%H-%M")
    filename = f"{timestamp}_export.csv"
    df.to_csv(filename, index=False)
    return filename

# Function to filter data based on user selections
def filter_data(df, industries, cities, funding_rounds, date_range):
    """
    Filter the dataframe based on user selections
    
    Parameters:
    df (pd.DataFrame): Input dataframe
    industries (list): Selected industries
    cities (list): Selected cities
    funding_rounds (list): Selected funding rounds
    date_range (list): Start and end date
    
    Returns:
    pd.DataFrame: Filtered dataframe
    """
    filtered_df = df.copy()
    
    if industries:
        filtered_df = filtered_df[filtered_df['industry'].isin(industries)]
    
    if cities:
        filtered_df = filtered_df[filtered_df['city'].isin(cities)]
    
    if funding_rounds:
        filtered_df = filtered_df[filtered_df['funding_round'].isin(funding_rounds)]
    
    if date_range:
        filtered_df = filtered_df[
            (filtered_df['funding_date'] >= date_range[0]) & 
            (filtered_df['funding_date'] <= date_range[1])
        ]
    
    return filtered_df

# Function to generate insights from data
def generate_insights(df):
    insights = []
    
    # Top industries by funding
    top_industries = df.groupby('industry')['funding_amount_millions'].sum().sort_values(ascending=False)
    if not top_industries.empty:
        top_industry = top_industries.index[0]
        top_industry_amount = top_industries.iloc[0]
        insights.append(f"The {top_industry} industry has received the most funding (€{top_industry_amount:.2f}M).")
    
    # Top cities by funding
    top_cities = df.groupby('city')['funding_amount_millions'].sum().sort_values(ascending=False)
    if not top_cities.empty:
        top_city = top_cities.index[0]
        top_city_amount = top_cities.iloc[0]
        insights.append(f"{top_city} leads in funding with €{top_city_amount:.2f}M raised.")
    
    # Fastest growing industry (last 6 months vs previous 6 months)
    today = datetime.now()
    six_months_ago = today - timedelta(days=180)
    twelve_months_ago = today - timedelta(days=365)
    
    recent_df = df[df['funding_date'] >= six_months_ago]
    previous_df = df[(df['funding_date'] < six_months_ago) & (df['funding_date'] >= twelve_months_ago)]
    
    if not recent_df.empty and not previous_df.empty:
        recent_by_industry = recent_df.groupby('industry')['funding_amount_millions'].sum()
        previous_by_industry = previous_df.groupby('industry')['funding_amount_millions'].sum()
        
        # Calculate growth rate for industries present in both periods
        growth_rates = {}
        for industry in set(recent_by_industry.index) & set(previous_by_industry.index):
            if previous_by_industry[industry] > 0:  # Avoid division by zero
                growth_rates[industry] = (recent_by_industry[industry] / previous_by_industry[industry]) - 1
        
        if growth_rates:
            fastest_growing = max(growth_rates.items(), key=lambda x: x[1])
            insights.append(f"{fastest_growing[0]} shows the fastest growth with a {fastest_growing[1]:.1%} increase in funding over the last 6 months.")
    
    # Largest round in the dataset
    if not df.empty:
        largest_round = df.loc[df['funding_amount_millions'].idxmax()]
        insights.append(f"The largest funding round was €{largest_round['funding_amount_millions']:.2f}M for {largest_round['startup_name']} ({largest_round['funding_round']}) in {largest_round['city']}.")
    
    # Most active cities
    city_counts = df['city'].value_counts()
    if not city_counts.empty:
        most_active_city = city_counts.index[0]
        city_deal_count = city_counts.iloc[0]
        insights.append(f"{most_active_city} is the most active city with {city_deal_count} funding deals.")
    
    # Most popular funding round
    round_counts = df['funding_round'].value_counts()
    if not round_counts.empty:
        most_common_round = round_counts.index[0]
        round_count = round_counts.iloc[0]
        insights.append(f"{most_common_round} is the most common funding round with {round_count} deals.")
    
    return insights

# Main application
def main():
    # Header
    st.markdown("<h1 class='main-header'>German Startup Funding Analyzer</h1>", unsafe_allow_html=True)
    st.markdown("Explore and visualize German startup funding trends across industries, cities, and time periods.")
    st.markdown(" > Search with API is recommended option.")
    st.markdown(" > However, there is a limitation with Groq API. For better search in the funding area, Crunchbase API and Dealroom API are recommended.")

    # Data Collection Section - Added at the beginning
    st.subheader("Data Collection")
    
    # API Options
    api_option = st.radio(
        "Choose data source:",
        ["Groq API (requires key)", "Web Scraping (real-time news)", "Demo Data (with real startup names)"]
    )
    
    # Date range selection for data collection
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input(
            "Start Date for Data Collection",
            value=datetime.now() - timedelta(days=365*2),  # Default to 2 years ago
            max_value=datetime.now()
        )
    
    with col2:
        end_date = st.date_input(
            "End Date for Data Collection",
            value=datetime.now(),
            max_value=datetime.now()
        )
    
    # API Key input if Groq is selected
    if api_option == "Groq API (requires key)":
        api_key = st.text_input("Enter Groq API Key:", type="password", 
                               help="You can get a Groq API key from https://console.groq.com/keys")
        if api_key:
            st.session_state['api_key'] = api_key
    
    # Convert to datetime for data processing
    start_datetime = datetime.combine(start_date, datetime.min.time())
    end_datetime = datetime.combine(end_date, datetime.max.time())
    
    # Button to initiate data collection
    if st.button("Collect German Startup Funding Data"):
        with st.spinner("Collecting data..."):
            # Determine which data source to use based on selection
            if api_option == "Groq API (requires key)":
                api_key = st.session_state.get('api_key', '')
                data = fetch_from_groq(start_datetime, end_datetime, api_key)
                
                if data is None or data.empty:
                    st.warning("Could not fetch data from Groq API. Falling back to demo data.")
                    data = fetch_german_startup_data(start_datetime, end_datetime)
            
            elif api_option == "Web Scraping (real-time news)":
                data = scrape_startup_funding_news()
                
                if data:
                    # Convert list of dicts to DataFrame if successful
                    data = pd.DataFrame(data)
                    
                    # Add missing columns
                    data['funding_date'] = pd.to_datetime(data['funding_date'])
                    data['year'] = data['funding_date'].dt.year
                    data['month'] = data['funding_date'].dt.month
                    data['quarter'] = (data['funding_date'].dt.month - 1) // 3 + 1
                else:
                    st.warning("Web scraping did not return any results. Falling back to demo data.")
                    data = fetch_german_startup_data(start_datetime, end_datetime)
            
            else:  # Demo data
                data = fetch_german_startup_data(start_datetime, end_datetime)
            
            if data is None or data.empty:
                st.error("No data found for the specified period.")
            else:
                # Save as CSV
                filename = save_data_as_csv(data)
                st.success(f"Data collected successfully! Saved to {filename}")
                
                # Save to session state for use in the app
                st.session_state['startup_data'] = data
                st.session_state['data_loaded'] = True
                
                # Create download button for the data
                csv = data.to_csv(index=False)
                st.download_button(
                    label="Download data as CSV",
                    data=csv,
                    file_name=f"german_startup_data_{datetime.now().strftime('%Y%m%dT%H-%M')}.csv",
                    mime="text/csv"
                )
    else:
        st.error("No data found for the specified period.")
    
    # Check if we have data to proceed
    if 'data_loaded' not in st.session_state or not st.session_state['data_loaded']:
        # Try to load from previously saved file or use placeholder message
        st.info("Please collect data first using the button above or upload a previously saved file.")
        
        uploaded_file = st.file_uploader("Upload previously saved CSV file", type="csv")
        if uploaded_file is not None:
            data = pd.read_csv(uploaded_file)
            
            # Convert date column to datetime
            data['funding_date'] = pd.to_datetime(data['funding_date'])
            
            # Add required columns if missing
            if 'year' not in data.columns:
                data['year'] = data['funding_date'].dt.year
            if 'month' not in data.columns:
                data['month'] = data['funding_date'].dt.month
            if 'quarter' not in data.columns:
                data['quarter'] = (data['funding_date'].dt.month - 1) // 3 + 1
            
            st.session_state['startup_data'] = data
            st.session_state['data_loaded'] = True
            st.success("Data loaded successfully!")
        else:
            # No data loaded yet, stop execution
            st.stop()
    
    # Get data from session state
    data = st.session_state['startup_data']
    
    # Filters in Sidebar
    st.sidebar.header("Analysis Filters")
    
    # Date Range Filter for Analysis (separate from data collection)
    min_date = data['funding_date'].min().date()
    max_date = data['funding_date'].max().date()
    analysis_date_range = st.sidebar.date_input(
        "Select Date Range for Analysis",
        [min_date, max_date],
        min_value=min_date,
        max_value=max_date
    )
    
    # Convert to datetime for filtering
    if len(analysis_date_range) == 2:
        analysis_date_range = [pd.Timestamp(date) for date in analysis_date_range]
    else:
        analysis_date_range = None
    
    # Multi-select filters
    industries = st.sidebar.multiselect("Select Industries", options=sorted(data['industry'].unique()))
    cities = st.sidebar.multiselect("Select Cities", options=sorted(data['city'].unique()))
    funding_rounds = st.sidebar.multiselect("Select Funding Rounds", options=sorted(data['funding_round'].unique()))
    
    # Apply filters
    filtered_data = filter_data(data, industries, cities, funding_rounds, analysis_date_range)
    
    # Display filter summary
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Showing {len(filtered_data)} of {len(data)} funding rounds**")
    
    # Add download button for filtered data
    csv_filtered = filtered_data.to_csv(index=False)
    st.sidebar.download_button(
        label="Download Filtered Data",
        data=csv_filtered,
        file_name=f"filtered_german_startup_data_{datetime.now().strftime('%Y%m%dT%H-%M')}.csv",
        mime="text/csv"
    )
    
    # Check if we have data after filtering
    if filtered_data.empty:
        st.warning("No data matches your filter criteria. Please adjust your filters.")
        st.stop()

    # Calculate key metrics
    total_funding = filtered_data['funding_amount_millions'].sum()
    avg_round_size = filtered_data['funding_amount_millions'].mean()
    num_deals = len(filtered_data)
    num_industries = filtered_data['industry'].nunique()

    # DASHBOARD LAYOUT
    # Key Metrics Row
    st.markdown("<h2 class='sub-header'>Key Metrics</h2>", unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    
    # # Export full dataset button
    # if st.button("Export Full Dataset as CSV"):
    #     csv_full = data.to_csv(index=False)
    #     st.download_button(
    #         label="Download Complete Dataset",
    #         data=csv_full,
    #         file_name=f"complete_german_startup_data_{datetime.now().strftime('%Y%m%dT%H-%M')}.csv",
    #         mime="text/csv",
    #         key="full_data_download"
    #     )
    
    # Helper function to create consistent cards
    def create_metric_card(col, title, value, unit=""):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <h3>{title}</h3>
                <p style="font-size:24px; font-weight:bold;">{value}{unit}</p>
                <p style="visibility: hidden;">_</p>  <!-- Hidden spacer -->
            </div>
            """, unsafe_allow_html=True)
    
    # Create cards
    create_metric_card(col1, "Total Funding", f"€{total_funding:,.2f}", "M")
    create_metric_card(col2, "Avg. Round Size", f"€{avg_round_size:,.2f}", "M")
    create_metric_card(col3, "Number of Deals", f"{num_deals:,}")
    create_metric_card(col4, "Industries", f"{num_industries}")

    # Visualization Tabs - updated to replace "Regional Analysis" with "City Analysis"
    st.markdown("<h2 class='sub-header'>Trend Analysis</h2>", unsafe_allow_html=True)
    tab1, tab2, tab3, tab4 = st.tabs(["Industry Analysis", "City Analysis", "Timeline Analysis", "Round Type Analysis"])
    
    # TAB 1: Industry Analysis
    with tab1:
        # Funding by Industry (Bar Chart)
        st.subheader("Funding by Industry")
        
        # Define industry_funding before referencing it
        industry_funding = filtered_data.groupby('industry')['funding_amount_millions'].sum().sort_values(ascending=False)
        
        industry_deal_count = filtered_data.groupby('industry').size()
        
        industry_df = pd.DataFrame({
            'Total Funding (€M)': industry_funding,
            'Number of Deals': industry_deal_count
        }).reset_index()
        
        fig1 = px.bar(
            industry_df,
            x='industry',
            y='Total Funding (€M)',
            hover_data=['Number of Deals'],
            color='Total Funding (€M)',
            labels={'industry': 'Industry', 'Total Funding (€M)': 'Total Funding (€M)'},
            height=500,
            color_continuous_scale=px.colors.sequential.Blues
        )
        fig1.update_layout(xaxis_title="Industry", yaxis_title="Total Funding (€M)")
        st.plotly_chart(fig1, use_container_width=True)

        # Average Funding Round by Industry (Bar Chart)
        st.subheader("Average Funding Round Size by Industry")
        avg_by_industry = filtered_data.groupby('industry')['funding_amount_millions'].mean().sort_values(ascending=False)
        
        fig2 = px.bar(
            avg_by_industry,
            labels={'industry': 'Industry', 'value': 'Average Funding (€M)'},
            height=400,
            color=avg_by_industry.values,
            color_continuous_scale=px.colors.sequential.Viridis
        )
        fig2.update_layout(xaxis_title="Industry", yaxis_title="Average Funding (€M)")
        st.plotly_chart(fig2, use_container_width=True)
        
        # Industry Breakdown by Funding Round Type
        st.subheader("Industry Breakdown by Funding Round Type")
        industry_round_breakdown = pd.crosstab(
            filtered_data['industry'], 
            filtered_data['funding_round'],
            values=filtered_data['funding_amount_millions'], 
            aggfunc='sum'
        ).fillna(0)
        
        fig3 = px.bar(
            industry_round_breakdown.reset_index().melt(id_vars='industry'),
            x='industry',
            y='value',
            color='funding_round',
            labels={'industry': 'Industry', 'value': 'Funding Amount (€M)', 'funding_round': 'Funding Round'},
            height=500,
            barmode='stack'
        )
        fig3.update_layout(xaxis_title="Industry", yaxis_title="Funding Amount (€M)")
        st.plotly_chart(fig3, use_container_width=True)
    
    # TAB 2: City Analysis (replacing Regional Analysis)
    with tab2:
        # Funding by City (Pie Chart)
        st.subheader("Funding Distribution by City")
        city_funding = filtered_data.groupby('city')['funding_amount_millions'].sum()
        
        fig4 = px.pie(
            names=city_funding.index,
            values=city_funding.values,
            hole=0.4,
            labels={'names': 'City', 'values': 'Total Funding (€M)'},
            height=400
        )
        fig4.update_traces(textinfo='percent+label', textposition='outside')
        st.plotly_chart(fig4, use_container_width=True)
        
        # Top Cities by Funding
        st.subheader("Top Cities by Funding")
        
        top_cities = city_funding.sort_values(ascending=False).reset_index()
        top_cities.columns = ['city', 'funding_amount_millions']
        
        fig5 = px.bar(
            top_cities.head(10),  # Show top 10 cities
            x='city',
            y='funding_amount_millions',
            color='funding_amount_millions',
            labels={'city': 'City', 'funding_amount_millions': 'Total Funding (€M)'},
            height=500,
            color_continuous_scale=px.colors.sequential.Reds
        )
        fig5.update_layout(xaxis_title="City", yaxis_title="Total Funding (€M)")
        st.plotly_chart(fig5, use_container_width=True)
        
        # Investment Flow - City to Industry Sankey Diagram
        st.subheader("Investment Flow: City to Industry")
        
        # Create links and nodes for Sankey diagram
        city_industry_funding = filtered_data.groupby(['city', 'industry'])['funding_amount_millions'].sum().reset_index()
        
        # Create nodes (unique cities and industries)
        cities_list = city_industry_funding['city'].unique().tolist()
        industries_list = city_industry_funding['industry'].unique().tolist()
        
        nodes = cities_list + industries_list
        
        # Create links (city to industry)
        source_indices = [cities_list.index(city) for city in city_industry_funding['city']]
        target_indices = [len(cities_list) + industries_list.index(industry) for industry in city_industry_funding['industry']]
        
        # Create Sankey diagram
        fig6 = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=nodes
            ),
            link=dict(
                source=source_indices,
                target=target_indices,
                value=city_industry_funding['funding_amount_millions'].tolist()
            )
        )])
        
        fig6.update_layout(height=600)
        st.plotly_chart(fig6, use_container_width=True)
    
    # TAB 3: Timeline Analysis
    with tab3:
        # Funding Over Time
        st.subheader("Funding Trends Over Time")
        
        # Aggregate by month and year
        timeline_option = st.radio(
            "Select time granularity:",
            ["Monthly", "Quarterly", "Yearly"],
            horizontal=True,
            key="timeline_granularity_radio"
        )
        
        if timeline_option == "Monthly":
            time_agg = filtered_data.groupby(['year', 'month'])['funding_amount_millions'].sum().reset_index()
            time_agg['date'] = pd.to_datetime(time_agg[['year', 'month']].assign(day=1))
            time_field = 'date'
            title = "Monthly Funding Trends"
        elif timeline_option == "Quarterly":
            time_agg = filtered_data.groupby(['year', 'quarter'])['funding_amount_millions'].sum().reset_index()
            time_agg['date'] = time_agg.apply(lambda x: f"Q{x['quarter']} {x['year']}", axis=1)
            time_field = 'date'
            title = "Quarterly Funding Trends"
        else:  # Yearly
            time_agg = filtered_data.groupby('year')['funding_amount_millions'].sum().reset_index()
            time_field = 'year'
            title = "Yearly Funding Trends"
        
        fig7 = px.line(
            time_agg,
            x=time_field,
            y='funding_amount_millions',
            markers=True,
            labels={time_field: 'Time Period', 'funding_amount_millions': 'Total Funding (€M)'},
            title=title,
            height=400
        )
        fig7.update_layout(xaxis_title="Time Period", yaxis_title="Total Funding (€M)")
        st.plotly_chart(fig7, use_container_width=True)
        
        # Deal Count Over Time
        st.subheader("Deal Count Over Time")
        
        if timeline_option == "Monthly":
            count_agg = filtered_data.groupby(['year', 'month']).size().reset_index(name='count')
            count_agg['date'] = pd.to_datetime(count_agg[['year', 'month']].assign(day=1))
            count_field = 'date'
        elif timeline_option == "Quarterly":
            count_agg = filtered_data.groupby(['year', 'quarter']).size().reset_index(name='count')
            count_agg['date'] = count_agg.apply(lambda x: f"Q{x['quarter']} {x['year']}", axis=1)
            count_field = 'date'
        else:  # Yearly
            count_agg = filtered_data.groupby('year').size().reset_index(name='count')
            count_field = 'year'
        
        fig8 = px.bar(
            count_agg,
            x=count_field,
            y='count',
            labels={count_field: 'Time Period', 'count': 'Number of Deals'},
            height=400,
            color='count',
            color_continuous_scale=px.colors.sequential.Reds
        )
        fig8.update_layout(xaxis_title="Time Period", yaxis_title="Number of Deals")
        st.plotly_chart(fig8, use_container_width=True)
        
        # Industry Trends Over Time
        st.subheader("Industry Funding Trends Over Time")
        
        # Select top N industries to display
        top_n = st.slider("Select number of top industries to display:", 3, 10, 5)
        
        # Get top N industries by total funding
        top_industries_list = industry_funding.head(top_n).index.tolist()
        
        # Filter for only these industries
        top_industry_data = filtered_data[filtered_data['industry'].isin(top_industries_list)]
        
        if timeline_option == "Monthly":
            industry_time_agg = top_industry_data.groupby(['year', 'month', 'industry'])['funding_amount_millions'].sum().reset_index()
            industry_time_agg['date'] = pd.to_datetime(industry_time_agg[['year', 'month']].assign(day=1))
            industry_time_field = 'date'
        elif timeline_option == "Quarterly":
            industry_time_agg = top_industry_data.groupby(['year', 'quarter', 'industry'])['funding_amount_millions'].sum().reset_index()
            industry_time_agg['date'] = industry_time_agg.apply(lambda x: f"Q{x['quarter']} {x['year']}", axis=1)
            industry_time_field = 'date'
        else:  # Yearly
            industry_time_agg = top_industry_data.groupby(['year', 'industry'])['funding_amount_millions'].sum().reset_index()
            industry_time_field = 'year'
        
        fig9 = px.line(
            industry_time_agg,
            x=industry_time_field,
            y='funding_amount_millions',
            color='industry',
            markers=True,
            labels={industry_time_field: 'Time Period', 'funding_amount_millions': 'Funding Amount (€M)', 'industry': 'Industry'},
            height=500
        )
        fig9.update_layout(xaxis_title="Time Period", yaxis_title="Funding Amount (€M)")
        st.plotly_chart(fig9, use_container_width=True)
    
    # TAB 4: Round Type Analysis
    with tab4:
        # Funding by Round Type
        st.subheader("Funding by Round Type")
        round_funding = filtered_data.groupby('funding_round')['funding_amount_millions'].sum().sort_values(ascending=False)
        round_deal_count = filtered_data.groupby('funding_round').size()
        
        round_df = pd.DataFrame({
            'Total Funding (€M)': round_funding,
            'Number of Deals': round_deal_count
        }).reset_index()
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig10 = px.bar(
                round_df,
                x='funding_round',
                y='Total Funding (€M)',
                color='funding_round',
                labels={'funding_round': 'Funding Round', 'Total Funding (€M)': 'Total Funding (€M)'},
                height=400
            )
            fig10.update_layout(xaxis_title="Funding Round", yaxis_title="Total Funding (€M)")
            st.plotly_chart(fig10, use_container_width=True)
        
        with col2:
            fig11 = px.bar(
                round_df,
                x='funding_round',
                y='Number of Deals',
                color='funding_round',
                labels={'funding_round': 'Funding Round', 'Number of Deals': 'Number of Deals'},
                height=400
            )
            fig11.update_layout(xaxis_title="Funding Round", yaxis_title="Number of Deals")
            st.plotly_chart(fig11, use_container_width=True)
        
        # Average Round Size
        st.subheader("Average Funding by Round Type")
        avg_by_round = filtered_data.groupby('funding_round')['funding_amount_millions'].mean().sort_values()
        
        fig12 = px.bar(
            avg_by_round,
            orientation='h',
            labels={'funding_round': 'Funding Round', 'value': 'Average Funding (€M)'},
            height=400,
            color=avg_by_round.values,
            color_continuous_scale=px.colors.sequential.Greens
        )
        fig12.update_layout(xaxis_title="Average Funding (€M)", yaxis_title="Funding Round")
        st.plotly_chart(fig12, use_container_width=True)
        
        # Round Distribution by Industry
        st.subheader("Round Type Distribution by Industry")
        # Select top N industries by funding
        top_n_rounds = st.slider("Select number of top industries:", 3, 10, 5, key="round_dist_slider")
        top_industries_rounds = industry_funding.head(top_n_rounds).index.tolist()
        
        # Filter for selected industries
        industry_rounds_data = filtered_data[filtered_data['industry'].isin(top_industries_rounds)]
        
        # Create a normalized stacked bar chart
        round_dist = pd.crosstab(
            industry_rounds_data['industry'],
            industry_rounds_data['funding_round'],
            normalize='index'
        ) * 100  # Convert to percentage
        
        fig13 = px.bar(
            round_dist.reset_index().melt(id_vars='industry'),
            x='industry',
            y='value',
            color='funding_round',
            labels={'industry': 'Industry', 'value': 'Percentage of Deals', 'funding_round': 'Funding Round'},
            height=500,
            barmode='stack'
        )
        fig13.update_layout(xaxis_title="Industry", yaxis_title="Percentage of Deals (%)")
        st.plotly_chart(fig13, use_container_width=True)
    
    # City Funding Map (New visualization)
    st.markdown("<h2 class='sub-header'>City Funding Map</h2>", unsafe_allow_html=True)
    
    # Aggregate funding by city
    city_funding_map = filtered_data.groupby('city')['funding_amount_millions'].sum().reset_index()
    city_deal_count = filtered_data.groupby('city').size().reset_index(name='deal_count')
    city_funding_map = city_funding_map.merge(city_deal_count, on='city')
    
    # Create map - Note: In a real implementation, we would add geocoding for precise locations
    # For this example, we'll use a simple bar chart as a placeholder
    fig_map = px.bar(
        city_funding_map.sort_values('funding_amount_millions', ascending=False),
        x='city',
        y='funding_amount_millions',
        color='deal_count',
        labels={
            'city': 'City', 
            'funding_amount_millions': 'Total Funding (€M)',
            'deal_count': 'Number of Deals'
        },
        height=500,
        color_continuous_scale=px.colors.sequential.Viridis
    )
    fig_map.update_layout(xaxis_title="City", yaxis_title="Total Funding (€M)")
    st.plotly_chart(fig_map, use_container_width=True)
    
    # AI-Generated Insights
    st.markdown("<h2 class='sub-header'>AI-Generated Insights</h2>", unsafe_allow_html=True)
    insights = generate_insights(filtered_data)

    st.markdown(f"<div class='insight-box'>", unsafe_allow_html=True)
    for i, insight in enumerate(insights):
        st.markdown(f"<p>{i + 1}. {insight}</p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
amount_patterns = [
    r'(\d+(?:\.\d+)?)\s*[Bb]illion\s*€',
    r'(\d+(?:\.\d+)?)\s*[Bb]'
]

def extract_funding_amount(title):
    """
    Extract funding amount in millions of euros from a given title.
    """
    for pattern in amount_patterns:
        match = re.search(pattern, title)
        if match:
            amount = float(match.group(1))
            
            # Convert billions to millions
            if '[Bb]illion' in pattern or '[Bb]' in pattern:
                amount *= 1000
                
            # If using default of millions, no conversion needed
            return round(amount, 2)
    
