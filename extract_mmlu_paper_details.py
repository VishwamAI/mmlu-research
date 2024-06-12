import requests
from bs4 import BeautifulSoup

# List of URLs to research papers related to MMLU projects
paper_links = [
    'https://paperswithcode.com/paper/gemini-a-family-of-highly-capable-multimodal-1',
    'https://paperswithcode.com/paper/gpt-4-technical-report-1',
    'https://paperswithcode.com/paper/the-claude-3-model-family-opus-sonnet-haiku',
    'https://paperswithcode.com/paper/leeroo-orchestrator-elevating-llms',
    'https://paperswithcode.com/paper/palm-2-technical-report-1',
    'https://paperswithcode.com/paper/scaling-instruction-finetuned-language-models',
    'https://paperswithcode.com/paper/replug-retrieval-augmented-black-box-language',
    'https://paperswithcode.com/paper/transcending-scaling-laws-with-0-1-extra',
    'https://paperswithcode.com/paper/mixtral-of-experts'
]

# Function to extract paper details from a given URL
def extract_paper_details(url):
    response = requests.get(url)
    response.encoding = 'utf-8'  # Ensure the response content is correctly decoded
    print(response.content)  # Print the response content for debugging
    soup = BeautifulSoup(response.content, 'html.parser')

    title_tag = soup.find('h1')
    if title_tag:
        print(f"Title tag found: {title_tag.get_text(strip=True)}")
    else:
        print("Title tag not found")
    title = title_tag.get_text(strip=True) if title_tag else "Title not available"

    abstract_div = soup.find('div', class_='paper-abstract')
    if abstract_div:
        print(f"Abstract found: {abstract_div.get_text(strip=True)}")
    else:
        print("Abstract not found")
    abstract = abstract_div.get_text(strip=True) if abstract_div else "Abstract not available"

    return {
        'title': title,
        'abstract': abstract,
        'url': url
    }

# Extract details for each paper and save them in a list
papers = []
for link in paper_links:
    try:
        print(f"Processing URL: {link}")
        paper_details = extract_paper_details(link)
        papers.append(paper_details)
    except Exception as e:
        print(f"Failed to extract details from {link}: {e}")

# Save the extracted details to a file
with open('/home/ubuntu/mmlu_paper_details.txt', 'w') as f:
    for paper in papers:
        f.write(f"Title: {paper['title']}\n")
        f.write(f"Abstract: {paper['abstract']}\n")
        f.write(f"URL: {paper['url']}\n")
        f.write("\n")
