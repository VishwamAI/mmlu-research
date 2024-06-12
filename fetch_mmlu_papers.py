from paperswithcode import PapersWithCodeClient

# Initialize the client with the API token
client = PapersWithCodeClient(token="your_secret_api_token")

# Fetch the list of papers related to the MMLU benchmark
papers = client.paper_list()

# Print the details of each paper
for paper in papers:
    print(f"Title: {paper['title']}")
    print(f"Authors: {paper['authors']}")
    print(f"Abstract: {paper['abstract']}")
    print(f"URL: {paper['url']}")
    print("\n")
