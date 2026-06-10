from bs4 import BeautifulSoup

# Load your HTML content
html_content = '\n'.join(open("ordfav_Rignak _ Danbooru.html").readlines())

# Parse the HTML content
soup = BeautifulSoup(html_content, 'html.parser')

# Find all <article> tags
articles = soup.find_all('article')

# Iterate through each article and print the URL with the data-id
for article in articles:
    data_id = article.get('data-id')
    if data_id:
        print(f"https://danbooru.donmai.us/posts/{data_id}")
