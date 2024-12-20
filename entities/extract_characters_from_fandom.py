from bs4 import BeautifulSoup
import requests
import json

url = 'https://wot.fandom.com/wiki/Category:Women'
response = requests.get(url)
html_content = response.content

soup = BeautifulSoup(html_content, 'html.parser')

# Find all <li> elements with the class 'category-page__member'
list_items = soup.find_all('li', class_='category-page__member')

# Extract the title value from each <a> element
titles = set()
for li in list_items:
    a_tag = li.find('a')
    if a_tag and 'title' in a_tag.attrs:
        titles.add(''.join(c for c in a_tag['title'].lower() if c.isalpha() or c.isspace()))

# Load existing JSON file
with open('./entities/wheel_of_time/seed_entities.json', 'r') as file:
    data = json.load(file)

# Add titles to the 'people' key
if 'people' not in data:
    data['people'] = []

titles = [[title] for title in list(titles)]

data['people'].extend(titles)

# Save updated JSON file
with open('./entities/wheel_of_time/seed_entities.json', 'w') as file:
    json.dump(data, file, indent=4)

# Print the extracted titles
for title in titles:
    print(title)