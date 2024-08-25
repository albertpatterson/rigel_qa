import requests
from bs4 import BeautifulSoup
import time
from blog_scraping.parse import get_post_data, get_classes
from util.util import write_data


def get_blog_url(category, page):
    base = f"https://blindcaveman.wordpress.com/category/{category}/"
    if page == 1:
        return base
    return f"{base}page/{page}/"


def get_all_blog_data_for_category(category):
    all_data = []
    page = 1
    while True:
        url = get_blog_url(category, page)
        print(url)
        response = requests.get(url)
        if response.url != url:
            print("break due to redirect")
            break

        if response.status_code < 200 or response.status_code >= 300:
            print("break due to request error")
            break

        soup = BeautifulSoup(response.text, "html.parser")
        divs = soup.select("#content #content-left>div")

        if "navlink" in get_classes(divs[-1]):
            divs = divs[:-1]

        all_data.extend([get_post_data(div) for div in divs])
        page += 1

        time.sleep(1)

    return all_data


categories = {
    "journal": "journal",
    "life": "life",
    "computing": "computing",
    "innovation": "innovation",
    "swimbikesleep": "swimbikesleep",
    "ham radio": "ham-radio",
    "!": "123789724",
    "health": "health",
    "word study": "word-study",
}


def collect_all_blog_data():
    all_data = {}
    for name, code in categories.items():
        all_data[name] = get_all_blog_data_for_category(code)
        time.sleep(1)

    write_data(all_data, "blog_data.pkl")
