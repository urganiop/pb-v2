import re, os
from bs4 import BeautifulSoup
from bs4 import Comment
from requests import get
from lxml import html
from selenium import webdriver


DIR = os.getcwd() + r'\interchange_files'


def tag_visible(element):
    if element.parent.name in ['style', 'script', 'head', 'title', 'meta', '[document]']:
        return False
    if isinstance(element, Comment):
        return False
    return True


def text_from_html(body):
    soup = BeautifulSoup(body, 'html.parser')
    texts = soup.findAll(text=True)
    visible_texts = filter(tag_visible, texts)
    return u" ".join(t.strip() for t in visible_texts)


def get_screenshot_on_link(site_links):
    driver = webdriver.Chrome()
    driver.get(site_links[0])
    driver.save_screenshot('scrtmp/google.png')
    driver.close()


def qtl_main(text_to_find):
    compiled_text = text_to_find.strip(' ')
    compiled_text = re.sub(' +', '+', compiled_text)
    find_string = '&q={}&oq={}'.format(compiled_text, compiled_text)

    url = 'http://www.google.ru/search?'+find_string
    page = get(url)
    webpage = html.fromstring(page.content)
    print('page accessed successfully')
    div_node = webpage.xpath('//div[@id="main"]')
    if not div_node:
        raise Exception("Div does not exist")
    else:
        valid_links = []
        link_list = div_node[0].xpath('//a/@href')
        for link in link_list:
            if re.findall('http', link) and not re.findall('google', link) and not re.findall('www.youtube', link):
                clean_link = link.replace(r'/url?q=', '')
                clean_link = re.sub('&sa=\S+', '', clean_link)
                valid_links.append(clean_link)
        print('отчищено')


    return valid_links