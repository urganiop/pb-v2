from urllib.request import urlopen, Request
from bs4 import BeautifulSoup
import pandas as pd


def pasrse_ezfz_answer_page(question_link):
    req = Request(question_link, headers={'User-Agent': 'Chrome/5.0'})
    html_page = urlopen(req).read()
    soup = BeautifulSoup(html_page, 'html.parser')
    code_with_solution = soup.find("div", {"itemprop": "articleBody"})

    empty = []
    question = []
    answer = []
    order = [empty, question, answer, answer, empty]
    n = 0
    for i, tag in enumerate(code_with_solution):
        if tag.name == 'h2':
            if n < 4:
                n += 1
        if tag.name == 'p':
            order[n].append(tag)

    return question, answer


def parse_ezfz_site(initial_link):
    req = Request(initial_link, headers={'User-Agent': 'Chrome/5.0'})
    html_page = urlopen(req).read()
    soup = BeautifulSoup(html_page, 'html.parser')
    candidates = soup.find_all("p")

    linklist = []
    for candidate in candidates:
        if len(candidate.find_all('a')) > 5:
            for a in candidate.find_all('a'):
                linklist.append(initial_link + a.get('href'))
    return linklist


def main_parser(mainl, subl):
    qa_list = []
    for sub in subl:
        link = mainl + sub
        for task_link in parse_ezfz_site(link):
            q, a = pasrse_ezfz_answer_page(task_link)
            qa_list.append([q, a])
            print('task done')

    df = pd.DataFrame(qa_list, columns=['question', 'answer'])
    df.to_csv('qa.csv', sep=';')


sublinks = ['kinematika/'
             'statika/',
             'molekulyarnaya-fizika/',
             'termodinamika/',
             'elektrostatika/',
             'postoyannyj-tok/']

mainlink = 'http://easyfizika.ru/zadachi/'

main_parser(mainlink, [sublinks[2]])



