import os
import re
from urllib.request import urlopen, Request
from bs4 import BeautifulSoup


DIR = os.getcwd() + r'\interchange_files'
SOLUTIONS_PATH = os.getcwd() + r'\solutions_html'
# with open(DIR + r'\valid_links.json', 'r', encoding='utf-8') as qf:
#     valid_links_list = load(qf)


def link_determin(alink):
    if re.findall('easyfizika', alink):
        print('ezfz')
        return 'ezfz'
    elif re.findall('znanija', alink):
        print('znanija')
        return 'znanija'
    elif re.findall('uchifiziku', alink):
        print('uchifiziku')
        return 'uchifiziku'
    elif re.findall('otvet.mail', alink):
        print('otvet')
        return 'otvet'
    else:
        return False



def get_page(vl):
    req = Request(vl, headers={'User-Agent': 'Chrome/5.0'})
    try:
        html_page = urlopen(req).read()
    except:
        return None
    return html_page


def ez_fz_text_from_html(body):
    soup = BeautifulSoup(body, 'html.parser')
    code_with_solution = soup.find("div", {"class": "entry-content"})
    return str(code_with_solution)


def znj_text_from_html(body):
    soup = BeautifulSoup(body, 'html.parser')
    code_with_solution = soup.find("div", {"class": "brn-kodiak-answer__content"})
    image = soup.select('div.brn-main-attachment > img')
    if image:
        code_with_solution = str(code_with_solution) + "<img src='{}'>".format(image[0]['src'])
    else:
        code_with_solution = str(code_with_solution)
    return code_with_solution


def uf_text_from_html(body):
    soup = BeautifulSoup(body, 'html.parser')
    code_with_solution = soup.find("div", {"class": "comment-data"})
    return str(code_with_solution)


def omr_text_from_html(body):
    soup = BeautifulSoup(body, 'html.parser')
    code_with_solution = soup.select('div.answer > div.atext > a')[0]
    code_with_solution = re.sub('<a href=([\s\S]*?)>','<p>', str(code_with_solution))
    code_with_solution = re.sub('</a>', '</p>', code_with_solution)
    print('omr')
    print(code_with_solution)
    return code_with_solution


def la_main(valid_links_list):
    print('длина списка: ', len(valid_links_list))
    if valid_links_list:
        for link in valid_links_list:
            if re.findall('easyfizika', link) or re.findall('znanija', link) or re.findall('uchifiziku', link) or re.findall(r'otvet.mail', link):
                which_link = link_determin(link)
                webpage = get_page(link)

                if which_link == 'ezfz':
                    code = ez_fz_text_from_html(webpage)
                elif which_link == 'znanija':
                    code = znj_text_from_html(webpage)
                elif which_link == 'uchifiziku':
                    code = uf_text_from_html(webpage)
                elif which_link == 'otvet':
                    code = omr_text_from_html(webpage)
                else:
                    code = None

                if code:
                    print('code: {}'.format(type(code)))
                    return code
                print('+')
        return '<p>Ответа нет</p>'
    else:
        code = '<p>Ответа нет</p>'
        return code