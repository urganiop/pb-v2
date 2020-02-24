import requests
from itertools import cycle
import traceback
import requests
import pandas as pd
from tbselenium.tbdriver import TorBrowserDriver
from lxml.html import fromstring
# def get_proxies():
#     url = 'https://free-proxy-list.net/'
#     response = requests.get(url)
#     parser = fromstring(response.text)
#     proxies = set()
#     for i in parser.xpath('//tbody/tr')[:10]:
#         if i.xpath('.//td[7][contains(text(),"yes")]'):
#             #Grabbing IP and corresponding PORT
#             proxy = ":".join([i.xpath('.//td[1]/text()')[0], i.xpath('.//td[2]/text()')[0]])
#             proxies.add(proxy)
#     return proxies
#
# #If you are copy pasting proxy ips, put in the list below
# #proxies = ['121.129.127.209:80', '124.41.215.238:45169', '185.93.3.123:8080', '194.182.64.67:3128', '106.0.38.174:8080', '163.172.175.210:3128', '13.92.196.150:8080']
# #proxies = get_proxies()
# proxy_pool = cycle(proxies)

from selenium import webdriver
from selenium.webdriver.firefox.firefox_profile import FirefoxProfile
from selenium.webdriver.firefox.firefox_binary import FirefoxBinary
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities

binary = FirefoxBinary(r"D:\Рабочий стол\Tor Browser\Browser\firefox.exe")
profile = FirefoxProfile(r"D:\Рабочий стол\Tor Browser\Browser\TorBrowser\Data\Browser\profile.default")
executable_path = 'D:\drivers\geckodriver.exe'


options = Options()
options.binary = binary
cap = DesiredCapabilities().FIREFOX
cap["marionette"] = True #optional
driver = webdriver.Firefox(firefox_options=options, capabilities=cap, executable_path="D:\\drivers\\geckodriver.exe")
driver.get("http://google.com/")
print("Headless Firefox Initialized")
driver.quit()

#html_page = requests.get('https://znanija.com/task/1417451', proxies={'https': '116.90.229.186'})
#html_page = requests.get('https://google.com', proxies={'https': '94.232.11.178:35106'})
# print(html_page)
def update_vpn_data():
    vpn_data = requests.get('http://www.vpngate.net/api/iphone/').text.replace('\r', '')
    servers = [line.split(',') for line in vpn_data.split('\n')]
    frame = pd.DataFrame()
    for s in servers[1:20]:
        print(s)

#update_vpn_data()
#try:
#vpn_data = requests.get('http://www.vpngate.net/api/iphone/')#.text.replace('\r', '')

#     servers = [line.split(',') for line in vpn_data.split('\n')]
#     labels = servers[2]
#     print(labels)
# #     labels[0] = labels[0][1:]
# #     servers = [s for s in servers[2:] if len(s) > 1]
# except:
#     print('Cannot get VPN servers data')
#     exit(1)
# #
# desired = [s for s in servers if country.lower() in s[i].lower()]
# found = len(desired)
# print('Found ' + str(found) + ' servers for country ' + country)
# if found == 0:
#     exit(1)
#
# supported = [s for s in desired if len(s[-1]) > 0]
# print(str(len(supported)) + ' of these servers support OpenVPN')
# # We pick the best servers by score
# winner = sorted(supported, key=lambda s: float(s[2].replace(',','.')), reverse=True)[0]
# print(winner)


