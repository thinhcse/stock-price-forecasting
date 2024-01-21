import sys, os

from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options

import time
import pandas as pd
from bs4 import BeautifulSoup

sys.path.insert(0,'/usr/lib/chromium-browser/chromedriver')
url = 'https://finance.yahoo.com/quote/%5EGSPC/history?period1=1136073600&period2=1624665600&interval=1d&filter=history&frequency=1d&includeAdjustedClose=true'

chrome_options = Options()
chrome_options.add_argument('--headless')
chrome_options.add_argument('--no-sandbox')
chrome_options.add_argument('--disable-dev-shm-usage')
driver = webdriver.Chrome('chromedriver',chrome_options=chrome_options)
driver.get(url)

key = 5655
last_size = driver.execute_script("return document.documentElement.scrollHeight")
while True:
  try:
    element = driver.find_element_by_xpath(f'//*[@id="Col1-1-HistoricalDataTable-Proxy"]/section/div[2]/table/tbody/tr[{key}]/td[1]/span')
    break
  except:
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight)")
    driver.find_element_by_tag_name('body').send_keys(Keys.END)
    time.sleep(2)
    new_size = driver.execute_script("return document.documentElement.scrollHeight")
    if new_size == last_size:
      break
    last_size = new_size

html_content = driver.page_source
driver.close()

soup = BeautifulSoup(html_content, 'lxml')

table = soup.find('table', attrs={'class': 'W(100%) M(0)'})
table_header = table.thead.find_all('tr')
table_data = table.tbody.find_all('tr')

headings = []
for th in table_header[0].find_all('th'):
  headings.append(th.text.replace('\n', ' ').strip())

data = []
for datum in table_data:
  row = [i.text for i in datum.find_all('td')]
  data.append(row)

df = pd.DataFrame(data = data, columns=headings)
df.to_csv(os.path.join(os.getcwd(), "data", 'sp500_data.csv'))

print('Succeeded!', len(table_data))