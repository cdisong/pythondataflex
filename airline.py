import pandas as pd 

airlines = "/Users/christine/Desktop/python/airlines.csv"



airline = pd.read_csv(airlines, low_memory=False)

airline.ix[airline['Description'].str.contains(':....', regex=True)]

##remove rows that evaluates this as true 
airline = airline.ix[airline['Description'].str.contains(':....', regex=True)]

airline['IATA'] = airline['Description'].str.extract(': (...)', expand=True) 

airline.to_csv('airline.csv')

mv airline.csv /Desktop 



######
https://en.wikipedia.org/wiki/List_of_airports_in_the_United_States

from bs4 import BeautifulSoup
from urllib.request import Request, urlopen

site= "https://en.wikipedia.org/wiki/List_of_airports_in_the_United_States"
hdr = {'User-Agent': 'Mozilla/5.0'}
req = Request(site,headers=hdr)
page = urlopen(req)
soup = BeautifulSoup(page, "lxml")
print(soup)


 
table = soup.find("table", { "class" : "wikitable sortable" })
print table

for row in table.findAll("tr"):
    cells = row.findAll("td")
    #For each "tr", assign each "td" to a variable.
    if len(cells) == 4:
        area = cells[0].find(text=True)
        district = cells[1].findAll(text=True)
        town = cells[2].find(text=True)
        county = cells[3].find(text=True)