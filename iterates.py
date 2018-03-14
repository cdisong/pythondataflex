

import pandas as pd 

jan = "/Users/christine/Desktop/air/janair.csv"
feb = "/Users/christine/Desktop/air/febair.csv"
mar = "/Users/christine/Desktop/air/marair.csv"
apr = "/Users/christine/Desktop/air/aprair.csv"
may = "/Users/christine/Desktop/air/mayair.csv"
jun = "/Users/christine/Desktop/air/junair.csv"
jul = "/Users/christine/Desktop/air/julair.csv"
aug = "/Users/christine/Desktop/air/augair.csv"
sep = "/Users/christine/Desktop/air/sepair.csv"
oct = "/Users/christine/Desktop/air/octair.csv"
nov = "/Users/christine/Desktop/air/novair.csv"
dec = "/Users/christine/Desnektop/air/decair.csv" 

a = pd.read_csv(jan, low_memory=False) 
b = pd.read_csv(feb, low_memory=False)
b = b.reset_index(drop=True) 

c = pd.read_csv(mar, low_memory=False) 
c = c.reset_index(drop=True) 

d = pd.read_csv(apr, low_memory=False) 
d = d.reset_index(drop=True) 

e = pd.read_csv(may, low_memory=False) 
e = e.reset_index(drop=True) 

f = pd.read_csv(jun, low_memory=False)
f = f.reset_index(drop=True) 
 
g = pd.read_csv(jul, low_memory=False) 
g = g.reset_index(drop=True) 

h = pd.read_csv(aug, low_memory=False) 
h = h.reset_index(drop=True) 

i = pd.read_csv(sep, low_memory=False) 
i = i.reset_index(drop=True) 

j = pd.read_csv(oct, low_memory=False)
j = j.reset_index(drop=True) 

k = pd.read_csv(nov, low_memory=False) 
k = k.reset_index(drop=True) 

l = pd.read_csv(dec, low_memory=False) 
l = l.reset_index(drop=True) 


airlines = pd.concat([a, b, c, d, e, f, g, h, i, j, k, l], axis=0, ignore_index=True)

columns = ['ORIGIN_AIRPORT_ID', 'ORIGIN']

airlines = airlines[columns]

airlines = airlines.drop_duplicates(keep='first')

import csv 







