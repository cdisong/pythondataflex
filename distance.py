import pandas as pd 
jan2016 = "/Users/christine/Desktop/distance/jan.csv"
feb2016 = "/Users/christine/Desktop/distance/feb.csv"
mar2016 = "/Users/christine/Desktop/distance/mar.csv"
apr2016 = "/Users/christine/Desktop/distance/apr.csv"
may2016 = "/Users/christine/Desktop/distance/may.csv"
jun2016 = "/Users/christine/Desktop/distance/jun.csv"
jul2016 = "/Users/christine/Desktop/distance/jul.csv"
aug2016 = "/Users/christine/Desktop/distance/aug.csv"
sep2016 = "/Users/christine/Desktop/distance/sep.csv"
oct2016 = "/Users/christine/Desktop/distance/oct.csv"
nov2016 = "/Users/christine/Desktop/distance/nov.csv"
dec2016 = "/Users/christine/Desktop/distance/dec.csv"


a = pd.read_csv(jan2016, low_memory=False) 
b = pd.read_csv(feb2016, low_memory=False)
b = b.reset_index(drop=True) 

c = pd.read_csv(mar2016, low_memory=False) 
c = c.reset_index(drop=True) 

d = pd.read_csv(apr2016, low_memory=False) 
d = d.reset_index(drop=True) 

e = pd.read_csv(may2016, low_memory=False) 
e = e.reset_index(drop=True) 

f = pd.read_csv(jun2016, low_memory=False)
f = f.reset_index(drop=True) 
 
g = pd.read_csv(jul2016, low_memory=False) 
g = g.reset_index(drop=True) 

h = pd.read_csv(aug2016, low_memory=False) 
h = h.reset_index(drop=True) 

i = pd.read_csv(sep2016, low_memory=False) 
i = i.reset_index(drop=True) 

j = pd.read_csv(oct2016, low_memory=False)
j = j.reset_index(drop=True) 

k = pd.read_csv(nov2016, low_memory=False) 
k = k.reset_index(drop=True) 

l = pd.read_csv(dec2016, low_memory=False) 
l = l.reset_index(drop=True) 

