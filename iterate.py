

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





