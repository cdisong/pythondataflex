import pandas 
import datetime, warnings, scipy 
import numpy
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

url = "/Users/christine/Desktop/python/flights.csv"
names = ['YEAR', 'MONTH', 'DAY', 'DAY_OF_WEEK', 'FLIGHT_NUMBER', 'TAIL_NUMBER', 'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT', 'SCHEDULED_DEPARTURE', 'DEPARTURE_TIME', 'DEPARTURE_DELAY', 'TAXI_OUT', 'WHEELS_OFF', 'SCHEDULED_TIME', 'ELAPSED_TIME', 'AIR_TIME', 'DISTANCE', 'WHEELS_ON', 'TAXI_IN', 'SCHEDULED_ARRIVAL', 'ARRIVAL_TIME', 'ARRIVAL_DELAY', 'DIVERTED', 'CANCELLED', 'CANCELLATION', 'AIR_SYSTEM_DELAY', 'SECURITY_DELAY', 'AIRLINE_DELAY', 'LATE_AIRCRAFT_DELAY', 'WEATHER_DELAY']



['YEAR', 'MONTH', 'DAY', 'DAY_OF_WEEK', 'FLIGHT_NUMBER', 'TAIL_NUMBER', 'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT', 'SCHEDULED_DEPARTURE', 'DEPARTURE_TIME',
 'DEPARTURE_DELAY', 'TAXI_OUT', 'WHEELS_OFF', 
 'SCHEDULED_TIME', 'ELAPSED_TIME', 'AIR_TIME', 'DISTANCE', 
 'WHEELS_ON', 'TAXI_IN', 'SCHEDULED_ARRIVAL', 'ARRIVAL_TIME', 
 'ARRIVAL_DELAY', 'DIVERTED', 'CANCELLED', 'CANCELLATION', 
 'AIR_SYSTEM_DELAY', 'SECURITY_DELAY', 'AIRLINE_DELAY', 'LATE_AIRCRAFT_DELAY', 'WEATHER_DELAY']


dataset = pandas.read_csv(url, names=names, dtype={ 'YEAR': int,  'MONTH': int, 'DAY': int, 'DAY_OF_WEEK': int, 'AIRLINE': str, 'FLIGHT_NUMBER': int, 'TAIL_NUMBER': object, 'ORIGIN_AIRPORT': str, 'DESTINATION_AIRPORT': str, 'SCHEDULED_DEPARTURE': int, 'DEPARTURE_TIME': int, 'DEPARTURE_DELAY': int, 'TAXI_OUT': int, 'WHEELS_OFF': int, 'SCHEDULED_TIME': int, 'ELAPSED_TIME': int, 'AIR_TIME': int, 'DISTANCE': int, 'WHEELS_ON': int, 'TAXI_IN': int, 'SCHEDULED_ARRIVAL': int, 'ARRIVAL_TIME': int, 'ARRIVAL_DELAY': int, 'DIVERTED': int, 'CANCELLED': int, 'CANCELLATION': int, 'AIR_SYSTEM_DELAY': int, 'SECURITY_DELAY': int, 'AIRLINE_DELAY': int, 'LATE_AIRCRAFT_DELAY': int, 'WEATHER_DELAY': int })



url = "/Users/christine/Desktop/python/jan2017.csv"

database = pandas.read_csv(url, low_memory=False)
>>> database['DATE'] = pandas.to_datetime(database[['YEAR', 'MONTH', 'DAY']])

##manually changed jan2017.csv
#  "DAY_0F_WEEK " TO "DAY", 
## "CRS_DEP_TIME" TO "SCHEDULED_DEPARTURE"
#_________________________________________________________
# Function that convert the 'HHMM' string to datetime.time
def format_heure(chaine):
    if pandas.isnull(chaine):
        return numpy.nan
    else:
        if chaine == 2400: chaine = 0
        chaine = "{0:04d}".format(int(chaine))
        heure = datetime.time(int(chaine[0:2]), int(chaine[2:4]))
        return heure
#_____________________________________________________________________
# Function that combines a date and time to produce a datetime.datetime
def combine_date_heure(x):
    if pandas.isnull(x[0]) or pandas.isnull(x[1]):
        return numpy.nan
    else:
        return datetime.datetime.combine(x[0],x[1])
#_______________________________________________________________________________
# Function that combine two columns of the dataframe to create a datetime format
def create_flight_time(df, col):    
    liste = []
    for index, cols in df[['DATE', col]].iterrows():    
        if pandas.isnull(cols[1]):
            liste.append(numpy.nan)
        elif float(cols[1]) == 2400:
            cols[0] += datetime.timedelta(days=1)
            cols[1] = datetime.time(0,0)
            liste.append(combine_date_heure(cols))
        else:
            cols[1] = format_heure(cols[1])
            liste.append(combine_date_heure(cols))
    return pandas.Series(liste)


database['SCHEDULED_DEPARTURE'] = create_flight_time(database, 'SCHEDULED_DEPARTURE')
database['DEPARTURE_TIME'] = database['DEPARTURE_TIME'].apply(format_heure)
database['SCHEDULED_ARRIVAL'] = database['SCHEDULED_ARRIVAL'].apply(format_heure)
database['ARRIVAL_TIME'] = database['ARRIVAL_TIME'].apply(format_heure)
# #__________________________________________________________________________
# df.loc[:5, ['SCHEDULED_DEPARTURE', 'SCHEDULED_ARRIVAL', 'DEPARTURE_TIME',
#              'ARRIVAL_TIME', 'DEPARTURE_DELAY', 'ARRIVAL_DELAY']]