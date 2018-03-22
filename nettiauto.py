import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn import linear_model


from bs4 import BeautifulSoup

#Pyydetään hakutuloksia
response = requests.get('https://www.nettiauto.com/skoda/octavia?sortCol=enrolldate&ord=DESC')


soup = BeautifulSoup(response.content, 'html.parser')

#haetaan kaikki autoihin liittyvät linkit
car_containers = soup.find_all('a', class_ = 'childVifUrl tricky_link')

#luodaan etusivulta saataville tiedoille listat
links = []
vuosimalli = []
mittarilukema = []
hinta = []
ids = []

#käydään auto kerraallaan läpi ja lisätään tiedot listaan
for car in car_containers:
    links.append(car.get('href'))
    vuosimalli.append(int(car.get('data-year')))
    mittarilukema.append(int(car.get('data-mileage')))
    hinta.append(int(car.get('data-price')))
    ids.append(car.get('data-id'))
    
    

    
"""

#pyydetään auton sivua
response = requests.get('https://www.nettiauto.com/skoda/octavia/9349082')


soup = BeautifulSoup(response.content, 'html.parser')

#lista jonne tallennetaan hinnat
prices = []

#etsitään hinta html:stä ja lisätään se hinta listaan
price = soup.find('span', class_ = 'small_text18 bold GAPrice').text
prices.append(price)

#etsitään taulu, jossa auton tiedot
table = soup.find('table', class_ ='data_table')

#luodaan lista, johon tallennetaan taulun tiedot
lista = []

#etsitään kaikki taulussa olevat tiedot ja lisätään ne listaan
for tr in table.find_all('tr'):
    tds = tr.find_all('td')
    for td in tds:
        lista.append(td.text)
    
#luodaan listat, joihin tallennetaan halutut tiedot
vuosimalli = []
moottori = []
mittarilukema = []
vetotapa = []
vaihteisto = []

#lisätään ominaisuuksia vastaavat arvot oikeisiin listoihin
for i in range(len(lista)):
    if lista[i] == "Vuosimalli":
        vuosimalli.append(lista[i+1])
    elif lista[i] == "Moottori":
        moottori.append(lista[i+1])
    elif lista[i] == "Mittarilukema":
        mittarilukema.append(lista[i+1])
    elif lista[i] == "Vetotapa":
        vetotapa.append(lista[i+1])
    elif lista[i] == "Vaihteisto":
        vaihteisto.append(lista[i+1])
    
    



#for link in links:
#    response = requests.get(link)
    
#    soup = BeautifulSoup(response.content, 'html.parser')
    
#    price = soup.find('span', class_ = 'small_text18 bold GAPrice').text
#    prices.append(price)
        
"""
    
        
 
# Adding movies to pandas dataframe
df = pd.DataFrame({'id':ids, 'hinta':hinta, 'mittarilukema':mittarilukema, 
                   'vuosimalli':vuosimalli})



#aloitetaan koneoppimisosuus

 # Classifier to be used
clf = linear_model.LinearRegression()
print(df.corr())

X = df['vuosimalli'].values[:, np.newaxis]     # Feature data set
y = df['hinta']                         # Label data set

classifier = clf.fit(X, y)

plt.scatter(X, y, color='g')
plt.plot(X, classifier.predict(X), color='y')

plt.show()
  