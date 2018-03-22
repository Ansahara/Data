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
data_box = soup.find_all('div', class_ = 'data_box')
infos = soup.find_all('div', class_ = 'vehicle_other_info clearfix_nett')



#luodaan listoja ominaisuuksille
vuosimalli = []
mittarilukema = []
kayttovoima = []
vaihteisto = []
links = []
hinta = []
ids = []
moottori = []
malli = []

#käydään infoboksi läpi ja lisätään tiedot listoihin
for info in infos:
    lis = info.find_all('li')
    i = 1
    for li in lis:
        jakojaannos = i % 4
        if jakojaannos == 1:
            vuosimalli.append(int(li.text))
        elif jakojaannos == 2:
            mittarilukema.append(li.text)
        elif jakojaannos == 3:
            kayttovoima.append(li.text)
        elif jakojaannos == 0:
            vaihteisto.append(li.text)
        i += 1
        
            
#käydään läpi data_boxi
for data in data_box:
    moottori.append(data.span.text)
    malli.append(data.div.text)

#käydään auto kerraallaan läpi  linkki ja lisätään tiedot listaan
for car in car_containers:
    links.append(car.get('href'))
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
    
        
 
# lisätään tiedot pandasin datafreimiin
df = pd.DataFrame({'id':ids, 'hinta':hinta, 'mittarilukema':mittarilukema, 
                   'vuosimalli':vuosimalli, 'kayttovoima':kayttovoima,
                   'malli':malli, 'moottori':moottori, 'vaihteisto':vaihteisto})

#muutetaan mittarilukema numeroksi
df['mittarilukema'] = df['mittarilukema'].str.replace('km','')
df['mittarilukema'] = df['mittarilukema'].str.replace(' ','')
df['mittarilukema'] = pd.to_numeric(df['mittarilukema'])

#muutetaan moottorin koko numeroksi
df['moottori'] = df['moottori'].str.replace('(','')
df['moottori'] = df['moottori'].str.replace(')','')
df['moottori'] = pd.to_numeric(df['moottori'])

#muutetaan kayttovoima numeroksi
df['kayttovoima'] = df['kayttovoima'].str.replace('Bensiini', '0')
df['kayttovoima'] = df['kayttovoima'].str.replace('Diesel', '1')
df['kayttovoima'] = pd.to_numeric(df['kayttovoima'])

#muutetaan vaihteisto numeroksi
df['vaihteisto'] = df['vaihteisto'].str.replace('Manuaali', '0')
df['vaihteisto'] = df['vaihteisto'].str.replace('Automaatti', '1')
df['vaihteisto'] = pd.to_numeric(df['vaihteisto'])



#aloitetaan koneoppimisosuus

 # Classifier to be used
clf = linear_model.LinearRegression()
print(df.corr())

X = df['mittarilukema'].values[:, np.newaxis]     # Feature data set
y = df['hinta']                         # Label data set

classifier = clf.fit(X, y)

plt.scatter(X, y, color='g')
plt.plot(X, classifier.predict(X), color='y')

plt.show()
  