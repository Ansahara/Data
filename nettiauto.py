import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt




from sklearn import preprocessing
from sklearn import linear_model
from sklearn import datasets
from sklearn import ensemble


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_predict
from sklearn.cross_validation import train_test_split


from bs4 import BeautifulSoup

#Pyydetään hakutuloksia
response = requests.get('https://www.nettiauto.com/audi/a3?page=1')



soup = BeautifulSoup(response.content, 'html.parser')


#haetaan seuraavan sivun url
url = soup.find('a', class_ = 'pageNavigation next_link').get('href')
url = 'https://www.nettiauto.com' + url

#luodaan lista, johon lisätään ansimäisen ja toisen sivun url
linkit = ['https://www.nettiauto.com/skoda/octavia?page=1']
linkit.append(url)

def rekursio(url, linkit):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    tarkistus = soup.find_all('span', class_ = 'disable next_link')
    if len(tarkistus) == 1:
        return
    else:
        url = soup.find('a', class_ = 'pageNavigation next_link').get('href')
        url = 'https://www.nettiauto.com' + url
        linkit.append(url)
        return rekursio(url, linkit)
    
rekursio(url, linkit)



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


for linkki in linkit:
    response = requests.get(linkki)
    soup = BeautifulSoup(response.content, 'html.parser')



    #haetaan kaikki autoihin liittyvät linkit
    car_containers = soup.find_all('a', class_ = 'childVifUrl tricky_link')
    data_box = soup.find_all('div', class_ = 'data_box')
    infos = soup.find_all('div', class_ = 'vehicle_other_info clearfix_nett')


    #käydään infoboksi läpi ja lisätään tiedot listoihin
    for info in infos:
        lis = info.find_all('li', class_ = 'bold')
        vuosimalli.append(lis[0].text)
        if len(lis) == 2:   
            mittarilukema.append(lis[1].text)
        else:
            mittarilukema.append('0')
    
    
        lis = info.find_all('li')
        if len(lis) == 4:
            kayttovoima.append(lis[2].text)
            vaihteisto.append(lis[3].text)
        else:
            kayttovoima.append('99999')
            vaihteisto.append('99999')
            
        
            
    #käydään läpi data_boxi
    for data in data_box:
        moottori.append(data.span.text)
        malli.append(data.div.text)

    #käydään auto kerraallaan läpi  linkki ja lisätään tiedot listaan
    for car in car_containers:
        links.append(car.get('href'))
        hinta.append(car.get('data-price'))
        ids.append(car.get('data-id'))
    
    

    
"""autokohtaiselta sivulta haettava tieto
tulee aika paljon http pyyntöjä.....

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
df = pd.DataFrame({'hinta':hinta, 'mittarilukema':mittarilukema, 
                   'vuosimalli':vuosimalli, 'kayttovoima':kayttovoima,
                   'moottori':moottori, 'vaihteisto':vaihteisto})
    
#muutetaan hinta numeroksi
df['hinta'] = pd.to_numeric(df['hinta'])
df['hinta'] = df['hinta'].fillna(0)

#muutetaan mittarilukema numeroksi
df['mittarilukema'] = df['mittarilukema'].str.replace('Ajamaton','0')
df['mittarilukema'] = df['mittarilukema'].str.replace('km','')
df['mittarilukema'] = df['mittarilukema'].str.replace(' ','')
df['mittarilukema'] = pd.to_numeric(df['mittarilukema'])

#muutetaan moottorin koko numeroksi
df['moottori'] = df['moottori'].str.replace('(','')
df['moottori'] = df['moottori'].str.replace(')','')
df['moottori'] = pd.to_numeric(df['moottori'])

#luodaan freimit johon tallentuu osuudet kokonaisotoksesta
vaihteisto_osuudet = df['vaihteisto'].value_counts().to_frame('maara')
kayttovoima_osuudet = df['kayttovoima'].value_counts().to_frame('maara')

#muutetaan kayttovoima numeroksi
df['kayttovoima'] = df['kayttovoima'].str.replace('Bensiini', '0')
df['kayttovoima'] = df['kayttovoima'].str.replace('Diesel', '1')
df['kayttovoima'] = df['kayttovoima'].str.replace('Hybridi', '2')
df['kayttovoima'] = df['kayttovoima'].str.replace('Kaasu', '3')
df['kayttovoima'] = pd.to_numeric(df['kayttovoima'])

#muutetaan vaihteisto numeroksi
df['vaihteisto'] = df['vaihteisto'].str.replace('Manuaali', '0')
df['vaihteisto'] = df['vaihteisto'].str.replace('Automaatti', '1')
df['vaihteisto'] = pd.to_numeric(df['vaihteisto'])


#Datan muovaamista ja visualisointia

#luodaan uusi datafreimi, jossa muut arvot vuosimallin mukaan keskiarvona
df_1 = df.groupby(['vuosimalli']).mean()
df_1['maara'] = df['vuosimalli'].value_counts()

#kuvaaja joka näyttää keskihinnan vuosimallille
df_1['hinta'].plot.bar()
plt.ylabel('hinta')
plt.show()

#kuvaaja näyttää keskikilometrit vuosimallille
df_1['mittarilukema'].plot.bar()
plt.ylabel('Km')
plt.show()

df_1['maara'].plot.bar()
plt.ylabel('maara')
plt.show()

#kuvaaaja näyttää vuosimallien osuudet kokonaisotannasta
df_1['maara'].plot.pie()
plt.show()

vaihteisto_osuudet['maara'].plot.pie()
plt.show()


kayttovoima_osuudet['maara'].plot.pie()
plt.show()




#aloitetaan koneoppimisosuus


df['vuosimalli'] = pd.to_numeric(df['vuosimalli'])

df.describe()

print(df.corr())



lr = LinearRegression()


y = df.hinta
to_remove = ['hinta']
x = df[df.columns.difference(to_remove)]
predict = cross_val_predict(lr, x, y, cv=30)

fig, ax = plt.subplots()
ax.scatter(y, predict, edgecolors=(0, 0, 0))
#ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()




df_2 = pd.DataFrame({'ennuste':predict, 'hinta':y, 'erotus':predict-y})
df['ennuste'] = predict
df['erotus'] =df['ennuste']-df['hinta']
df = df.sort_values(by='erotus', axis=0 , ascending=False)
df = df.sort_values(by='hinta', axis=0, ascending=True)





#train testing

reg = LinearRegression()

labels = df['hinta']
train1 = df.drop(['hinta'],axis=1)

x_train , x_test , y_train , y_test = train_test_split(train1 , labels , test_size = 0.10,random_state =2)
reg.fit(x_train,y_train)
reg.score(x_test,y_test)

clf = ensemble.GradientBoostingRegressor(n_estimators = 400, max_depth = 5, min_samples_split = 2,
          learning_rate = 0.1, loss = 'ls')

clf.fit(x_train, y_train)
clf.score(x_test,y_test)


df['prediction'] = clf.predict(df.drop(['hinta'],axis=1)) 
df['erotus'] =df['prediction']-df['hinta']
df = df.sort_values(by='erotus', axis=0 , ascending=False) 



prediction = pd.DataFrame({'vuosimalli':[2007], 'mittarilukema': [200000], 'moottori': [1.6], 'kayttovoima':0, 'vaihteisto':0})
hinta = clf.predict(prediction)











