import requests
import pandas as pd
import numpy as np
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


from bs4 import BeautifulSoup

#Pyydetään hakutuloksia
response = requests.get('https://www.nettiauto.com/audi/a3?sortCol=enrolldate&ord=DESC')


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
df = pd.DataFrame({'hinta':hinta, 'mittarilukema':mittarilukema, 
                   'vuosimalli':vuosimalli, 'kayttovoima':kayttovoima,
                   'moottori':moottori, 'vaihteisto':vaihteisto})

#muutetaan mittarilukema numeroksi
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
df['kayttovoima'] = pd.to_numeric(df['kayttovoima'])

#muutetaan vaihteisto numeroksi
df['vaihteisto'] = df['vaihteisto'].str.replace('Manuaali', '0')
df['vaihteisto'] = df['vaihteisto'].str.replace('Automaatti', '1')
df['vaihteisto'] = pd.to_numeric(df['vaihteisto'])

"""
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

#kuvaaaja näyttää vuosimallien osuudet kokonaisotannasta
df_1['maara'].plot.pie()
plt.show()

vaihteisto_osuudet['maara'].plot.pie()
plt.show()


kayttovoima_osuudet['maara'].plot.pie()
plt.show()
"""

#aloitetaan koneoppimisosuus


array = df.values 
X = array[:,0:5]
y = array[:,0]
validation_size = 0.20
seed = 7
scoring = 'accuracy'

X_train, X_validation, y_train, y_validation = model_selection.train_test_split(
        X, y, test_size=validation_size, random_state=seed)




# Spot Check Algorithms
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
# evaluate each model in turn
results = []
names = []
for name, model in models:
	kfold = model_selection.KFold(n_splits=10, random_state=seed)
	cv_results = model_selection.cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)
    
    
# Compare Algorithms
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()



# Make predictions on validation dataset
lr = LogisticRegression()
lr.fit(X_train, y_train)
predictions = lr.predict(X_validation)
print(accuracy_score(y_validation, predictions))
print(confusion_matrix(y_validation, predictions))
print(classification_report(y_validation, predictions))









