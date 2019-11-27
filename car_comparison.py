#!/usr/bin/env python
# coding: utf-8

# In[2]:


from IPython import get_ipython
try:
    cfg = get_ipython().config
    get_ipython().system('jupyter nbconvert --to script Autovergleich.ipynb')
    print('Converting')
     #-*- coding: utf-8 -*-
    get_ipython().magic('reset -sf')
    print("\033[H\033[J")
    on_jup = True #cfg['IPKernelApp']['parent_appname'] == 'ipython-notebook'
except Exception:
    on_jup = False
    print('Not running in jupyter')


# In[2]:


import requests
from bs4 import BeautifulSoup
import pandas as pd
from random import randint
import time
import re
import os
from os import chdir, path
import sys
from datetime import datetime
import math
import numpy as np
import seaborn as sns
from pprint import pprint
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score
from IPython.core.debugger import set_trace
# use set_trace() to interrupt for debugging


# In[3]:


def CleanData(filein,fileout):
    
    # load table
    df = pd.read_csv(filein,sep=';')
    
    # split and reformat manufacturer and model
    def getMan(c):
        return c.rsplit(" ")[0].upper()
    def getMod(c):
        c = re.sub("elektro|Elektro|electro|Electro|electric|Electric", "e", str(c))
        return ' '.join(c.rsplit(" ")[1:]).lower()
    df['Manufacturer'] = df['Modell'].map(getMan)
    df['Modell'] = df['Modell'].map(getMod)

    # extract digits
    def getDigits(c):
        try:
            return int(''.join(i for i in c if i.isdigit()))
        except Exception:
            return None
    df['Preis'] = df['Preis'].map(getDigits)
    df['Neupreis'] = df['Neupreis'].map(getDigits)
    df['km'] = df['km'].map(getDigits)
    df['Leergewicht'] = df['Leergewicht'].map(getDigits)
    df['PS'] = df['PS'].map(getDigits)
    
    # compute age in days
    def convD1(d1):
        return datetime.strptime(d1, '%Y-%m-%d %H:%M:%S.%f')
    def convD2(d1,d2):
        try:
            return datetime.strptime(d1, '%m.%Y')
        except Exception:
            return d2
    df['Datum'] = df['Datum'].map(convD1)
    df['Inverkehrsetzung_dt'] = df.apply(lambda x: convD2(x['Inverkehrsetzung'], x['Datum']), axis=1)
    df['days'] = df['Datum'].sub(df['Inverkehrsetzung_dt'], axis=0) / np.timedelta64(1, 'D')
    df['days'] = df['days'].astype(int)
    
    # remove unused columns
    valColumns = ['url','Datum','Preis','days','km','Manufacturer','Modell','Inverkehrsetzung','Inverkehrsetzung_dt','Klasse','Fahrzeugart','Aussenfarbe',
                  'Türen','Sitze','Innenfarbe','PS','Leergewicht','Neupreis']
    df = df.loc[:,valColumns]
    
    # remove car if it is older than 10 years    
    df = df[df.days < 10*365]
    # remove car if it drove more than 100'000 km
    df = df[df.km < 100000]
    # remove car if costs more than 50'000 CHF
    df = df[df.Preis < 50000]
    # remove cars from a manufacturer with less than 10 cars
    df.groupby("Manufacturer").filter(lambda x: len(x) > 10)
    
    # save to csv
    df.to_csv(fileout, sep=';', encoding='utf-8')


# In[4]:


def getCars(URLs,filein):

    if os.path.exists(filein):
        # load table with cars
        df = pd.read_csv(filein,sep=';')
        # remove existing entries
        URLs = set(URLs - set(df.url))
    else:
        print("Creating new list")
        df = pd.DataFrame(columns=[''])
        
    print(str(len(URLs)) + ' new cars found')
    
    row = len(df)-1
    for ii,item in enumerate(URLs):
        print(item)
        contnt = requests.get(item)
        soup = BeautifulSoup(contnt.content, 'html.parser')
        
        #print(soup.prettify())
        #with open("item.txt", "w") as text_file:
        #    text_file.write(item + '\n')
        #    text_file.write(soup.prettify())

        # get all details
        try:
            dt = {}
            dt['url'] = item
            prp = soup.get_text().split('\"value\":\"')
            for pp in prp:
                if len(pp.split('\"label\":\"')) > 1:
                    f00 = pp.split('\"')
                    if f00[0] != '':
                        dt[f00[4]] = f00[0]
        
            titl = prp[0].split('title')[1].split('\"')[2]
            dt['Modell'],dt['Klasse'] = [i.strip().replace(")","").replace("(","") for i in titl.rsplit("(",1)]
            dt['Preis'] = prp[1].split('\"')[0]
            dt['Datum'] = str(datetime.today())
            
            row = row + 1
            for key in dt.keys():
                df.loc[row,key] = dt[key]

            df_export = df.copy()
            df_export = df_export.drop(df_export.columns[df_export.columns.str.contains('unnamed',case = False)],axis = 1)
            df_export.to_csv(filein, sep=';', encoding='utf-8')
        except Exception:
            print("Car does not exist.")


# In[5]:


def getURLs():
    advert_age = 2
    print('Checking pages from the last ' + str(advert_age) + ' days.')
    pagerange = range(1,sys.maxsize)
    URLs = set()
    for page in pagerange:
        if page == 1:
            contnt = requests.get('https://www.autoscout24.ch/de/autos/alle-marken?fuel=16&age=%s&page=%s&vehtyp=10' % (advert_age,page))
            #contnt = requests.get('https://www.autoscout24.ch/de/autos/alle-marken?fuel=16&page=%s&st=1&vehtyp=10' % (page))
            soup = BeautifulSoup(contnt.content, 'html.parser')
            time.sleep(5)
        
        contnt = requests.get('https://www.autoscout24.ch/de/autos/alle-marken?fuel=16&age=%s&page=%s&vehtyp=10' % (advert_age,page))
        soup = BeautifulSoup(contnt.content, 'html.parser')

        s = str(soup)
        idx1 = s.find('listingIdList')
        idx2 = s.find('vehicleGone')
        
        sx = s[idx1:idx2].split(',')
        for e in sx:
            m = ''.join(i for i in e if i.isdigit())
            if m != "":
                URLs.add('http://www.autoscout24.ch/%s' % m)
        
        if sx[0] == '':
            print('Last page reached. Stopping')
            break
        else:
            print('Page ' + str(page))

    return URLs


# In[6]:


def PlotStats(fileout):
    
    # load cleaned data table as data frame
    df = pd.read_csv(fileout,sep=';')
    
    if not os.path.exists('plot'):
        os.mkdir('plot')
                    
    M = df['Modell'].unique()
    classnames, indices = np.unique(M, return_inverse=True)
    #pprint(classnames[indices])
    ManList = df['Manufacturer'].unique()
    
    # plot some distributions across all models
    cVars = [["Preis"],["km"],["days"]]
    for cVar in cVars:
        ax = sns.distplot(df[cVar])
        plt.xlabel(cVar[0], fontsize=18)
        plt.show()
        fname = 'plot/dist_' + cVar[0] + '.png'
        fig = ax.get_figure()
        fig.savefig(fname,bbox_inches='tight')
        plt.close('all')
                
    for ci,cVar2 in enumerate(cVars):
        for cVar1 in cVars[ci:]:
            if cVar1 != cVar2:
                ax = sns.scatterplot(x=cVar1[0], y=cVar2[0], hue='Manufacturer',data=df)
                plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
                plt.show()
                fname = 'plot/scatter_' + cVar1[0] + '_' + cVar2[0] + '.png'
                fig = ax.get_figure()
                fig.savefig(fname,bbox_inches='tight')
                plt.close('all')
                
                f, axes = plt.subplots(ncols=3, nrows=math.ceil(len(ManList)/3), figsize=(16, 18), facecolor='w', edgecolor='k')
                f.subplots_adjust(hspace = .5, wspace=.3)
                axes = axes.ravel()
                for i,m in enumerate(ManList):
                    if np.sum(df['Manufacturer'] == m) > 2:
                        sns.regplot(x=cVar1[0], y=cVar2[0], data=df[df['Manufacturer'] == m],ax=axes[i], order = 1);
                        axes[i].set_title(m)
                    else:
                        axes[i].set_title(m + ' (less than 2 data points)')
                plt.show()
                fname = 'plot/regplot_' + cVar1[0] + '_' + cVar2[0] + '.png'
                f.savefig(fname,bbox_inches='tight')
                plt.close('all')


# In[7]:


def getStats(fileout):
    
    folderpath = os.path.dirname(os.path.abspath(fileout))
    
    # load cleaned data table as data frame
    df = pd.read_csv(fileout,sep=';')
    df_copy = df.copy()
    
    # pick relevant columns
    valColumns = ['Preis',
     'Manufacturer',
     'days',
     'km']
    df = df[valColumns]
    
    # create dummy variables for the categorical regressor "Modell"
    df = pd.get_dummies(df, columns = ["Manufacturer"])

    # remove all regressors with only 1 valid data point
    df = df[[col for col in df if df[col].sum() > 1]]
    df.sum()
    
    # setup a linear model
    model = LinearRegression()
    
    Y = df[["Preis"]].values
    X = df.drop(labels = ["Preis"], axis = 1).values
    
    # train/test
    X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size = 0.25)
    model.fit(X_train,y_train)
    yTestPredicted = model.predict(X_test)
    r2 = r2_score(y_test,yTestPredicted)
    print(model.score(X_test,y_test))
    #print("Intercept: " + str(model.intercept_))
    #print("Coef: " + str(model.coef_))
    
    # train/test n-fold cross validation
    scores = cross_val_score(model, X, Y, cv=10, scoring = "r2")
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    
    # mixed effect model
    md = smf.mixedlm("Preis ~ km + days", df_copy, groups=df_copy["Modell"])
    mdf = md.fit()
    print(mdf.summary())
    
    # find best deals (highest price difference between predicted and true)
    ndays = 21
    mindiff = 10000
    df_copy['pricePred'] = (mdf.predict(df_copy)).astype(int)
    df_copy['priceDiff'] = (df_copy.Preis - df_copy.pricePred).astype(int)
    result = df_copy.sort_values(by=['priceDiff'])
    def convD1(d1):
        return datetime.strptime(d1, '%Y-%m-%d %H:%M:%S.%f')
    result['Datum'] = result['Datum'].map(convD1)

    result['age_advert'] = (result['Datum'].sub(datetime.today(), axis=0) / np.timedelta64(1, 'D')).astype(int)
    result = result[result.priceDiff < -mindiff]
    result = result[result.age_advert > -ndays]
    result = result[result.PS > 70]
    result = result.drop(['Datum', 'Inverkehrsetzung_dt','Leergewicht','Klasse','Türen','Sitze'], axis=1)
    print('\n\n' + 70*'-' + '\nBest deals in the last ' +str(ndays) + ' days (at least ' + str(mindiff) + ' CHF difference)\n' + 70*'-' + '\n\n')
    pprint(result)
    print(folderpath + '/cars_best_deals.csv')
    result.to_csv(folderpath + '/cars_best_deals.csv', sep=';', encoding='utf-8')


# In[1]:


if not on_jup:    
    # set working directory
    pathname = path.dirname(sys.argv[0])
    if not pathname == '':
        pprint(pathname)
        chdir(pathname)
filein  = 'cars_raw.csv'
fileout = 'cars.csv'


# In[9]:


URLs = getURLs()


# In[10]:


getCars(URLs,filein)


# In[11]:


CleanData(filein,fileout)


# In[12]:


getStats(fileout)


# In[13]:


if on_jup:
    get_ipython().run_line_magic('matplotlib', 'inline')
    PlotStats(fileout)


# In[14]:


def check_status():
    dfr = pd.read_csv('cars_raw.csv',sep=';')
    dfs = pd.read_csv('cars_raw_status.csv',sep=';')

    # initiate all as online
    if not 'status' in dfr.columns:
        dfr["status"] = 'online'
    
    # replace values for those that were checked in the past
    for i,item in enumerate(dfs['url']):
        dfr.loc[dfr['url'] == item, ['status']] = dfs['status'][i]

    # check status of cars marked as online (including new entries)
    for i, (item, s) in enumerate(zip(dfr['url'], dfr["status"])):
        if s == 'online':
            contnt = requests.get(item)
            soup = BeautifulSoup(contnt.content, 'html.parser')
            if 'Fahrzeug nicht gefunden' in soup.get_text():
                dfr["status"].iloc[i] = 'offline'
                print('item: ' + str(i) + ' -> offline')
            else:
                dfr["status"].iloc[i] = 'online'
                print('item: ' + str(i) + ' -> online')

    dfr.to_csv('cars_raw_status.csv', sep=';', encoding='utf-8')

#if on_jup:
#    check_status()

