#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 11:56:48 2020

@author: kylealden
"""


# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
Spyder Editor

gather data from covid website
"""

import pandas as pd
import requests
import datetime
#import psycopg2
#import sqlalchemy

#gather latest covid data from washington post source, write to file as latest
url = 'https://inv-covid-data-prod.elections.aws.wapo.pub/us-states-current/us-states-current-combined.csv'
#url = 'https://covid19.healthdata.org/api/metadata/location'
data = requests.get(url)
cov_data = r'C:\Users\kylea\GIS_Projects\covid_data\wapo_data_latest.csv'
file2 = open(cov_data,'w')
file2.write(data.text)
file2.close()

df_covid_states = pd.read_csv(cov_data)

#gather historical covid data from washington post source, write to file as latest
url = r'https://inv-covid-data-prod.elections.aws.wapo.pub/us-daily-historical/us-daily-historical-combined.csv'
data = requests.get(url)
cov_data_hist = r'C:\Users\kylea\GIS_Projects\covid_data\wapo_data_hist.csv'
file2 = open(cov_data_hist,'w')
file2.write(data.text)
file2.close()
df = pd.read_csv(cov_data_hist)
 
#create a daily counts column, first sort by state and date, then subtract yesterday from today,
#then make sure we only include from within a state by zeroing out any negative values
df.sort_values(by=['state','date'],inplace=True)
df.drop([ u'statePost', u'rolling7DayConfirmed', u'rolling7DayDeaths', u'testedPositive', u'testedNegative', u'tested', u'grade', u'score', u'pending', u'hospTotal', u'hospTotalIcuBeds',u'hospTotalAcuteBeds', u'hospCurrent', u'hospCurrentIcuBeds', u'hospCurrentAcuteBeds', u'hospCurrentVents'], axis=1,inplace=True)
#df.columns
df["confirmed_daily"] = df.confirmed.diff()
df["confirmed_daily"] = df["confirmed_daily"].clip(lower=0)
df.fillna(0,inplace=True)


#create a df that is just today
today = datetime.datetime.now().strftime("%Y-%m-%d") 
df_today = df[df.date == today]

#create a last 7 days df
today_min8 = (datetime.datetime.now() - datetime.timedelta(days=8)).strftime("%Y-%m-%d")
today_min1 = (datetime.datetime.now() - datetime.timedelta(days=1)).strftime("%Y-%m-%d")
#create 8 days ago to 1 day ago df
df_7days_leading = df[(df.date.astype('datetime64[ns]') > (datetime.datetime.now() - datetime.timedelta(days=8))) & (df.date.astype('datetime64[ns]') < (datetime.datetime.now() - datetime.timedelta(days=1)))] 
df_7days_leading = df_7days_leading[df.date.astype('datetime64[ns]') > (datetime.datetime.now() - datetime.timedelta(days=8))] 

#create a 7 day moving average leading up to today
df_7days_leading['date'].value_counts()
df_7_days_mv_av = df_7days_leading.groupby(['state']).confirmed_daily.mean()
df_7_days_mv_av.columns = ['state', '7d_mv_agv']
df_today = pd.merge(df_today, df_7_days_mv_av, on='state')
df_today.columns = [u'state', u'statePostal', u'updated', u'confirmed', u'deaths',
       u'recovered', u'date', u'source', u'confirmed_daily',
       u'mv_avg']

df_today.sort_values(by=['confirmed_daily', 'mv_avg'], inplace=True)


#for my state by state prediction, use basic ARIMA ('Auto Regressive Integrated Moving Average' model) 
from statsmodels.tsa.arima_model import ARIMA
#eliminate American Samoa as it has no confirmed cases
state_list = df_today[df_today.state != 'American Samoa'].state.sort_values().to_list()
forecast_lst = []

#loop through states and run a model against each state (using universal parameters)
for state in state_list:
    try:
        df_not_today = df[df.date != today]
        series = df_not_today[df_not_today.state == state].confirmed_daily
        series_az_dts =  df_not_today[df_not_today.state == 'Arizona'].date
        series_dts= pd.to_datetime(series_az_dts)
        model = ARIMA(series, order=(7,1,0),dates=series_dts)
        model_fit = model.fit(disp=0)
        forecast = int(model_fit.forecast()[0])
        data_t = [state,forecast]
        forecast_lst.append(data_t)
        #print(state)
        #print(str(forecast))
    except:
        print(state)
 
# create a dataframe of the ARIMA forecasts
df_forecast = pd.DataFrame(forecast_lst, columns=['state','forecast_cases'])


# Write out a finding of how many states have reported, how many remain, and how many cases we expect
df_exp_sum = df_today[df_today.confirmed_daily == 0].groupby(['confirmed_daily']).sum()['mv_avg']
count_state = str(len(df_today[df_today.confirmed_daily == 0]))
df_exp_sum.loc[0].astype('int').astype('str')
remaining_states = df_today[df_today.confirmed_daily == 0].state.to_list()
str(df_today.confirmed_daily.sum())
finding = ("The Washington Post has reported " + str(int(df_today.confirmed_daily.sum())) + " confirmed COVID-19 cases today. " + count_state + " states are left to report. I expect about " + (df_exp_sum.loc[0]*.9).astype('int').astype('str') + ' to ' +  (df_exp_sum.loc[0]*1.1).astype('int').astype('str') + ' more cases today. The remaining states are: ' + ', '.join(remaining_states) )
print(finding)
state_list = df_today.state.sort_values().to_list()



### Create image with every state's trends#############################3
import matplotlib.pyplot as plt


from matplotlib.pyplot import figure
figure(num=None, figsize=(14, 7), dpi=280, facecolor='w', edgecolor='k')
fig, axes = plt.subplots(nrows=14, ncols=4, sharex='col')
fig.subplots_adjust(hspace=0.4,wspace=0.1)
#fig.suptitle('COVID-19 Case Trends Per State')
#fig.text(1,1,"Red bars represent every day within 20% of a state's peak.", verticalalignment='center', horizontalalignment='center',)

for ax, st in zip(axes.flatten(), state_list):
    dates = df[df.state == st].date.to_list()
    values = df[df.state == st].confirmed_daily.to_list()
    stateP = df[df.state == st].statePostal.to_list()[1]
    mv7 = df[df.state == st].confirmed_daily.rolling(7).mean().to_list()
    #ax = x.plot.bar(x='date', y='confirmed_daily', rot=0)
    clrs = ['grey' if (x < max(values)*.8) else 'red' for x in values ]
    ax.bar(dates,values,color=clrs)
    #ax.bar(feature, bins=len(np.unique(data.data.T[0]))//2)
    ax.set_title(st, loc='center', fontdict={'fontsize': 8, 'fontweight': 'medium'}, pad=1)    
    #ax.set_title(stateP, loc='center', fontdict={'fontsize': 8, 'fontweight': 'medium'}, pad=1)
    #ax.set(title=stateP)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    #print(str(max(values)) + ' ' + st)
    mv7 = df[df.state == st].confirmed_daily.rolling(7).mean().to_list()
    #ax2 = ax.twinx()
    ax.plot(ax.get_xticks(), mv7) #linechart
    #ax2.get_yaxis().set_visible(False)

#plt.figtext(1,4,finding,fontsize=10,ha='center',wrap=True)   
fig.set_size_inches(7,10)
#fig.savefig('covid_by_stat.png', dpi=100)

import os
strFile = r'C:\Users\kylea\GIS_Projects\covid_data\covid_by_state_graphs_.png'
if os.path.isfile(strFile):
   os.remove(strFile)   # Opt.: os.system("rm "+strFile)
plt.savefig(strFile,  dpi=250)


#create a national plot
df_national = df.groupby(['date']).confirmed_daily.sum()
df_national = df_national.to_frame(name='confirmed_daily')
df_national['date'] = df_national.index
plt.rcParams['figure.figsize'] = (8, 6)
plt.rcParams['font.size'] = 10
ax1 = plt.axes()
x = df_national.date.to_list()
y = df_national.confirmed_daily.to_list()
x_axis = ax1.axes.get_xaxis()
x_axis.set_visible(False)
plt.bar(x, height=y)
#nat_plot = df_national.plot.bar(x='date', y='confirmed_daily', rot=0)

nat_file = r'C:\Users\kylea\GIS_Projects\covid_data\covid_national_.png'
if os.path.isfile(nat_file):
   os.remove(nat_file)   # Opt.: os.system("rm "+strFile)
plt.savefig(nat_file,  dpi=250)


############## Create pdf with findings, predictions, and data using reportlab ##################
df_today_gph = df_today.merge(df_forecast,how='left', on='state')
#df_today.columns
df_today_gph.drop(['statePostal','updated','confirmed','deaths','recovered','source','date'],axis=1,inplace=True)
#print(df_today.sort_values(by=['State']))

df_today_gph.sort_values(by=['state'], inplace=True)
df_today_gph.columns = ['State','Cases Today','7day Moving Average', 'ARIMA Prediction']
df_today_gph['7day Moving Average'] = df_today_gph['7day Moving Average'].astype('int')
df_today_gph['Cases Today'] = df_today_gph['Cases Today'].astype('int')
#df_today_html = df_today.sort_values(by=['State']).to_html(r'C:\Users\kylea\GIS_Projects\covid_data\covid_by_state_today.html',index=False)

from reportlab.platypus import *
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors
import numpy as np

styles = getSampleStyleSheet()
styleN = styles['Normal']
styleH = styles['Heading1']
story = []

pdf_name =r'C:\Users\kylea\GIS_Projects\covid_data\covid_by_state.pdf'
doc = SimpleDocTemplate(
    pdf_name,
    pagesize=letter,
    bottomMargin=.4 * inch,
    topMargin=.6 * inch,
    rightMargin=.8 * inch,
    leftMargin=.8 * inch)

#add finding
date_formatted = today = datetime.datetime.now().strftime("%B %d, %Y") 
H = Paragraph(('COVID-19 Cases in the US: What to expect today '+ date_formatted +'?'), styleH)
story.append(H)
P = Paragraph(finding, styleN)
story.append(P)

story.append(Spacer(1,.5*inch))
#add national trend graph
C = Paragraph(('COVID-19 Cases per day'), styleH)
story.append(C)
im = Image(nat_file,7*inch,5*inch)
story.append(im)

story.append(PageBreak())

# add state by state reporting
lista = [df_today_gph.columns[:,].values.astype(str).tolist()] + df_today_gph.values.tolist()

colwidths = 50
ts = [('ALIGN', (1,1), (-1,-1), 'CENTER'),
     ('LINEABOVE', (0,0), (-1,0), 1, colors.black),
     ('LINEBELOW', (0,0), (-1,0), 1, colors.black),
     ('FONT', (0,0), (-1,0), 'Times-Bold'),
     ('LINEBELOW', (0,-1), (-1,-1), 1, colors.black),
     ('BACKGROUND',(1,1),(-2,-2),colors.white),
     ('TEXTCOLOR',(0,0),(1,-1),colors.black),
     ('INNERGRID',  (0,0), (-1,-1), 1, colors.black),
     ('BOX', (0,0), (-1,-1), 1, colors.black)]
t1 = Table(lista, style=ts);

story.append(t1)

story.append(PageBreak())

#add image with every state
#add finding
Ti = Paragraph('COVID-19 Case Trends Per State', styleH)
story.append(Ti)
No = Paragraph("Red bars represent days where a state was within 20% of its highest daily COVID total.", styleN)
story.append(No)
im2 = Image(strFile,7*inch,9*inch)
story.append(im2)
#print to pdf
doc.build(
    story
)
