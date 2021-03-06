#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 18:00:37 2020

@author: Felipe A C Pereira
"""

import numpy as np
import pandas as pd
from model_SIR import SIR_BETAS
import multiprocessing as mp

gen_plots = False #If you want to see the fits, change to True
stand_error = True #If false, uses the non-normalized residuals
nproc = 1 #The number of cores to use.
nrand = 16 # number of (random) initial conditions to run the LS code.
pred_days = 7 #Number of days to predict in the future.
min_casos = 150 # minimum of cases to the city be fitted
min_dias = 15 # minimum of days with cases to the city be fitted.

if gen_plots:
    import matplotlib.pyplot as plt
    plt.close('all')


#Convert the parameters from the fitted model to the summary format
def create_summary(model, nbetas, ibgeid):
    temp = dict()
    temp['ibgeID'] = ibgeid
    temp['I0'] = model.pars_opt_ls['x0']
    temp['gamma'] = model.pars_opt_ls['gamma']
    temp['t0'] = model.t[0]
    model.pars_opt_ls['tcut'].sort()
    for j in range(nbetas):
        temp['beta_{}'.format(j)] = model.pars_opt_ls['beta'][j]
    for j in range(nbetas-1):
        temp['tchange_{}'.format(j)] = model.pars_opt_ls['tcut'][j]
    return temp


#Generate the output dataframe from the fitted data 
def create_output(model, data, pred_days, ibgeid, Ne):
    tts, Y, mY = model.predict(t=np.r_[model.t, model.t[-1]+np.arange(1,1+pred_days)],
                               coefs='LS', model_output=True)
    mY = Ne * mY
    filler = np.nan*np.ones(pred_days)
    temp = dict()
    time = pd.to_datetime( 2020000 + tts, format='%Y%j')
    temp['date'] = ["{:04}-{:02}-{:02}".format(y,m,d) for y,m,d in zip(time.year, time.month, time.day)]
    temp['ibgeID'] = ibgeid
    temp['city'] =  data.city.to_list()[0]
    temp['state'] = data.state.to_list()[0]
    temp['newCases'] = np.r_[data['newCases'].to_numpy(), filler]
    temp['mortes'] = np.r_[data['deaths'].to_numpy(), filler]
    temp['TOTAL'] = np.r_[data['totalCases'].to_numpy(), filler]
    temp['totalCasesPred'] = Y
    temp['residuo_quadratico'] = np.r_[model._residuals(model.pos_ls)**2, filler]
    temp['res_quad_padronizado'] = np.r_[model._residuals(model.pos_ls, stand_error=True)**2,
                                        filler]
    temp['suscetivel'] = mY[:,0]
    temp['infectado'] = mY[:,1]
    temp['recuperado'] = mY[:,2]
    return pd.DataFrame(temp) 


#A simple function to remove inconsistent data points, it only works
#with individual points
def filter_inconsistent_data(data):
    dat = data.copy()
    dat = dat.reset_index(drop=True)
    corr_counts = dat['totalCases'].to_numpy()
    dcounts = np.diff(corr_counts)
    nerrs = np.flatnonzero(dcounts < 0)
    if len(corr_counts) >= 3:
        for nerr in nerrs:
            if nerr == len(dcounts)-1:
                if corr_counts[-1] < corr_counts[-3]:
                    corr_counts[-1] = 0
                else:
                    corr_counts[-2] = 0
            else:
                if corr_counts[nerr] > corr_counts[nerr+2]:
                    corr_counts[nerr] = 0
                else:
                    corr_counts[nerr+1] = 0
    dat['totalCases'] = corr_counts
    return dat[dat['totalCases'] > 0]


#Load population data
pops = pd.read_csv('../data/populacao_municipios_ibge_20200622.csv')
pops['ibgeID'] = 100000 * pops['COD. UF'] + pops['COD. MUNIC']

#Load epidemic data
url = 'https://raw.githubusercontent.com/wcota/covid19br/master/cases-brazil-cities-time.csv'
data = pd.read_csv(url)
data['date'] = pd.to_datetime(data['date'], yearfirst=True)
data['DayNum'] = data['date'].dt.dayofyear
cidades = data.ibgeID.unique()


#Prepare output dataFrames
nbetas_max = 3
columns = ['DayNum', 'totalCases', 'newCases', 'deaths', 'city', 'state']
outp_par = {'ibgeID':[], 'gamma':[], 'I0':[], 't0':[]}
for i in range(nbetas_max):
    outp_par['beta_{}'.format(i)] = []
for i in range(nbetas_max-1):
    outp_par['tchange_{}'.format(i)] = []
outp_par = pd.DataFrame(outp_par)

outp_data = pd.DataFrame({'UF':[], 'city':[], 'date':[], 'ibgeID':[],  'newCases':[],   'mortes':[],
                         'TOTAL':[], 'totalCasesPred':[], 'residuo_quadratico':[],
                         'res_quad_padronizado':[],	'suscetivel':[],
                         'infectado':[], 'recuperado':[]})

#Create a list with all cities
cities_list = [data[data["ibgeID"] == cidade] for cidade in cidades]


not_computed = pd.DataFrame({"city":[], "ibgeID":[]})

def run_model(data):
    cidade = data["ibgeID"].iloc[0]
    data_fil = data[columns]
    data_fil = data_fil[data_fil['totalCases'] > 0]
    data_fil = data_fil.sort_values(by='DayNum')
    data_fil = filter_inconsistent_data(data_fil)
    if (data_fil['totalCases'].max() > min_casos) and (len(data_fil) > min_dias) and (cidade > 1e6):
        Ne = pops[pops['ibgeID'] == cidade][' POPULAÇÃO ESTIMADA '].to_list()[0]
        if len(data_fil) >= 30:
            nbetas = 3
        else:
            nbetas = 2
        try:
            model = SIR_BETAS(Ne, nproc)
            model.fit_lsquares(data_fil['totalCases'].to_numpy(), data_fil['DayNum'].to_numpy(),
                               nbetas=nbetas, stand_error=True, nrand=nrand)
            temp = create_summary(model, nbetas, cidade)
            parameters = outp_par.append(temp, ignore_index = True)
            results_df = pd.concat([outp_data,create_output(model, data_fil, pred_days,\
                                cidade, Ne)], sort=False)
            return(results_df)
        except:
            new_temp = pd.DataFrame({"city": data_fill["city"].loc[0], "ibgeID": cidade})
            global not_computed
            not_computed.append(new_temp)
            
#print(run_model(cities_list[0]))
pool = mp.Pool(32)
results = pool.map(run_model, cities_list)
pool.close()
pool.join()
data_result = pd.concat(results)

data_result.to_csv('../results/municipio.csv', index=False)
not_computed.to_csv("../results/not_computed.csv", index = False)

        




