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
nproc = 1 #The number of cores to use
nrand = 20 # number of (random) initial conditions to run the LS code.
pred_days = 7 #Number of days to predict in the future.
nbetas = 3 #Number of betas to be used in the fitting routine.

if gen_plots:
    import matplotlib.pyplot as plt
    plt.close('all')

#Convert the parameters from the fitted model to the summary format
def create_summary(model, nbetas, estado):
    temp = dict()
    temp['state'] = estado
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
def create_output(model, data, pred_days, state, Ne):
    tts, Y, mY = model.predict(t=np.arange(model.t[0], model.t[-1]+1+pred_days),
                               coefs='LS', model_output=True)
    mY = Ne * mY
    filler = np.nan*np.ones(pred_days)
    temp = dict()
    time = pd.to_datetime( 2020000 + tts, format='%Y%j')
    temp['date'] = ["{:04}-{:02}-{:02}".format(y,m,d) for y,m,d in zip(time.year, time.month, time.day)]
    temp['state'] = state
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


#State Populations estimated from IBGE (2019, updated in 20200622)
pops = {'RO':	1777225,
        'AC':	881935,
        'AM':	4144597,
        'RR': 	605761,
        'PA':	8602865,
        'AP':	845731,
        'TO':	1572866,
        'MA':	7075181,
        'PI':	3273227,
        'CE':	9132078,
        'RN':	3506853,
        'PB':   4018127,
        'PE':	9557071,
        'AL':   3337357,
        'SE':   2298696,
        'BA':   14873064,
        'MG':   21168791,
        'ES':	4018650,
        'RJ':   17264943,
        'SP':   45919049,
        'PR':   11433957,
        'SC':   	7164788,
        'RS':	11377239,
        'MS':	2778986,
        'MT':   3484466,
        'GO':   7018354,
        'DF':	3015268,
        'TOTAL':210147125
}

#Load epidemic data
url = 'https://raw.githubusercontent.com/wcota/covid19br/master/cases-brazil-states.csv'
data = pd.read_csv(url)
data['date'] = pd.to_datetime(data['date'], yearfirst=True)
data['DayNum'] = data['date'].dt.dayofyear
states = data.state.unique()

#%%

#Prepare output dataFrames
columns = ['DayNum', 'totalCases', 'newCases', 'deaths']
outp_par = {'state':[], 'gamma':[], 'I0':[], 't0':[]}
for i in range(nbetas):
    outp_par['beta_{}'.format(i)] = []
for i in range(nbetas-1):
    outp_par['tchange_{}'.format(i)] = []
outp_par = pd.DataFrame(outp_par)
outp_data = pd.DataFrame({'date':[], 'state':[], 'newCases':[],   'mortes':[],
                         'TOTAL':[], 'totalCasesPred':[], 'residuo_quadratico':[],
                         'res_quad_padronizado':[],	'suscetivel':[],
                         'infectado':[], 'recuperado':[]})

#Create a list with data
list_df = [data[data["state"] == state] for state in states]


def run_model(data):
    estado = data["state"].iloc[0]
    Ne = pops[estado]
    model = SIR_BETAS(Ne, nproc)
    model.fit_lsquares(data['totalCases'].to_numpy(), data['DayNum'].to_numpy(),
                       nbetas=nbetas, stand_error=True, nrand=nrand)

    temp = create_summary(model, nbetas, estado)
    parameters = outp_par.append(temp, ignore_index=True)
    results_df = pd.concat([outp_data,create_output(model, data, pred_days,\
                                                   estado, Ne)], sort=False)

    tts, Y = model.predict(coefs = 'LS')
    return(results_df)

    
#print(run_model(list_df[0]))
pool = mp.Pool(32)
results = pool.map(run_model, list_df)
pool.close()
pool.join()
data_result = pd.concat(results)

data_result.to_csv('../results/estados.csv', index=False)



