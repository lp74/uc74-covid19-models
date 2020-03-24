#!/usr/bin/python3

# ––––––––––––––––––––––––––––––
# CORE
# ––––––––––––––––––––––––––––––
from datetime import datetime
from datetime import timedelta
from scipy import integrate, optimize
from scipy.integrate import odeint
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import json
import math
import numpy as np
import os
import re
import requests
import simplejson
import sys
import time

DATA_FOLDER = './data'
if not os.path.exists(DATA_FOLDER):
    os.makedirs(DATA_FOLDER)

# ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
# G A T H E R I N G - D A T A
# ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

COUNRTY_URL = "https://raw.githubusercontent.com/pcm-dpc/COVID-19/master/dati-json/dpc-covid19-ita-andamento-nazionale.json"
REGION_URL = "https://raw.githubusercontent.com/pcm-dpc/COVID-19/master/dati-json/dpc-covid19-ita-regioni.json"
PROV_URL = "https://raw.githubusercontent.com/pcm-dpc/COVID-19/master/dati-json/dpc-covid19-ita-province.json"

class Points:
    def __init__(self, data):
        self.data = data
    def hospitalized(self):
        return [y["ricoverati_con_sintomi"] for y in [x for x in self.data]]
    def icu(self):
        return [y["terapia_intensiva"] for y in [x for x in self.data]]
    def hospitalized_tot(self):
        return [y["totale_ospedalizzati"] for y in [x for x in self.data]]
    def isolated(self):
        return [y["isolamento_domiciliare"] for y in [x for x in self.data]]
    def positive_tot(self):
        return [y["totale_attualmente_positivi"] for y in [x for x in self.data]]
    def positivie_curr(self):
        return [y["nuovi_attualmente_positivi"] for y in [x for x in self.data]]
    def recovered(self):
        return [y["dimessi_guariti"] for y in [x for x in self.data]]
    def dead(self):
        return [y["deceduti"] for y in [x for x in self.data]]
    def rd(self):
        return self.dead() + self.recovered()
    def total(self):
        return [y["totale_casi"] for y in [x for x in self.data]]
    def wad(self):
        return [y["tamponi"] for y in [x for x in self.data]]
    def dates(self):
        return [y["data"] for y in [x for x in self.data]]

def request2Json(url):
    res = requests.get(url)
    decoded =res.content.decode('utf-8-sig')
    return json.loads(decoded)

def region_code(x):
    return str(x[0]) + '_' +re.sub(r'[\.\'\ ]', "_", x[1])

def equality(y, n):
    return y["codice_regione"] == n["codice_regione"] and y["denominazione_regione"] == n["denominazione_regione"]
class Data:
    def getRegionPoint(self, codice_regione, point):
        return [y[point] for y in [x for x in self.regions_data] if equality(y, n)]
    def __init__(self, country_URL, regions_URL, provinces_URL):
        self.country_URL = country_URL
        self.regions_URL = regions_URL
        self.provinces_URL = provinces_URL
    def request(self):
        self.counry_data = request2Json(self.country_URL) 
        self.regions_data = request2Json(self.regions_URL)
        self.provinces_data = request2Json(self.provinces_URL)

    def getCountry(self):
        return Points(self.counry_data)
    def getRegions(self):
        return self.regions_data
    def getRegion(self, n):
        return Points([y for y in [x for x in self.regions_data] if equality(y, n)])
    
    def getCountrySize(self):
        return len(self.counry_data)
    def getRegionSize(self, n):
        return len([y for y in [x for x in self.regions_data] if equality(y, n)])
    def getRegionName(self, n):
        return [y for y in [x for x in self.regions_data] if equality(y, n)]
    def getRegionDate(self, n):
        return [y["data"] for y in [x for x in self.regions_data] if equality(y, n)]
    def getProvinces(self, n):
        return Points(self.provinces_data)
    def getLastDate(self):
        return self.counry_data[-1]["data"]
    def getRegionsCodes(self):
        region_list = list(set([(y["codice_regione"], y["denominazione_regione"]) for y in [x for x in self.regions_data]]))
        return [{
            "id": region_code(x), 
            "codice_regione": x[0], 
            "denominazione_regione": x[1]
        } for x in region_list]

# ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
# E X P O N E N T I A L
# ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

class LogRegression:
    def compute(self, ts, ys, start=0):
        _ys = np.array(ys[start:], copy=True)
        self.N = _ys.size
        #ts = np.arange(0, self.N, 1)
        self.TS = ts
        self.XS = np.vstack([self.TS, np.ones(self.N)]).T
        self.YS = np.array(np.log(_ys))
        reg = LinearRegression().fit(self.XS, self.YS)
        self.params = ( reg.coef_[0], reg.intercept_)
        return ( reg.coef_[0], reg.intercept_)
    def getParams(self):
        return self.params

def forecastLogRegression(points):
    ys = np.array(points.positivie_curr())
    dates = np.array(points.dates())
    ts = np.argwhere(ys > 0)
    xs = ts
    # regression (log)
    logRegressor = LogRegression()
    (m, q) = logRegressor.compute(ts.ravel(), ys[ts].ravel())
    # forecast
    xs_hat = ts
    ys_hat = ts * m + q
    r2 = r2_score(np.array(ys[ts]), np.exp(ys_hat))
    # result
    result = {
        "dates": dates[ts].ravel().tolist(),
        "xs": xs.ravel().tolist(),
        "ys": ys[ts].ravel().tolist(),
        "xs_hat": xs_hat.ravel().tolist(),
        "ys_hat": np.exp(ys_hat).ravel().tolist(),
        "params": [m, q],
        "r2": r2
    }
    return result

def date2FileSuffix(date):
    return datetime.strptime(date, '%Y-%m-%d %H:%M:%S').strftime("%Y%m%d")

def computeExponential(input_data, start = 0):
    output_data = forecastLogRegression(input_data["data"])
    # filename = "./data/exponential-" + input_data["name"] + "-" + date2FileSuffix(input_data["timestamp"]) + ".json"
    # resFile = open(filename, "w")
    # resFile.write(simplejson.dumps(output_data, indent=4, sort_keys=True))
    # resFile.close()
    return output_data

# ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
# S I G M O I D
# ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

def computeSigmoid(input_data, start = 0):
    output_data = forecastSigmoid(input_data["data"])
    # filename = "./data/sigmoid-" + input_data["name"] + "-" + date2FileSuffix(input_data["timestamp"]) + ".json"
    # resFile = open(filename, "w")
    # resFile.write(simplejson.dumps(output_data, indent=4, sort_keys=True))
    # resFile.close()
    return output_data

def forecastSigmoid(points):
    ys = np.array(points.total())
    dates = np.array(points.dates())
    ts = np.argwhere(ys > 0)
    xs = ts

    # --------------------
    # ORD. DIFF. EQ.
    # --------------------
    def fit_sigmoid(x, r, K):
        def sigmoid_ode(y, t):
            return r * y * (1 - y/K)

        y0 = ys[ts][0]
        solution = odeint(sigmoid_ode, y0, x)
        return solution[:,0]

    fit_params, fit_covariance = curve_fit(fit_sigmoid, ts.ravel(), ys[ts].ravel(), p0=(2, 1000))

    # --------------------
    # SIGMOID FORECAST
    # --------------------
    xs_hat = np.arange(0, 60, 1)
    ys_hat = fit_sigmoid(xs_hat, fit_params[0], fit_params[1])
    score = r2_score(np.array(ys[ts]), ys_hat[ts])

    delta_ys_hat = np.diff(ys_hat)
    np.all(delta_ys_hat[0] == delta_ys_hat)

    result = {
        "dates": dates[ts].ravel().tolist(),
        "xs": xs.ravel().tolist(),
        "ys": ys[ts].ravel().tolist(),
        "xs_hat": xs_hat.ravel().tolist(),
        "ys_hat": ys_hat.ravel().tolist(),
        "params": [fit_params.ravel().tolist(), fit_covariance.ravel().tolist()],
        "r2": score
    }

    return result

# ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
#  C O M P U T I N G - R E S U L T S
# ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

dmpc_covid_data = Data(COUNRTY_URL, REGION_URL, PROV_URL)
dmpc_covid_data.request()
last_date = dmpc_covid_data.getLastDate()
regions_set = dmpc_covid_data.getRegionsCodes()

results = {}

input_data = {
    "timestamp": dmpc_covid_data.getLastDate(),
    "name":  '0_Italia',
    "data":   dmpc_covid_data.getCountry(),   
}
results[input_data["name"]] = {
    "exponential": computeExponential(input_data),
    "sigmoid": computeSigmoid(input_data)
}

for i in regions_set:
    input_data = {
        "timestamp": dmpc_covid_data.getLastDate(),
        "name":  i["id"],
        "data":   dmpc_covid_data.getRegion(i),   
    }
    results[input_data["name"]] = {
    "exponential": computeExponential(input_data, i),
    "sigmoid": computeSigmoid(input_data, i)
}

filename = "./data/" + date2FileSuffix(last_date) + ".json"
resFile = open(filename, "w")
resFile.write(simplejson.dumps(results, indent=4, sort_keys=True))
resFile.close()