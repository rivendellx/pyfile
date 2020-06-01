# -*- coding: utf-8 -*-
"""
Created on Sun May 31 18:15:27 2020

@author: RivenD

The following have not been included

"Omega Ratio","% of Positive Months"
"""
import numpy as np
import pandas as pd
from scipy import stats
import statistics as st


def EVA(smb,n):
    """
    smb: pass an arrary or series here
    n  : n=12 for monthly data; n=250 for daily data
    """

    p=0.99
    name=["Annualized Mean","t-statistics","Annualized Volatility","Sharpe Ratio","Skewness","Excess Kurtosis",
          "99%VaR(Cornish-Fisher)","Annualized Downside Volatility","Sortino Ratio",
          "Maximum Drawdown","Max.Draw. Length","CER"]
    title=smb.name
    
    #annualized mean
    gm=(st.geometric_mean(smb+1)-1)*n
    mu=st.mean(smb)
    
    #annualized S.D.
    sd=st.stdev(smb)
    asd=st.sqrt(n)*st.stdev(smb)
    
    #annualized Sharpe
    sharpe=gm/asd
    
    #skewness and kurtosis
    skew=stats.skew(smb)
    kurt=stats.kurtosis(smb)
    
    #t-stats
    tvalue=stats.ttest_1samp(smb,0)[0]
    
    #99% VaR-Cornish_Fisher
    inter = stats.norm.isf(p)
    zcf=inter+((inter**2-1)*skew/6)+((inter**3-3*inter)*kurt/24)-((2*inter**3-5*inter)*(skew**2)/36)
    VaR=mu+sd*zcf*(-1)
    
    #downside volatility
    nret=smb.loc[smb<0]
    dvola=st.stdev(nret)*st.sqrt(n)
    
    #sortino
    sortino=((1+mu)**n-1)/dvola
    
    #maxmium drawdown
    uvi=(1+smb).cumprod()
    dif=np.maximum.accumulate(uvi) #return the largest value so far
    draw=uvi/dif-1
    maxdraw=min(draw)
    ddend=np.argmin(draw)
    ddstart=np.argmax(uvi[:ddend])
    ddlength=ddend-ddstart
    
    #CER
    cer=((1+smb)**(1-5)-1)/(-4)
    CER=st.mean(cer)*n
    
    #Export
    number=[gm,tvalue,asd,sharpe,skew,kurt,VaR,dvola,sortino,maxdraw,ddlength,CER]
    output=pd.DataFrame({'Stats':name, title:number})
    print(output)
    return(output)
