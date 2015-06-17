##
##  BIOMETRIC FEATURES
##
##  J.A.Henderson jilliana@magneticore.com
##
##  Functions to calculate human factors such as mouse and keyboard dynamics 
##  for use in the Angry User Experiment. 
##

import pandas as pd
import seaborn as sns
import time as clk
import calendar as cal
import numpy as np
import os
import re
import traceback as trace
from datetime import datetime,timedelta
import math as mt
import matplotlib.pyplot as plt
import numpy as np

def validateData(aColumn):
    if len(aColumn) > 0:
        valid = True
    else:
        valid = False
    return valid

def print_full(x):
    pd.set_option('display.max_rows', len(x))
    print x
    pd.reset_option('display.max_rows')


def rollingMean(df,freq):
    ## Function based on python reply from stackoverflow (user2689410's originally posted Aug. 27, 2013)
    def f(x):
      #dslice = col[x-pd.datetools.to_offset(freq).delta/2+timedelta(0,0,1): 
      #             x+pd.datetools.to_offset(freq).delta/2]
      dslice = col[x-pd.datetools.to_offset(freq).delta+timedelta(0,0,1):x]  
      return dslice.mean()
    data = df.copy()
    dfRS = pd.DataFrame()
    idx = pd.Series(data.index.to_pydatetime(), index=data.index)
    for colname, col in data.iteritems():
        rollingMean = idx.apply(f)
        rollingMean.name = "Rolling Mean"
        dfRS = dfRS.join(rollingMean,how='outer')
    return dfRS    

def rollingCount(df,freq):
    ## Function based on python reply from stackoverflow (user2689410's originally posted Aug. 27, 2013)
    def f(x):
      #dslice = col[x-pd.datetools.to_offset(freq).delta/2+timedelta(0,0,1): 
      #             x+pd.datetools.to_offset(freq).delta/2]
      dslice = col[x-pd.datetools.to_offset(freq).delta+timedelta(0,0,1):x]  
      return dslice.count()
    data = df.copy()
    dfRS = pd.DataFrame()
    idx = pd.Series(data.index.to_pydatetime(), index=data.index)
    for colname, col in data.iteritems():
        rollingCount = idx.apply(f)
        rollingCount.name = "Rolling Count"
        dfRS = dfRS.join(rollingCount,how='outer')
    return dfRS    
    
##-----------------------------------------
##  KEYBOARD DYNAMICS
##-----------------------------------------    
    
def getRidOfDoubleDown(df):
  dfKey = df.copy()  
  doubleDown = []
  lastrow = 1  
  for row in dfKey.iterrows():      
    try:
      ## Keep the first instance the key is pressed down and the first instance the key is Up.  
      if (row[1]["pressed"] == 0 and lastrow==0) or (row[1]["pressed"] == 1 and lastrow==1): 
        doubleDown.append(row[0])
      lastrow = row[1]["pressed"]  
    except:
      lastrow = row[1]["pressed"]  
      #print row
      trace.print_exc()
  for i in doubleDown: 
    dfKey = dfKey.drop(i)        
  labelN = dfKey.index[-1]  
  if dfKey["pressed"].loc[labelN] == 0: 
    #print '   Key ending in Down[0] position ==> drop it.'
    dfKey = dfKey.drop(labelN)
  ## COUNT: ##      
  dfCounts = dfKey.groupby(["pressed"]).count()
  if dfCounts.loc[0]["time"]==dfCounts.loc[1]["time"]: 
    #print '    ',dfCounts.loc[0]["time"],dfCounts.loc[1]["time"]    
    #print '    *** SUCCESS ****'    
    return dfKey
  else:
    #print '    ',dfCounts.loc[0]["time"],dfCounts.loc[1]["time"]    
    #print '    *** Could not get rid of Double Down! ***' 
    dfKey = getRidOfDoubleDown(dfKey)
  return dfKey
    
def getKeyDuration(df):
  countDF = df.groupby(["keycode","pressed"]).count()  
  codes = sorted(df.groupby("keycode").groups.keys())
  pieces = []
  ## for each keycode, check double-down and calculate key duration:  
  for code in codes:
    rule = df["keycode"]==code
    if not countDF.loc[code,0]["time"]==countDF.loc[code,1]["time"]:
        #print "Counts not equal! keycode: ",code," counts: ",countDF.loc[code,0]["time"]," ",countDF.loc[code,1]["time"]   
        ## If first pressed value == 1, eliminate it. Then Get rid of double down for that key code ##
        dff = getRidOfDoubleDown(df[rule])
    else:
        dff = df[rule]        
    ruleDown = dff['pressed']==0
    ruleUp = dff['pressed']==1
    dfDown = dff[ruleDown][["time"]]
    dfUp = dff[ruleUp][["time","key type","key","key colour"]]
    downtime =  [float(int(dfUp["time"].loc[i])-int(dfDown["time"].loc[j]))/1000.0 for i,j in zip(dfUp.index,dfDown.index)]
    dateclock = [pd.to_datetime(c,unit='ms') for c in dfUp["time"]]
    pieces.append(pd.DataFrame({
                    "duration": pd.Series(downtime,index=dateclock),
                    "keycode": pd.Series([code for d in dateclock],index=dateclock),
                    "key type": pd.Series([t for t in dfUp["key type"]],index=dateclock),
                    "key": pd.Series([k for k in dfUp["key"]],index=dateclock),                   
                    "key colour": pd.Series([k for k in dfUp["key colour"]],index=dateclock)                   
                   }))
  durationDF = pd.concat(pieces)  
  durationDF.index = pd.DatetimeIndex(durationDF.index)
  durationDF.index.names = ["DateTime"] 
  return durationDF  
    
def getKeyLatency(df):    
  countDF = df.groupby(["keycode","pressed"]).count()  
  codes = sorted(df.groupby("keycode").groups.keys())
  pieces = []
  ## for each keycode, check double-down and calculate key duration:  
  for code in codes:
    rule = df["keycode"]==code
    if not countDF.loc[code,0]["time"]==countDF.loc[code,1]["time"]:
        #print "Counts not equal! keycode: ",code," counts: ",countDF.loc[code,0]["time"]," ",countDF.loc[code,1]["time"]   
        ## If first pressed value == 1, eliminate it. Then Get rid of double down for that key code ##
        dff = getRidOfDoubleDown(df[rule])
    else:
        dff = df[rule] 
    pieces.append(dff)
  latencyDF = pd.concat(pieces)
  latencyDF = latencyDF.sort()
  tdeltas = np.diff(latencyDF.index.values)
  tdeltas = np.insert(tdeltas,0,np.timedelta64(0,'ns'))
  latencyDF['latency'] = pd.Series([td/np.timedelta64(1,'s') for td in tdeltas],index=latencyDF.index) 
  latencyDF.index = pd.DatetimeIndex(latencyDF.index)
  latencyDF.index.names = ["DateTime"]  
  return latencyDF  
  
def getMouseDynamics(data):
    ddx,ddy,ddt = 1,1,1 #minimal pixel movements
    indices = [c for c in data.index]    
    xPosition = [d for d in data['x']]
    yPosition = [d for d in data['y']]
    mouseClock = [d for d in data['time']]
    ## CHECK FOR MOUSE SPEED FATER THAN RECORDED:
    dx =[(xPosition[i]-xPosition[i-1])  if i>0 else 0 for i in range(len(mouseClock))]
    dy = [(yPosition[i]-yPosition[i-1])  if i>0 else 0 for i in range(len(mouseClock))]
    ds = [np.sqrt(float(dx[i])**2 + float(dy[i])**2) for i in range(len(mouseClock))]
    dt = [(mouseClock[i]-mouseClock[i-1])  if i>0 else 0 for i in range(len(mouseClock))]  
    xPos,yPos,clock,dateclock = [],[],[],[]
    for x,y,s,delt,t,d in zip(xPosition,yPosition,ds,dt,mouseClock,indices):
        if delt==0 and s>0:
            continue
            #print 'Skip row: ',x,y,s,t,d
        else:    
            xPos.append(x)
            yPos.append(y)
            clock.append(t)
            dateclock.append(d)
    ## COMPUTE DYNAMICS:
    N = len(dateclock)
    if not N==len(data): print "Eliminated ",len(data)-N," data points where the speed of the mouse was faster than the key-capture program could record."
    dx = [float(xPos[i]-xPos[i-1]) if not i==0 else 0.0 for i in range(N)]
    dy = [float(yPos[i]-yPos[i-1]) if not i==0 else 0.0 for i in range(N)]
    ds = [np.sqrt(x**2 + y**2) for x,y in zip(dx,dy)]
    dt = [float(clock[i]-clock[i-1]) if not i==0 else 0.0 for i in range(N)]
    theta = [np.arctan(y/x) if x > 0 else 0.0 for x,y in zip(dx,dy)]     
    dtheta = [(theta[i]-theta[i-1])  if i>0 else 0.0 for i in range(N)]
    curve = [th/s if s > 0 else 0.0 for th,s in zip(dtheta,ds)]
    dcurve = [(curve[i]-curve[i-1])  if i>0 else 0.0 for i in range(N)]
    vx = [x/t if t > 0 else 0.0 for x,t in zip(dx,dt)]
    vy = [y/t if t > 0 else 0.0 for y,t in zip(dy,dt)]
    vel = [np.sqrt(x**2 + y**2)  for x,y in zip(vx,vy)]   
    vdot = [(vel[i] - vel[i-1])/dt[i] if (i > 1 and dt[i] > 0) else 0.0 for i in range(N)]
    jerk = [(vdot[i] - vdot[i-1])/dt[i] if (i > 1 and dt[i] > 0) else 0.0 for i in range(N)]
    omega = [th/t if t > 0 else 0.0 for th,t in zip(dtheta,dt)]     
    ## Add data to mouse dynamics pandas dataframe:
    mdDict = {
              "x" : pd.Series(xPos,index=dateclock),
              "y" : pd.Series(yPos,index=dateclock),
              "dx" : pd.Series(dx,index=dateclock),
              "dy" : pd.Series(dy,index=dateclock),
              "ds" : pd.Series(ds,index=dateclock),
              "dt" : pd.Series(dt,index=dateclock),
              "theta" : pd.Series(theta,index=dateclock),
              "dtheta" : pd.Series(dtheta,index=dateclock),
              "curve" : pd.Series(curve,index=dateclock),
              "dcurve" : pd.Series(dcurve,index=dateclock),
              "vx" : pd.Series(vx,index=dateclock),
              "vy" : pd.Series(vy,index=dateclock),
              "v" : pd.Series(vel,index=dateclock),
              "a" : pd.Series(vdot,index=dateclock),
              "jerk" : pd.Series(jerk,index=dateclock),
              "w" : pd.Series(omega,index=dateclock)
              }  
    mdDF = pd.DataFrame(mdDict)   
    mdDF.index = pd.DatetimeIndex(mdDF.index)
    mdDF.index.names = ["DateTime"]  
    return mdDF  
    
## PRINTING FUNCTIONS ##


def printKeyTypeDNA(keyGroups,keydataDF,outputFile="",printIt=False):
    sns.set_style("white")
    labelDict = {'fontsize': 16, 'weight' : 'roman'}
    fig,ax = plt.subplots(figsize=(20,5))
    for g in keyGroups.groups:
        colours = [c for c in keyGroups.get_group(g)['key colour']]
        x = [i for i in keyGroups.get_group(g)['keycode'].index]
        y = [k for k in keyGroups.get_group(g)['keycode']]
        ax.scatter(x,y,s=30,marker='o',c=colours,linewidths=0,alpha=0.5,label=g)
    box = ax.get_position()
    ax.set_position([box.x0,box.y0,box.width*0.8,box.height])
    ax.set_xlim(keydataDF.index[0],keydataDF.index[-1])
    ax.legend(loc='center left',bbox_to_anchor=(1,0.5),fancybox=True)
    ax.set_ylabel("keycode",fontdict=labelDict)
    ax.set_xlabel("clock",fontdict=labelDict)
    plt.show()
    if printIt:
        fig.savefig(outputFile,format='png',dpi=256)
    plt.close(fig)
    plt.clf()
    return
    
def printKeyTypeDNAwithLabels(keyGroups,keydataDF,labelsDF,outputFile="",printIt=False):
    sns.set_style("white")
    labelDict = {'fontsize': 16, 'weight' : 'roman'}
    fig,ax = plt.subplots(figsize=(18,10))
    for g in keyGroups.groups:
        colours = [c for c in keyGroups.get_group(g)['key colour']]
        x = [i for i in keyGroups.get_group(g)['keycode'].index]
        y = [k for k in keyGroups.get_group(g)['keycode']]
        #ax.scatter(x,y,s=100,marker='|',c=colours,linewidths=1,alpha=0.8,label=g)
        ax.scatter(x,y,s=30,marker='o',c=colours,linewidths=0,alpha=0.5,label=g)
    colours = sns.color_palette("GnBu_d",len(labelsDF))    
    for n,(d,l) in enumerate(zip(labelsDF.index,labelsDF['label'])):
        ax.plot([d,d],[0,225],color=colours[n],linewidth=3,alpha=0.5,label=l)
    box = ax.get_position()
    ax.set_position([box.x0,box.y0+box.height*0.7,box.width,box.height*0.3])
    ax.set_xlim(keydataDF.index[0],keydataDF.index[-1])
    ax.legend(loc='upper center',bbox_to_anchor=(0.5,-0.4))
    ax.set_ylabel("keycode",fontdict=labelDict)
    ax.set_xlabel("clock",fontdict=labelDict)
    plt.show()
    if printIt:
        fig.savefig(outputFile,format='png',dpi=256)
    plt.close(fig)
    plt.clf()
    return    
    
def printKeyTypeDNAwithActiveRanges(keyGroups,keydataDF,activeRanges,outputFile="",printIt=False):
    sns.set_style("white")
    labelDict = {'fontsize': 16, 'weight' : 'roman'}
    fig,ax = plt.subplots(figsize=(18,5))
    for g in keyGroups.groups:
        colours = [c for c in keyGroups.get_group(g)['key colour']]
        x = [i for i in keyGroups.get_group(g)['keycode'].index]
        y = [k for k in keyGroups.get_group(g)['keycode']]
        ax.scatter(x,y,s=30,marker='o',c=colours,linewidths=0,alpha=0.5,label=g)
    for (x1,x2) in activeRanges:
        ax.fill_betweenx(y=[0,255],x1=[x1,x1],x2=[x2,x2],color="skyblue",alpha=0.1)
    ax.legend()
    ax.set_xlim(keydataDF.index[0],keydataDF.index[-1])
    ax.set_ylabel("keycode",fontdict=labelDict)
    ax.set_xlabel("clock",fontdict=labelDict)
    plt.show()
    if printIt:
        fig.savefig(outputFile,format='png',dpi=256)
    plt.close(fig)
    plt.clf()
    return 
    
def printKeyLatency(latencyDF,labelsDF,activeRanges,outputFile="",printIt=False):

    rule = latencyDF["pressed"]==0
    labelDict = {'fontsize': 16, 'weight' : 'roman'}
    sns.set_style("white")
    fig,(ax2,ax1,ax) = plt.subplots(nrows=3,ncols=1,figsize=(18,15),sharex=True)

    ## Plot key latency ##
    xdata = latencyDF[rule].index
    ydata = latencyDF[rule]["latency"]
    colours = [c for c in latencyDF[rule]['key colour']]
    ax.scatter(xdata,ydata,s=20,c=colours,linewidths=0,alpha=0.8)
    ax.set_xlim(latencyDF.index[0],latencyDF.index[-1])
    ax.plot([latencyDF.index[0],latencyDF.index[-1]],[latencyDF["latency"].quantile(0.01),latencyDF["latency"].quantile(0.01)],color='k',
            linewidth=2,linestyle="--",alpha=0.5,label="1percent of data below this line")
    colours = sns.color_palette("GnBu_d",len(labelsDF))    
    for n,(d,l) in enumerate(zip(labelsDF.index,labelsDF['label'])):
        ax.plot([d,d],[0,3],color=colours[n],linewidth=3,alpha=0.5,label=l)
        ax.text(d-timedelta(seconds=1),0.15,str(n),fontsize=12,color='grey',weight="bold",horizontalalignment='right',verticalalignment='bottom')
    ax.set_ylabel("key latency (s)",fontdict=labelDict)
    box = ax.get_position()
    ax.set_position([box.x0,box.y0+box.height*0.3,box.width,box.height*0.7])
    ax.legend(loc='upper center',bbox_to_anchor=(0.5,-0.3))
    ax.set_ylim(-0.01,1.0)

    ## HIGHLIGHT CALM TYPING ##
    x1,x2 = labelsDF.iloc[1]["Key Data Ranges"]
    ax.fill_betweenx(y=[0,255],x1=[x1,x1],x2=[x2,x2],color="skyblue",alpha=0.2)
    ax1.fill_betweenx(y=[0,255],x1=[x1,x1],x2=[x2,x2],color="skyblue",alpha=0.2)
    ax2.fill_betweenx(y=[0,255],x1=[x1,x1],x2=[x2,x2],color="skyblue",alpha=0.2)

    ## HIGHLIGHT RAPID TYPING UNDER PRESSURE##
    x1,x2 = labelsDF.iloc[13]["Key Data Ranges"]
    ax.fill_betweenx(y=[0,255],x1=[x1,x1],x2=[x2,x2],color="skyblue",alpha=0.2)
    ax1.fill_betweenx(y=[0,255],x1=[x1,x1],x2=[x2,x2],color="skyblue",alpha=0.2)
    ax2.fill_betweenx(y=[0,255],x1=[x1,x1],x2=[x2,x2],color="skyblue",alpha=0.2)

    ## Active Ranges ##
    for (x1,x2) in activeRanges:
    
        ## Rolling mean:
        freq = "1S"
        roll = rollingMean(latencyDF[rule][x1+timedelta(seconds=1):x2][["latency"]],freq)
        roll=roll.fillna(0.0)
        rollY = [y for y in roll["Rolling Mean"]]
        xdata = [i for i in roll.index]
        ax1.plot(xdata,rollY,color='r',linewidth=3,alpha=0.2)

        ## Rolling count:
        count = rollingCount(latencyDF[rule][x1+timedelta(seconds=1):x2][["latency"]],freq)
        count=count.fillna(0.0)
        countY = [y for y in count["Rolling Count"]]
        xdata = [i for i in count.index]
        ax2.plot(xdata,countY,color='g',linewidth=2,alpha=0.2)

    ax1.text(xdata[-2],8.,"rolling 1-second MEAN",fontsize=12,color='grey',weight="bold",horizontalalignment='right',verticalalignment='bottom')
    ax1.set_ylim(0,10)

    ax2.text(xdata[-2],17.,"rolling 1-second COUNT",fontsize=12,color='grey',weight="bold",horizontalalignment='right',verticalalignment='bottom')
    ax2.set_ylim(0,20)
    
    plt.show()
    if printIt:
        fig.savefig(outputFile,format='png',dpi=256)
    plt.close(fig)
    plt.clf()
    return    
    
def printKeyDuration(durationDF,labelsDF,activeRanges,outputFile="",printIt=False):

    keyGroups = durationDF.groupby('key type')
    labelDict = {'fontsize': 16, 'weight' : 'roman'}
    sns.set_style("white")
    fig,(ax2,ax1,ax) = plt.subplots(nrows=3,ncols=1,figsize=(18,15),sharex=True)

    for g in keyGroups.groups:
        colours = [c for c in keyGroups.get_group(g)['key colour']]
        x = [i for i in keyGroups.get_group(g)['duration'].index]
        y = [k for k in keyGroups.get_group(g)['duration']]
        ax.scatter(x,y,s=30,marker='o',c=colours,linewidths=0,alpha=0.5,label=g)
    ax.plot([durationDF.index[0],durationDF.index[-1]],[durationDF["duration"].quantile(0.01),durationDF["duration"].quantile(0.01)],
            color='k',linewidth=2,linestyle="--",alpha=0.5,label="1percent of data below this line")
    colours = sns.color_palette("GnBu_d",len(labelsDF))    
    for n,(d,l) in enumerate(zip(labelsDF.index,labelsDF['label'])):
        ax.plot([d,d],[0,3],color=colours[n],linewidth=3,alpha=0.5,label=l)
        ax.text(d-timedelta(seconds=1),0.15,str(n),fontsize=12,color='grey',weight="bold",horizontalalignment='right',verticalalignment='bottom')
    box = ax.get_position()
    ax.set_position([box.x0,box.y0+box.height*0.3,box.width,box.height*0.7])
    ax.legend(loc='upper center',bbox_to_anchor=(0.5,-0.3))
    ax.set_xlim(durationDF.index[0],durationDF.index[-1])
    ax.set_ylim(0,0.2)
    ax.set_ylabel("key duration (s)",fontdict=labelDict)

    ## HIGHLIGHT CALM TYPING ##
    x1,x2 = labelsDF.iloc[1]["Key Data Ranges"]
    ax.fill_betweenx(y=[0,255],x1=[x1,x1],x2=[x2,x2],color="skyblue",alpha=0.2)
    ax1.fill_betweenx(y=[0,255],x1=[x1,x1],x2=[x2,x2],color="skyblue",alpha=0.2)
    ax2.fill_betweenx(y=[0,255],x1=[x1,x1],x2=[x2,x2],color="skyblue",alpha=0.2)

    ## HIGHLIGHT RAPID TYPING UNDER PRESSURE##
    x1,x2 = labelsDF.iloc[13]["Key Data Ranges"]
    ax.fill_betweenx(y=[0,255],x1=[x1,x1],x2=[x2,x2],color="skyblue",alpha=0.2)
    ax1.fill_betweenx(y=[0,255],x1=[x1,x1],x2=[x2,x2],color="skyblue",alpha=0.2)
    ax2.fill_betweenx(y=[0,255],x1=[x1,x1],x2=[x2,x2],color="skyblue",alpha=0.2)

    ## HIGHLIGHT KEYBOARD SMASH##
    x1,x2 = labelsDF.iloc[10]["Key Data Ranges"]
    ax.fill_betweenx(y=[0,255],x1=[x1,x1],x2=[x2,x2],color="grey",alpha=0.2)
    ax1.fill_betweenx(y=[0,255],x1=[x1,x1],x2=[x2,x2],color="grey",alpha=0.2)
    ax2.fill_betweenx(y=[0,255],x1=[x1,x1],x2=[x2,x2],color="grey",alpha=0.2)
    x1,x2 = labelsDF.iloc[11]["Key Data Ranges"]
    ax.fill_betweenx(y=[0,255],x1=[x1,x1],x2=[x2,x2],color="grey",alpha=0.2)
    ax1.fill_betweenx(y=[0,255],x1=[x1,x1],x2=[x2,x2],color="grey",alpha=0.2)
    ax2.fill_betweenx(y=[0,255],x1=[x1,x1],x2=[x2,x2],color="grey",alpha=0.2)

    ## Rolling mean:
    freq = "1S"
    roll = durationDF[["duration"]].resample(freq)
    roll = roll.fillna(0.0)
    rollY = [y for y in roll["duration"]]
    xdata = [i for i in roll.index]
    ax1.plot(xdata,rollY,color='r',linewidth=3,alpha=0.2)
    ## Rolling count:
    count = durationDF[["duration"]].resample(freq,how="count")
    count = count.fillna(0.0)
    countY = [y for y in count["duration"]]
    ax2.plot(xdata,countY,color='g',linewidth=3,alpha=0.2)
    
    ax1.text(durationDF.index[-2],2.5,"1-second resample MEAN",fontsize=12,color='grey',weight="bold",horizontalalignment='right',verticalalignment='bottom')
    ax1.set_ylim(0,3)
    ax1.set_xlim(durationDF.index[0],durationDF.index[-1])

    ax2.text(durationDF.index[-2],12.,"1-second resample COUNT",fontsize=12,color='grey',weight="bold",horizontalalignment='right',verticalalignment='bottom')
    ax2.set_ylim(0,15)
    ax2.set_xlim(durationDF.index[0],durationDF.index[-1])
    
    plt.show()
    if printIt:
        fig.savefig(outputFile,format='png',dpi=256)
    plt.close(fig)
    plt.clf()
    return        
    
def printmouseVelocity(mousedataDF,keydataDF,vDF,labelsDF,outputFile="",printIt=False):
    
    sns.set_style("white")
    labelDict = {'fontsize': 16, 'weight' : 'roman'}
    td = timedelta(seconds=30)
    N = len(labelsDF)
    fig,axes = plt.subplots(nrows=N,ncols=2,figsize=(18,N*5))
    axes=fig.axes
    axi=0
    ranges = [(labelsDF.index[i-1],labelsDF.index[i]) if i > 0 else (mousedataDF.index[0],labelsDF.index[i]) for i in range(len(labelsDF))]
    for r,l in zip(ranges,labelsDF['label']):
        # PLOT MOUSE POSITION AND CLICK ##
        mdata = mousedataDF[r[0]:r[1]]
        colours = sns.color_palette('cubehelix',len(mdata))
        ax=axes[axi]
        xdata = [x for x in mdata['x']]
        ydata = [-y for y in mdata['y']]
        ax.scatter(xdata,ydata,s=20,c=colours,alpha=0.8,linewidth=0)
        # MOUSE CLICK INFO ##
        kdata = keydataDF[r[0]:r[1]]
        left = kdata['key']=='left click'
        right = kdata['key']=='right click'
        for i in kdata[left].index:
            if len(mdata[:i])>0:
                x = mdata[:i]['x'][-1]
                y = mdata[:i]['y'][-1]
                ax.scatter(x,-y,s=40,c='w',alpha=0.8,marker="D",linewidth=1)
        for i in kdata[right].index:
            if len(mdata[:i])>0:
                x = mdata[:i]['x'][-1]
                y = mdata[:i]['y'][-1]
                ax.scatter(x,-y,s=40,c='k',alpha=0.8,marker="D",linewidth=1)
        # WRITE LABELS ##        
        x1,x2 = ax.get_xlim()
        y1,y2 = ax.get_ylim()
        ax.set_ylim(y1,y2*1.1)
        ax.text(x1,y1+((y2-y1)*0.05),l[:65],fontsize=12,color='k',weight="bold",horizontalalignment='left',verticalalignment='bottom')
        ax.text(x1,y1,l[80:],fontsize=12,color='k',weight="bold",horizontalalignment='left',verticalalignment='bottom')
        axi+=1
        # PLOT MOUSE VELOCITY ##
        ax=axes[axi]
        vdata = vDF[r[0]:r[1]]
        xdata = [i for i in vdata.index]
        ydata = [v for v in vdata['vnorm']]
        ax.plot(xdata,ydata,color='k',linewidth=1,alpha=0.8)
        #ax.set_yscale(u'log')    
        ax.set_ylabel("Mouse Velocity",fontdict=labelDict)
        ax.set_xlabel("clock: 30-second span",fontdict=labelDict)
        if len(xdata)>0:
            x1 = xdata[0]
        else:
            x1 = r[0]
        ax.set_xlim(x1,x1+td)
        ax.set_ylim(0.0,0.5)
        axi+=1
        
    plt.show()
    if printIt:
        fig.savefig(outputFile,format='png',dpi=256)
    plt.close(fig)
    plt.clf()
    return    
    