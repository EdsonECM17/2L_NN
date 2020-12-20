# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 18:16:51 2020

GENERAL_UTILS

Common use functions
@author: edson
"""

def normalizar(r,lb,ub):
    return (r-lb)/(ub-lb);


def desnormalizar(n,lb,ub):
    return n*(ub-lb)+lb;


def maxp(V):
    #(val,pos)=maxp(V)
    n=len(V);
    pos=0;
    val=V[pos];
    for e in range(n):
        if V[e]>val:
            val=V[e];
            pos=e;          
            
    return val,pos

def minp(V):
    #(val,pos)=minp(V)
    n=len(V);
    pos=0;
    val=V[pos];
    for e in range(n):
        if V[e]<val:
            val=V[e];
            pos=e;
            
    return val,pos

