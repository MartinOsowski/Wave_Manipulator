# -*- coding: utf-8 -*-
"""
Created on Sat Apr 28 18:55:09 2018

@author: M.Osowski
"""

import numpy as np
import matplotlib.pyplot as plt
import numpy.random as rnd
import cmath
from scipy.fftpack import fft, ifft
import sounddevice as sd
import wave


def sup(sig_1, sig_2):
    new_sig = sig()
    if sig_1.N == sig_2.N:
        new_sig.time_eq = sig_1.time_eq + sig_2.time_eq
    else:
        if sig_1.N > sig_2.N:
            new_sig.set_domain(sig_1.lower_bound, sig_1.upper_bound)
            tempsignal = np.resize(sig_2.time_eq, sig_1.N)
            new_sig.time_eq = sig_1.time_eq + tempsignal
        else:
            new_sig.set_domain(sig_2.lower_bound, sig_2.upper_bound)
            tempsignal = np.resize(sig_1.time_eq, sig_2.N)
            new_sig.time_eq = tempsignal + sig_2.time_eq
    
    new_sig.ft()
    new_sig.plot()
    
    return new_sig

def seq(sig_1, sig_2):
    new_sig = sig()
    new_sig.time_eq = np.concatenate((sig_1.time_eq, sig_2.time_eq))
    new_sig.set_domain(0, (sig_1.upper_bound - sig_1.lower_bound) + (sig_2.upper_bound - sig_2.lower_bound))
    
    new_sig.plot()
    
    return new_sig

def mult(sig_1, sig_2):
    new_sig = sig()
    if sig_1.N > sig_2.N:
            new_sig.set_domain(sig_1.lower_bound, sig_1.upper_bound)
            tempsignal = np.resize(sig_2.time_eq, sig_1.N)
            new_sig.time_eq = np.multiply(sig_1.time_eq, tempsignal)
    else:
        new_sig.set_domain(sig_2.lower_bound, sig_2.upper_bound)
        tempsignal = np.resize(sig_1.time_eq, sig_2.N)
        new_sig.time_eq = np.multiply(sig_2.time_eq, tempsignal)

    new_sig.ft()
    new_sig.plot()
    
    return new_sig

class sig:   
            
    def __init__(self, freq = 1., A = 1., rand = False):
        self.T = 1./44100.
        self.lower_bound = 0.
        self.upper_bound = 4.
        self.N = int((self.upper_bound - self.lower_bound)/self.T)
        self.domain = np.linspace(self.lower_bound, self.upper_bound, self.N)
        self.fdomain = np.linspace(0.0, 1.0/(2.0*self.T), self.N//2)
        self.fspacing = (1.0/(2.0*self.T))/(self.N//2)
        
        if rand:
            print('random')
            self.randw()
        else:
            self.time_eq = A*np.cos(2*np.pi*freq*self.domain)
            self.fourier = fft(self.time_eq)
            self.time_eq_str = '' 
        
            
        self.plot()
        
        print("Wave initialized")
      
    def set_time_eq(self):
        self.time_eq_str = ''
        func = str(input("Enter the time domain signal equation in term of the variable t: ",))
        print()
        
        self.time_eq_str = func.replace("t", "self.domain").replace("cos", "np.cos").replace("sin", "np.sin").replace("pi", "np.pi")
        self.time_eq = eval(self.time_eq_str)
        print("Time domain wave equation set to:")
        print(func)  
        
        self.ft()
        self.plot()
    
    def set_domain(self, lower, upper):
        self.lower_bound = float(lower)
        self.upper_bound = float(upper)
        if self.upper_bound < self.lower_bound or self.upper_bound == self.lower_bound:
            print("Upper bound must be larger than lower bound")
        else:
            self.N = int((self.upper_bound - self.lower_bound)/self.T)
            self.domain = np.linspace(self.lower_bound,self.upper_bound,int(self.N))
            self.fdomain = np.linspace(0.0, 1.0/(2.0*self.T), self.N//2)
            self.fspacing = (1.0/(2.0*self.T))/(self.N//2)
            self.time_eq = np.resize(self.time_eq, len(self.domain))
            print("Domain set to: ",self.lower_bound,",", self.upper_bound)
        self.ft()
        self.plot()
        
    def plot(self):
        plt.close('all')
        plt.figure()
        
        ax1 = plt.subplot(121)
        ax1.plot(self.domain, self.time_eq)
        ax1.set_title('Time Domain Signal')
        
        ax2 = plt.subplot(122)
        ax2.plot(self.fdomain, abs(self.fourier[0:int(self.N//2)])/self.N*2)
        ax2.set_title('Frequency Spectrum')
        
#        if np.size(np.where(abs(self.fourier[0:int(self.N//2)])/self.N*2>0.1)[-1]) != 0:
#            maxind = np.where(abs(self.fourier[0:int(self.N//2)])/self.N*2>0.1)[-1][-1]
#            ax2.set_xlim((0, self.fdomain[maxind]*(1.3+np.exp(-self.fdomain[maxind]))))
        
        plt.show()
        
    def phasor(self):
        self.stacking = True
        self.stack_counter = 0
        self.time_eq_str = ''
        self.time_eq = np.zeros(self.N)
        while self.stacking:
                        
            self.phase = float(input("Enter the phase angle in degrees: ",))
            print()
            
            while True:
                self.choice = str(input("Enter f: frequency, t: period  ",))
                print()
                if self.choice == "f":
                    self.fo = float(input("Enter the frequency in Hz: ",))
                    break
                elif self.choice == "t":
                    self.fo = 1. / float(input("Enter the period in s: ",))
                    break
                else: 
                    print("Please type either f or t")
                    print()
            print()
            
            self.A = float(input("Enter the wave amplitude: ",)) 
            print()
            
            if self.stack_counter == 0:                 
                self.time_eq_str = "".join((str(self.A),"*np.cos(2*np.pi*",str(self.fo),"*self.domain+np.pi/180.*",str(self.phase),")"))
            else:
                self.time_eq_str = "".join((self.time_eq_str," + ",str(self.A),"*np.cos(2*np.pi*",str(self.fo),"*self.domain+np.pi/180.*",str(self.phase),")"))
            self.time_eq += eval(self.time_eq_str)
            
            print("Time domain wave equation set to:")
            print(self.time_eq_str.replace("np.cos", "cos").replace("np.pi", "pi").replace("self.domain", "t").replace("+pi/180.*0.0", ""))
            print()
                            
            while True:
                self.stack_choice = str(input("Stack? y/n: ",))
                if self.stack_choice == "n":
                    self.stacking = False
                    break
                elif self.stack_choice == "y":
                    self.stacking = True
                    self.stack_counter +=1
                    break
                else:
                    print("Please type either y or n")
                    print()
        
        self.fourier = fft(self.time_eq)
        self.plot()
            
    def stack(self):
        self.stacking = True
        self.time_eq_str = ''

        while self.stacking:
                        
            self.phase = float(input("Enter the phase angle in degrees: ",))
            print()
            
            while True:
                self.choice = str(input("Enter f: frequency, t: period  ",))
                print()
                if self.choice == "f":
                    self.fo = float(input("Enter the frequency in Hz: ",))
                    break
                elif self.choice == "t":
                    self.fo = 1. / float(input("Enter the period in s: ",))
                    break
                else: 
                    print("Please type either f or t")
                    print()
            print()
            
            self.A = float(input("Enter the wave amplitude: ",)) 
            print()
                           
            self.time_eq_str = "".join((self.time_eq_str," + ",str(self.A), "*np.cos(2*np.pi*",str(self.fo),"*self.domain+np.pi/180.*",str(self.phase),")"))
            self.time_eq += eval(self.time_eq_str)
            print("Time domain wave equation set to:")
            print(self.time_eq_str.replace("np.cos", "cos").replace("np.pi", "pi").replace("self.domain", "t").replace("+pi/180.*0.0", ""))
            print()
                            
            while True:
                self.stack_choice = str(input("Stack? y/n: ",))
                if self.stack_choice == "n":
                    self.stacking = False
                    break
                elif self.stack_choice == "y":
                    self.stacking = True
                    break
                else:
                    print("Please type either y or n")
                    print()
            
        plt.close()
        plt.show()
        plt.plot(self.domain, self.time_eq)
        
    def stack_func(self):
        self.time_eq_str = ''
        print("Enter the time domain wave equation to add in term of the variable t:")
        func = str(input())
        print()
        
        self.time_eq_str = "".join((self.time_eq_str, " + ", func.replace("t", "self.domain").replace("cos", "np.cos").replace("sin", "np.sin").replace("pi", "np.pi")))
        self.time_eq += eval(self.time_eq_str)
        print("Time domain wave equation set to:")
        print(self.time_eq_str.replace("np.cos", "cos").replace("np.pi", "pi").replace("self.domain", "t").replace("+pi/180.*0.0", ""))
        print()
        
        self.fourier = fft(self.time_eq)
        self.plot()
        
    def randw(self, fmax = 1000., amax = 5., size = 15, amp_seed = False, freq_seed = False, phase_seed = False):
        if amp_seed != False:
            rnd.seed(amp_seed)
            print('Amplitude seed: ', amp_seed)
        self.amps = amax*rnd.rand(size)
        
        if freq_seed != False:
            rnd.seed(freq_seed)
            print('Frequency seed: ', freq_seed)
        self.freqs = fmax*rnd.rand(size)

        if phase_seed != False:
            rnd.seed(phase_seed)
            print('Phase seed: ', phase_seed)
        self.shifts = 360.*rnd.rand(size)
        
        self.time_eq_str = "".join((str(self.amps[0]),"*np.cos(2*np.pi*",str(self.freqs[0]),"*self.domain+np.pi/180.*",str(self.shifts[0]),")"))
        
        for i in range(size):
            if i != 0:
                self.time_eq_str = "".join((self.time_eq_str, '+', str(self.amps[i]),"*np.cos(2*np.pi*",str(self.freqs[i]),"*self.domain+np.pi/180.*",str(self.shifts[i]),")"))
                
        self.time_eq = eval(self.time_eq_str)
        self.ft()
        self.plot()
     
    def ftransform(self):
        self.fourier = np.zeros(self.N) + 0j
        for k in range(self.N):
            s = complex(0)
            for t in range(self.N):
                angle = 2j*cmath.pi*k*t/self.N 
                s += self.time_eq[t] * cmath.exp(-angle)
            self.fourier[k] = s/self.N*2          
        
    def ft(self):
        self.fourier = fft(self.time_eq)
              
    def ift(self):
        self.time_eq = np.real(ifft(self.fourier))
        
    def lpass(self, fmax):
        self.fourier[int(fmax/self.fspacing):] = np.zeros(len(self.fourier[int(fmax/self.fspacing):])) 
        self.ift()
        
        self.plot()
        
    def hpass(self, fmin):
        self.fourier[:int(fmin/self.fspacing)] = np.zeros(len(self.fourier[:int(fmin/self.fspacing)])) 
        self.ift()

        self.plot()
       
    def bfilter(self, fmin, fmax):
        self.fourier[int(fmax/self.fspacing):] = np.zeros(len(self.fourier[int(fmax/self.fspacing):])) 
        self.fourier[:int(fmin/self.fspacing)] = np.zeros(len(self.fourier[:int(fmin/self.fspacing)])) 
        self.ift()

        self.plot()
        
    def play(self):
        sd.play(self.time_eq, 44100)
    
    def stop(self):
        sd.stop()
    
    def rec(self, t = 10.):
        self.set_domain(0, t)
        self.time_eq = sd.rec(int(t/self.T), samplerate = 1/self.T, channels = 1, dtype='float64')
        self.time_eq = np.reshape(self.time_eq, int(t/self.T))
                                  
        self.ft()
        self.plot()
    
    def wav(self, filepathn):
        if '.wav' not in filepathn:
            filepath = ''.join((filepathn,'.wav'))
        else: 
            filepath = filepathn
        fp = wave.open(filepath)
        nchan = fp.getnchannels()
        N = fp.getnframes()
        dstr = fp.readframes(N*nchan)
        data = np.frombuffer(dstr, np.int16)
        data = np.reshape(data, (-1,nchan))
        self.time_eq = data[:,0]
        self.set_domain(0,len(self.time_eq)*self.T)
        self.ft()
        self.plot()
        
    def power(self):
        peaks = np.where(abs(self.fourier[0:self.N//2])/self.N*2 > max(self.fourier[0:self.N//2]/self.N*2)*0.01)[-1]
        sig_power = np.sum(abs(self.fourier[0:self.N//2][peaks]/self.N*2)**2)
        print('Signal Power:', sig_power)
    
    def modulate(self, cfreq, modtype = 'polar'):
        if modtype == 'polar':
            carrier = sig(cfreq)
            carrier.set_domain(self.lower_bound, self.upper_bound)
            self.time_eq = np.multiply(carrier.time_eq, self.time_eq)
         
        self.ft()
        self.plot()
        
    def normalize(self, norm = 1.):
        maxval = max(self.time_eq)
        self.time_eq = self.time_eq/maxval * float(norm)
        
        self.ft()
        self.plot()
    
    def block(self, width = 1, amp = 1, start = 1, avg = 0):
        self.time_eq = avg*np.ones(self.N)
        print(len(self.time_eq))
        print(len(self.time_eq[start*self.N:(start+width)*self.N]))
        self.time_eq[int(start/self.T):int((start+width)/self.T)] = (avg+amp)*np.ones(len(self.time_eq[int(start/self.T):int((start+width)/self.T)]))
        
        self.ft()
        self.plot()
                

class bitsig(sig, object):
    
    def __init__(self, bitlength = 1., bitamplitude = 1., p = 'unipolar'):
        super(bitsig, self).__init__()
        self.bitinput(bitlength, bitamplitude, p)
    
    def bitinput(self, bitlength = 1., bitamplitude = 1., p = 'unipolar'):
        bitstring = ' '.join(str(input('Enter bits as a sequence of 0s and 1s without spaces: ',)))
        bitarray = np.fromstring(bitstring, sep = ' ', dtype = int)
        print(bitarray)
        self.set_domain(0, len(bitarray)*bitlength)
        self.time_eq = np.zeros(len(self.domain))
        
        it = np.nditer(bitarray, flags=['f_index'])
        
        if p == 'unipolar':
            for elem in it:
                j = it.index
                i = j/len(bitarray)
                i1 = int(i*self.N)
                i2 = int((i+1)*self.N)
                
                if elem == 0:
                    self.time_eq[i1:i2] = np.zeros(len(self.time_eq[i1:i2]))
                if elem == 1:
                    print(i)
                    self.time_eq[i1:i2] = bitamplitude * np.ones(len(self.time_eq[i1:i2]))
        if p == 'polar':
            for elem in it:
                j = it.index
                i = j/len(bitarray)
                i1 = int(i*self.N)
                i2 = int((i+1)*self.N)
                
                if elem == 0:
                    self.time_eq[i1:i2] = -bitamplitude * np.ones(len(self.time_eq[i1:i2]))
                if elem == 1:
                    print(i)
                    self.time_eq[i1:i2] = bitamplitude * np.ones(len(self.time_eq[i1:i2]))
        self.ft()
        self.plot()
        
        
        
        
    

             
            
            
        
        