#!/usr/bin/env python
'''
Generates a Gaussian 3D distribution from the values in the file "gauss.json", to be read as input by Mithra.
Use with:
python generate_gauss.py
'''

import numpy as np
import json
import matplotlib.pyplot as plt
import pandas as pd
import sys

def gauss( x, mu = 0.0, sig = 1.0 ):
    return np.exp( - ( x - mu )**2 / ( 2 * sig**2 ) )

def xyGauss( sig = 1.0 ):
    accept = 0
    while not accept:
        x = 4 * sig * np.random.uniform(-1.0,1.0)
        y = 4 * sig * np.random.uniform(-1.0,1.0)
        accept = np.random.uniform() < gauss(np.sqrt(x**2 + y**2), sig = sig)
    return [x,y]

def generate_gaussian ( nPart = 1000, gamma = 100, direction = [0.,0.,1.], position = [0.,0.,0.],
                        sigmaPosition = [1.,1.,1.], sigmaMomentum = [1.,1.,1.],
                        transverseTruncation = 10., longitudinalTruncation = 10., filename = 'gauss.tsv', plot = False ):
    '''
    ------------------------------------------------------------------------------------------------
    Generates a Gaussian distribution of electrons and saves it as an input file for Mithra or OPAL
    ------------------------------------------------------------------------------------------------
    ------------------------------------------------------------------------------------------------
    Arguments obtained from json file 'gauss.json' 
    {
    	"nPart" : (integer),
        "gamma": (float) Average gamma of the bunch,
        "direction": (array of floats) Direction of bunch. Norm must be 1,
        "position": (array of floats) Average position of bunch,
        "sigmaPosition": (array of floats) Standard deviation of the bunch distribution,
        "sigmaMomentum": (array of floats) Standard deviation of the bunch's momentum distribution,
        "transverseTruncation": (float) Truncation of transverse distribution,
        "longitudinalTruncation": (float) Truncation of longitudinal distribution,
    	"filename" : (string) File in which to store distribution,
    	"plot" : (boolean) Plot the distrbution or not
    }
    ------------------------------------------------------------------------------------------------
    ------------------------------------------------------------------------------------------------
    Example:
    python3 generate_gauss.py
    python3 generate_gauss.py --help    
    '''
    assert abs(np.linalg.norm(direction) - 1) < 1e-5, print('Direction needs to have a norm = 1')
    nPart = int(nPart)

    x0 = position[0]
    y0 = position[1]
    z0 = position[2]
    bg = np.sqrt(gamma**2 - 1)
    px0 = direction[0] * bg
    py0 = direction[1] * bg
    pz0 = direction[2] * bg
    x = []
    y = []
    z = []
    px = []
    py = []
    pz  =[]
    
    for i in range(nPart):
        [xi,yi] = xyGauss()
        xi *= sigmaPosition[0]
        yi *= sigmaPosition[1]
        zi = np.random.randn() * sigmaPosition[2]
        [pxi,pyi] = xyGauss()
        pxi *= sigmaMomentum[0]
        pyi *= sigmaMomentum[1]
        pzi = np.random.randn() * sigmaMomentum[2]
        
        if  np.linalg.norm([xi,yi]) < transverseTruncation and abs(zi) < longitudinalTruncation:
            x.append(x0 + xi)
            y.append(y0 + yi)
            z.append(z0 + zi)
            px.append(px0 + pxi)
            py.append(py0 + pyi)
            pz.append(pz0 + pzi)
            
    # Save distribution in file
    ## Write particle number in first line
    print('\nSaving particles in file...')
    file = open( filename, mode = 'w' )
    file.write( str(nPart) + '\n' )
    file.close()
    ## Write the distribution
    df = pd.DataFrame([x,px,y,py,z,pz])
    df = df.T
    df.to_csv( filename, sep = '\t', header = False, index = False, mode = 'a' )

    # Plot distribution
    if plot:
        print('\nPlot distribution')
        nbins = int( nPart / 20 )
        fs = 12
        fig, ax = plt.subplots( 2, 4 )
        fig.set_size_inches(15, 10)
        ax[0,0].hist( x, bins = nbins )
        ax[0,0].set_xlabel('$x$ [m]', fontsize = fs)
        ax[0,1].hist( y, bins = nbins )
        ax[0,1].set_xlabel('$y$ [m]', fontsize = fs)
        ax[0,2].hist( z, bins = nbins )
        ax[0,2].set_xlabel('$z$ [m]', fontsize = fs)
        ax[1,0].hist( px, bins = nbins )
        ax[1,0].set_xlabel('$p_x$ [ ]', fontsize = fs)
        ax[1,1].hist( py, bins = nbins )
        ax[1,1].set_xlabel('$p_y$ [ ]', fontsize = fs)
        ax[1,2].hist( pz, bins = nbins )
        ax[1,2].set_xlabel('$p_z$ [ ]', fontsize = fs)

        ax[0,3].hist2d( x, y, bins = 100, cmin = 1 )
        ax[0,3].set_xlabel('$x$ [m]', fontsize = fs)
        ax[0,3].set_ylabel('$y$ [m]', fontsize = fs)
        ax[1,3].hist2d( px, py, bins = 100, cmin = 1 )
        ax[1,3].set_xlabel('$p_x$ [m]', fontsize = fs)
        ax[1,3].set_ylabel('$p_y$ [m]', fontsize = fs)

        plt.show()

        
# Run program
if __name__ == "__main__":
    for arg in sys.argv:
        if arg == '--help':
            help(generate_gaussian)
        elif arg.startswith('-'):
            print(arg, 'is not an option.')
            exit()
            
    jsonFile = 'gauss.json'
    print('Reading from', jsonFile)
    with open(jsonFile) as json_file:
        kwargs = json.load(json_file)
        print(kwargs)
        generate_gaussian(**kwargs)

