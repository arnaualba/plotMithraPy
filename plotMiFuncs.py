import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from decimal import Decimal
import os
from datetime import date
from scipy import ndimage, integrate, interpolate, signal, stats

units = {
    'T' : 1e-12,
    'G' : 1e-9,
    'M' : 1e-6,
    'K' : 1e-3,
    '' : 1,
    'm' : 1e3,
    '$\mu$' : 1e6,
    'n' : 1e9,
    'p' : 1e12,
    'f' : 1e15
}

colors = [ 'k', 'b', 'r', 'g', 'y', 'm', 'c' ]
fs = 14  # Default fontsize
clight = 3e8

sampNames = ['t', 'x', 'y','z', 'px', 'py','pz',
             'sig_x', 'sig_y', 'sig_z', 'sig_px', 'sig_py', 'sig_pz']
sampUnits = ['s', 'm', 'm', 'm', ' ', ' ', ' ',
             'm', 'm', 'm', ' ', ' ', ' ', 'm', ' ']

def importRadPower( fname, show = False ):
    ''' 
    Returns dataFrame
    -fname : (String) Filename to read data from
    -show  : (Boolean) Print column names
    '''
    names = ['z_lab', 'P']
    df = pd.read_csv(fname, sep='\t', header = None, index_col = False, names = names)
    df = df.dropna(axis = 'columns')
    if show:
        print(str(len(names)) + ' given names: ', names, '\n' , str(len(df.columns)), ' columns' )
    return df

def importStat( fname, show = False ):
    ''' 
    Returns dataFrame
    -fname : (String) Filename to read data from
    -show  : (Boolean) Print column names
    '''
    sampNames = ['t', 'x', 'y','z', 'px', 'py','pz',
                 'sig_x', 'sig_y', 'sig_z', 'sig_px', 'sig_py', 'sig_pz']
    sampUnits = ['s', 'm', 'm', 'm', ' ', ' ', ' ',
                 'm', 'm', 'm', ' ', ' ', ' ', 'm', ' ']
    df = pd.read_csv(fname, sep='\t', header = None, index_col = False, names = sampNames)
    df = df.dropna()
    if show:
        print( fname )
    return df

def plotStat( ax, df, quants, factors = [1,1], fs = fs, gamma_ = 1, z_lab_shift = 0, **kwargs):
    ''' 
    Plots given quantities on axis ax
    -ax : (matplotlib axis)
    -df : (dataFrame)
    -quants : (list of strings) Have to be a column names
    -factors : (list of floats)
    -gamma_ : (double > 1) gamma un undulator, used to compute z_lab
    -z_lab_shift : (double) When getting z_lab, a shift can be added to get the real position of z_lab
    '''
    quants = ['z_lab', 'z_lab'] + quants  # Plot z_lab if an empty list was given
    sampNames = ['t', 'x', 'y','z', 'px', 'py','pz',
                 'sig_x', 'sig_y', 'sig_z', 'sig_px', 'sig_py', 'sig_pz']
    sampUnits = ['s', 'm', 'm', 'm', ' ', ' ', ' ',
                 'm', 'm', 'm', ' ', ' ', ' ', 'm', ' ']
    # Create z_lab if necessary
    if 'z_lab' in quants[2:]:
        beta_ = np.sqrt(1 - 1 / gamma_**2)
        t = df['t']
        z = df['z']
        z_lab = [gamma_ * (z[i] + beta_ * clight * t[i]) for i,_ in enumerate(t)]
        if z_lab_shift != 0:
            z_lab += z_lab_shift - z_lab[0]
        df = df.assign(z_lab = z_lab)
        sampNames.append('z_lab')
        sampUnits.append('m')
        
    # Get quantities to plot        
    x = np.array(df[quants[-2]])
    y = np.array(df[quants[-1]])

    # Get units
    factors = [1, 1] + factors
    x *= factors[-2]
    y *= factors[-1]

    # Get axis labels
    rev_units = dict(map(reversed, units.items()))
    labs = [ quants[-2] + ' [' + rev_units[factors[-2] ] + sampUnits[sampNames.index(quants[-2])] + ']',
             quants[-1] + ' [' + rev_units[factors[-1] ] + sampUnits[sampNames.index(quants[-1])] + ']' ]    

    # Plot
    ax.plot( x, y, **kwargs)
    ax.tick_params( axis = 'both', labelsize = fs )
    ax.ticklabel_format( axis = 'both', style = 'sci', scilimits = (-1, 3) )
    ax.set_xlabel( labs[0], fontsize = fs )
    ax.set_ylabel( labs[1], fontsize = fs )
    



def importProfile( fname, show = False ):
    ''' 
    Returns bunch profile at all timeStamps as a dataframe
    -fname : (String) Filename to read data from. Put # instead of numbers. eg tests/test2/bunch-profile/bunch-p#-#.txt
    -show  : (Boolean) Print info
    '''
    dirname = fname[:fname.rfind('/')]
    pNames = ['q', 'x', 'y', 'z', 'px', 'py', 'pz']
    if show:
        print( 'columns = ', pNames )

    # Get time stamps and number of processors
    time_stamps = []
    nump = 0
    for name in os.listdir( dirname ):
        ind1 = name.find('-')
        ind2 = name.rfind('-')
        nup = int(name[ ind1 + len('-p') : ind2 ])
        tis = int(name[ ind2 + len('-') : -len('.txt') ])
        if nup > nump:
            nump = nup
        if not tis in time_stamps:
            time_stamps.append( tis )
    time_stamps.sort()
    nump += 1
    if show:
        print( 'time steps = ', time_stamps )
        print( 'num processors = ', nump )
        
    # Get the data
    posp = fname.find('#')
    post = fname.rfind('#')
    data = []    
    for t, ti in enumerate(time_stamps):
        tname = fname[:post] + str( ti ) + fname[post+1:]
        if show:
            print(t, tname)
        for p in range(nump):
            pname = tname[:posp] + str(p) + tname[posp+1:]
            df = pd.read_csv(pname, sep='\t', header = None, names = pNames)
            df['timeStep'] = np.ones( len(df.index) ) * ti
            data.append( df  )

    return [pd.concat(data), time_stamps]


def plotProfile( ax, df, quants, gamma_ = 1, timeStep = 0, factors = [1,1], type = 'hist2d', frame = 'comoving', show_und = False, rb = 0, dt = 1, Lu = 1, nbins = 100, fs = 14, ls = '-', lw = 2, color = 0 ):
    '''
    Plots the data given. 
    -ax : (matplotlib axis)
    -df : (dataFrame)
    -quants : (list of strings) Have to be a column name or 'E'
    -gamma_ : (float) Speed of the comoving frame. Only necessary if E is to be plotted or if you want to show undulator
    -timeStep : (int) Timestep of the data to plot
    -factors : (list of floats)
    -type : (string) 
    -- 'hist2d' for colormap
    -- 'hist' for histogram of x axis
    -- 'scatter' for scatter plot
    -- 'mod' to get plot of modulation of y axis as line plot
    -frame : (string) 'comoving', 'lab', 'labSameTime'. Note that Energy is always in lab frame
    -show_und : (boolean) Show position of undulator at all times
    -rb : (float) Rb of undulator.
    -dt : (float) Simulation timestep. 
    -Lu : (float) Undulator length. Only necessary if show_und is true
    '''
    pNames = ['q', 'x', 'y', 'z', 'px', 'py', 'pz']
    pUnits = [' ', 'm', 'm', 'm', ' ', ' ', ' ' ]

    d_to_plot = df[ df['timeStep'] == timeStep ]
    beta_ = np.sqrt( 1 - 1 / np.power(gamma_,2.) )
    clight = 3e8
    time = ( timeStep + 1 ) * dt / gamma_  # Time comoving frame
    
    # Get energy in lab frame
    if 'E' in quants:
        mc2 = .511 * 1e6  # eV/c2 electron mass
        pNames = pNames + ['E']
        pUnits = pUnits + ['eV']
        px = np.array(d_to_plot['px'])
        py = np.array(d_to_plot['py'])
        pz = np.array(d_to_plot['pz'])
        E = []
        for i, bg in enumerate( zip( px, py, pz) ):
            g_i = np.sqrt( 1 + np.inner(bg,bg) )  # Get gamma of particle
            g_i = gamma_ * ( g_i + beta_ * bg[2] )  # Lorentz transfo on gamma
            E.append( mc2 * g_i )
        d_to_plot = d_to_plot.assign( E = E )
        
    # Get quantities in lab frame if necessary
    if frame != 'comoving':
        x  = np.array( d_to_plot['x']  )
        y  = np.array( d_to_plot['y']  )
        z  = np.array( d_to_plot['z']  )
        z_max = np.max( z )
        px = np.array(d_to_plot['px'])
        py = np.array(d_to_plot['py'])
        pz = np.array(d_to_plot['pz'])
        for i, bg in enumerate( zip( px, py, pz) ):
            deltaT = beta_ * gamma_ * ( z_max - z[i] )  # Get time shift to apply on particle s.t. they are at same time in lab frame
            # Lorentz transformations
            z[i] =  gamma_ * ( z[i] - gamma_ * rb + beta_ * clight * time )
            g_i = np.sqrt( 1 + np.inner(bg,bg) )  # Get gamma of particle
            pz[i] = gamma_ * ( pz[i] + beta_ * g_i )
            if frame == 'labSameTime':
                # Apply time shift
                x[i] += ( bg[0] / g_i ) * deltaT
                y[i] += ( bg[1] / g_i ) * deltaT
                z[i] += ( bg[2] / g_i ) * deltaT
        d_to_plot = d_to_plot.assign( x = x )
        d_to_plot = d_to_plot.assign( y = y )
        d_to_plot = d_to_plot.assign( z = z )
        d_to_plot = d_to_plot.assign( pz = pz )        

    # Get quantities to plot
    quants = ['z', 'z'] + quants  # Plot z if an empty list was given
    x = np.array( d_to_plot[quants[-2]] )
    y = np.array( d_to_plot[quants[-1]] )
    # Get units
    factors = [1, 1] + factors
    x *= factors[-2]
    y *= factors[-1]
    factors = np.abs(factors)
    # Get axis labels
    rev_units = dict(map(reversed, units.items()))
    labs = [ quants[-2] + ' [' + rev_units[factors[-2] ] + pUnits[pNames.index(quants[-2])] + ']',
             quants[-1] + ' [' + rev_units[factors[-1] ] + pUnits[pNames.index(quants[-1])] + ']' ]

    # Plot stuff
    if type == 'scatter':
        l = len(x)
        ax.scatter( x[::nbins], y[::nbins], marker = '.', color = 'C' + str(color), zorder = 2)
    elif type == 'hist2d':
        ax.hist2d( x, y, bins = nbins, cmin = 1 , cmap=plt.cm.jet, zorder = 2 )
    elif type == 'hist':
        ax.hist( x, bins = nbins, color = 'C' + str(color), zorder = 2 )
        labs[1] = 'Density [arb]'
    elif type == 'mod':
        bin_edges = np.linspace( np.min(x), np.max(x), nbins )
        bin_size = bin_edges[1] - bin_edges[0]
        digitized = np.digitize( x, bin_edges )  # Get data organised in bins
        bin_means = [y[digitized == i].mean() for i in range(1, len(bin_edges))]
        bin_means -= bin_means[0]
        labs[1] = '$\Delta$' + labs[1]
        ax.plot( bin_edges[1:] - .5*bin_size, bin_means, zorder = 2, lw = lw, ls = ls)

    if show_und and quants[-2] == 'z':
        if frame == 'comoving':
            undStart = gamma_ * rb - beta_ * clight * time
            undEnd = undStart + Lu / gamma_
        else:
            undStart = 0.0
            undEnd = undStart + Lu
        ax.axvspan( undStart * factors[-2], undEnd * factors[-2], color='yellow', zorder = 1)

    ax.tick_params( axis = 'both', labelsize = fs )
    ax.ticklabel_format( axis = 'both', style = 'sci', scilimits = (-1, 3) )
    ax.set_xlabel( labs[0], fontsize = fs )
    ax.set_ylabel( labs[1], fontsize = fs )
    ax.text( 1.2, .4, 'timeStep = ' + str(timeStep) + ',\n ref. frame: ' + frame, transform=ax.transAxes, fontsize = fs, ha='center' )

    return [x,y]


def importScreen( fname, index_screens = [], show = False, pNames = [] ):
    ''' 
    Returns bunch profile on all screens as a dataframe
    -fname : (String) Filename to read data from. Put # instead of numbers. eg tests/test2/bunch-screen/bunch-p#-screen#.txt
    -index_screens : (list of ints) Indices of screens to import. By default it will import all screens.
    -show  : (Boolean) Print info
    '''
    dirname = fname[:fname.rfind('/')]
    if len(pNames) == 0:
        # pNames = ['q', 'x', 'y', 't', 'px', 'py', 'pz']
        pNames = ['x', 'y', 't', 'px', 'py', 'pz']
    if show:
        print( 'columns = ', pNames )

    # Get number of processors and screens
    nums = 0
    nump = 0
    for name in os.listdir( dirname ):
        ind1 = name.find('-')
        ind2 = name.rfind('-')
        nup = int(name[ ind1 + len('-p') : ind2 ])
        nus = int(name[ ind2 + len('-screen') : -len('.txt') ])
        if nup > nump:
            nump = nup
        if nus > nums:
            nums = nus
    nump += 1
    nums += 1
    if show:
        print( 'Number of screens = ', str(nums), ', number of processors = ', str(nump) )
    
    for i,ind in enumerate(index_screens):  # Such that -1 indexing works
        if ind < 0:
            index_screens[i] = nums + ind
        
    # Get the data
    posp = fname.find('#')
    poss = fname.rfind('#')
    data = []
    screenPos = []
    for s in range(nums):
        if (not s in index_screens) and len(index_screens) > 0:
            continue
        sname = fname[:poss] + str( s ) + fname[poss+1:]
        if show:
            print(s, sname)
        for p in range(nump):
            pname = sname[:posp] + str(p) + sname[posp+1:]
            with open(pname) as f:
                first_line = f.readline()
                ind = first_line.find('=')
                if ind != -1:
                    spos = float( first_line[ ind + 1:] )
                else:
                    spos = s
            if not spos in screenPos:
                screenPos.append(spos)
            df = pd.read_csv(pname, sep='\t', skiprows = 1, header = None, names = pNames)
            df = df.reset_index()
            df['screenPos'] = np.ones( len(df.index) ) * spos
            df['screenNum'] = np.ones( len(df.index) ) * s
            data.append( df  )
    screenPos.sort()
    if show:
        print('Screens at ', screenPos)

    return [pd.concat(data), screenPos]


def importScreenXY( fname, index_screens = [], show = False, pNames = [], xquant = 't', yquant = 'E', index_screen = 0, reduce_factor = 1, sliceT = [] ):
    ''' 
    Returns x and y of the bunch screen data
    -fname : (String) Filename to read data from. Put # instead of numbers. eg tests/test2/bunch-screen/bunch-p#-screen#.txt
    -index_screens : (list of ints) Indices of screens to import. By default it will import all screens.
    -xquant : (string) Quantity you want (from pNames list or E for energy)
    -yquant : (string) Quantity you want
    -index_screen : (unsigned int) index of screen to get data from
    -reduce_factor : (double) proportion of random values to ignore in order to reduce computational memory
    -show  : (Boolean) Print info
    '''
    dirname = fname[:fname.rfind('/')]
    if len(pNames) == 0:
        # pNames = ['q', 'x', 'y', 't', 'px', 'py', 'pz']
        pNames = ['x', 'y', 't', 'px', 'py', 'pz']
    if show:
        print( 'columns = ', pNames )

    # Get number of processors and screens
    nums = 0
    nump = 0
    for name in os.listdir( dirname ):
        ind1 = name.find('-')
        ind2 = name.rfind('-')
        nup = int(name[ ind1 + len('-p') : ind2 ])
        nus = int(name[ ind2 + len('-screen') : -len('.txt') ])
        if nup > nump:
            nump = nup
        if nus > nums:
            nums = nus
    nump += 1
    nums += 1
    if show:
        print( 'Number of screens = ', str(nums), ', number of processors = ', str(nump) )
    
    for i,ind in enumerate(index_screens): # such that -1 indexing works
        if ind < 0:
            index_screens[i] = nums + ind
    if index_screen < 0:
        index_screen += nums
    x = np.empty(0)
    y = np.empty(0)
    t = np.empty(0)
    # Get the data
    posp = fname.find('#')
    poss = fname.rfind('#')
    data = []
    screenPos = []
    ogN = 0
    for s in range(nums):
        if (not s in index_screens) and len(index_screens) > 0:
            continue
        if s != index_screen:
            continue
        sname = fname[:poss] + str( s ) + fname[poss+1:]
        if show:
            print(s, sname)
        for p in range(nump):
            pname = sname[:posp] + str(p) + sname[posp+1:]
            with open(pname) as f:
                first_line = f.readline()
                ind = first_line.find('=')
                if ind != -1:
                    spos = float( first_line[ ind + 1:] )
                else:
                    spos = s
            if not spos in screenPos:
                screenPos.append(spos)
            if show:
                print('Reading', pname)
            df = pd.read_csv(pname, sep='\t', skiprows = 1, header = None, names = pNames)
            df = df.reset_index()
            if reduce_factor > 1:
                ogN += df.shape[0]
                d = []
                for i in range(df.shape[0]):
                    if np.random.rand() > 1 / reduce_factor:
                        d.append(i)
                df = df.drop(d)
            x = np.append(x,np.array(df[xquant]))
            if len(sliceT) == 2:
                t = np.append(t,np.array(df['t']))
            if yquant == 'E':
                mc2 = .511 * 1e6  # eV/c2 electron mass
                px = np.array(df['px'])
                py = np.array(df['py'])
                pz = np.array(df['pz'])
                E = []
                for i, bg in enumerate( zip( px, py, pz) ):
                    g_i = np.sqrt( 1 + np.inner(bg,bg) )  # Get gamma of particle
                    E.append( mc2 * g_i )
                y = np.append(y,np.array(E))
            elif yquant == 'none':
                y = np.empty(0)
            else:
                y = np.append(y,np.array(df[yquant]))
    screenPos.sort()
    if reduce_factor > 1:
        print(ogN, 'particles has been reduced to', len(x))
    if len(sliceT) == 2:
        t -= t.mean()
        t *= -1
        delI = []
        for i,ti in enumerate(t):
            if ti < sliceT[0] or ti > sliceT[1]:
                delI.append(i)
        x = np.delete(x, delI)
        y = np.delete(y, delI)
    if show:
        print('Screens at ', screenPos)

    return [x,y]


def plotScreen( ax, df, quants, screenNum = 0, factors = [1,1], limx = [], limy = [],
                type = 'hist2d', nbins = 100, fs = 14, ls = '-', lw = 2, color = 0, maxHH = .3 ):
    '''
    Plots the data given. 
    -ax : (matplotlib axis)
    -df : (dataFrame)
    -quants : (list of strings) Have to be a column name or 'E'
    -screenNum : (int) Number of the screen to plot
    -factors : (list of floats)
    -limx : (2 element list) xlimits
    -limy : (2 element list) ylimits
    -maxHH : (float [0,1]) max height of histogram in hist2d-hist
    -type : (string) 
    -- 'hist2d' for colormap
    -- 'hist' for histogram of x axis
    -- 'scatter' for scatter plot
    -- 'mod' to get plot of modulation of y axis as line plot
    -- 'hist2d-hist' for both
    -nbins : (int) Number of bins to use for the histograms and the modulation plot
    '''
    pNames = ['q', 'x', 'y', 't', 'px', 'py', 'pz']
    pUnits = [' ', 'm', 'm', 's', ' ', ' ', ' ' ]

    if screenNum < 0:
        screenNum = np.max(df['screenNum']) + 1 + screenNum

    if 'screenNum' in df.columns:
        d_to_plot = df[ df['screenNum'] == screenNum ]
        screenPos = d_to_plot.iloc[0]['screenPos']
    else:
        d_to_plot = df
        screenPos = screenNum

    clight = 3e8
    
    # Get energy
    if ('E' in quants) and not 'E' in df.columns:
        mc2 = .511 * 1e6  # eV/c2 electron mass
        px = np.array(d_to_plot['px'])
        py = np.array(d_to_plot['py'])
        pz = np.array(d_to_plot['pz'])
        E = []
        for i, bg in enumerate( zip( px, py, pz) ):
            g_i = np.sqrt( 1 + np.inner(bg,bg) )  # Get gamma of particle
            E.append( mc2 * g_i )
        d_to_plot = d_to_plot.assign( E = E )
    pNames = pNames + ['E']
    pUnits = pUnits + ['eV']

    # Get quantities to plot
    quants = ['t', 't'] + quants  # Plot t if an empty list was given
    x = np.array( d_to_plot[quants[-2]] )
    y = np.array( d_to_plot[quants[-1]] )
    if quants[-2] == 't':
        x -= x.mean()
    # Get units
    factors = [1, 1] + factors
    x *= factors[-2]
    y *= factors[-1]
    factors = np.abs(factors)
    # Get axis labels
    rev_units = dict(map(reversed, units.items()))
    if factors[-2] in rev_units:
        unitx = rev_units[factors[-2]]
    else:
        unitx = str(factors[-2]) + '*'
    if factors[-1] in rev_units:
        unity = rev_units[factors[-1]]
    else:
        unity = str(factors[-1]) + '*'
    labs = [ quants[-2] + ' [' + unitx + pUnits[pNames.index(quants[-2])] + ']',
             quants[-1] + ' [' + unity + pUnits[pNames.index(quants[-1])] + ']' ]
    # Remove particles out of limits
    if len(limx) == 2:
        rm_index = np.append(np.where(x < limx[0]), np.where(x > limx[1]))
        nbins = int(nbins * (1 - rm_index.size / x.size))
        x = np.delete(x, rm_index)
        y = np.delete(y, rm_index)
    if len(limy) == 2:
        rm_index = np.append(np.where(y < limy[0]), np.where(y > limy[1]))
        nbins = int(nbins * (1 - rm_index.size / x.size))
        x = np.delete(x, rm_index)
        y = np.delete(y, rm_index)

    # Plot stuff
    if type == 'scatter':
        ax.scatter( x[::nbins], y[::nbins], marker = '.', color = 'C' + str(color), zorder = 2)
    elif 'hist2d' in type:
        hi = ax.hist2d( x, y, bins = nbins, cmin = 1 , cmap=plt.cm.jet, zorder = 2)
        cbar = plt.colorbar(hi[3], ax = ax)
        cbar.set_label('Number of macro particles', fontsize = fs)
        if type == 'hist2d-hist':
            ax2 = ax.twinx()
            hist, xPoints = np.histogram(x, bins = nbins, density = True)
            xPoints += .5 * (xPoints[1] - xPoints[0])
            ax2.plot(xPoints[:-1], hist / np.max(hist) * maxHH, width = xPoints[1] - xPoints[0], color = 'C' + str(color))
            ax2.set_ylim(0, 1)
            ax2.tick_params(axis='y', right = False, labelright = False)
    elif type == 'hist':
        ax.hist( x, bins = nbins, color = 'C' + str(color), zorder = 2 )
        labs[1] = 'Density [arb]'
    elif type == 'mod':
        bin_edges = np.linspace( np.min(x), np.max(x), nbins )
        bin_size = bin_edges[1] - bin_edges[0]
        digitized = np.digitize( x, bin_edges )  # Get data organised in bins
        bin_means = [y[digitized == i].mean() for i in range(1, len(bin_edges))]
#        bin_means -= bin_means[0]
        bin_means -= np.array(bin_means).mean()
        labs[1] = '$\Delta$' + labs[1]
        ax.plot( bin_edges[1:] - .5*bin_size, bin_means, zorder = 2, lw = lw, ls = ls)

    ax.tick_params( axis = 'both', labelsize = fs )
    ax.ticklabel_format( axis = 'both', style = 'sci', scilimits = (-1, 3), useOffset = False )
    ax.set_xlabel( labs[0], fontsize = fs )
    ax.set_ylabel( labs[1], fontsize = fs )
#     ax.text( 1.2, .4, str(screenPos) + ' m', transform=ax.transAxes, fontsize = fs, ha='center' )

    return [x,y]


def plotScreenXY( ax, x, y, quants, factors = [1,1], limx = [], limy = [],
                  type = 'hist2d', nbins = 100, fs = 14, ls = '-', lw = 2, color = 0, maxHH = .3, enable_cbar = True, denomYhist = 2):
    '''
    Plots the data given. 
    -ax : (matplotlib axis)
    -x,y : quantities to plot
    -quants : (list of strings) Have to be a column name or 'E'
    -screenNum : (int) Number of the screen to plot
    -factors : (list of floats)
    -limx : (2 element list) xlimits
    -limy : (2 element list) ylimits
    -maxHH : (float [0,1]) max height of histogram in hist2d-hist
    -type : (string) 
    -- 'hist2d' for colormap
    -- 'hist' for histogram of x axis
    -- 'scatter' for scatter plot
    -- 'mod' to get plot of modulation of y axis as line plot
    -- 'hist2d-hist' for both
    -nbins : (int) Number of bins to use for the histograms and the modulation plot
    '''
    pNames = ['q', 'x', 'y', 't', 'px', 'py', 'pz', 'z']
    pUnits = [' ', 'm', 'm', 's', ' ', ' ', ' ', 'm' ]

    clight = 3e8
    pNames = pNames + ['E']
    pUnits = pUnits + ['eV']

    # Get quantities to plot
    if len(quants) != 2:
        print('error, 2 quantities need to be given in quants = []')
    if quants[-2] == 't':
        x -= x.mean()
    # Get units
    factors = [1, 1] + factors
    x *= factors[-2]
    y *= factors[-1]
    factors = np.abs(factors)
    # Get axis labels
    rev_units = dict(map(reversed, units.items()))
    if factors[-2] in rev_units:
        unitx = rev_units[factors[-2]]
    else:
        unitx = str(factors[-2]) + '*'
    if factors[-1] in rev_units:
        unity = rev_units[factors[-1]]
    else:
        unity = str(factors[-1]) + '*'
    labs = [ quants[-2] + ' [' + unitx + pUnits[pNames.index(quants[-2])] + ']',
             quants[-1] + ' [' + unity + pUnits[pNames.index(quants[-1])] + ']' ]
    # Remove particles out of limits
    if len(limx) == 2:
        rm_index = np.append(np.where(x < limx[0]), np.where(x > limx[1]))
        nbins = int(nbins * (1 - rm_index.size / x.size))
        x = np.delete(x, rm_index)
        y = np.delete(y, rm_index)
    if len(limy) == 2:
        rm_index = np.append(np.where(y < limy[0]), np.where(y > limy[1]))
        nbins = int(nbins * (1 - rm_index.size / x.size))
        x = np.delete(x, rm_index)
        y = np.delete(y, rm_index)

    # Plot stuff
    if type == 'scatter':
        ax.scatter( x[::nbins], y[::nbins], marker = '.', color = 'C' + str(color), zorder = 2)
    elif 'hist2d' in type:
        hi = ax.hist2d( x, y, bins = nbins, cmin = 1 , cmap=plt.cm.jet, zorder = 2)
        if enable_cbar == True:
            cbar = plt.colorbar(hi[3], ax = ax)
            cbar.set_label('Number of macro particles', fontsize = 10)
        if type == 'hist2d-hist':
            nbinsHist = 50
            ax2 = ax.twinx()
            hist, xPoints = np.histogram(x, bins = nbinsHist, density = True)
            FWHMx = getFWHM(hist, xPoints)
            stdx = np.std(x)
            xPoints += .5 * (xPoints[1] - xPoints[0])
            ax2.plot(xPoints[:-1], hist / np.max(hist) * maxHH, lw = lw, color = 'C' + str(color), ls = ls)
            ax2.set_ylim(0, 1)
            ax2.tick_params(axis='y', right = False, labelright = False)
            ax3 = ax.twiny()
            hist, xPoints = np.histogram(y, bins = nbinsHist, density = True)
            FWHMy = getFWHM(hist, xPoints, denom = denomYhist)
            stdy = np.std(y)
            binsize = xPoints[1] - xPoints[0]
            xPoints += .5 * binsize
            xPoints = np.insert(xPoints, 0, xPoints[0] - binsize)
            hist = np.insert(hist, 0, 0.0)
            hist = np.append(hist, 0.0)
            ax3.plot(1 - hist / np.max(hist) * maxHH, xPoints, lw = lw, color = 'C' + str(color), ls = ls)
            ax3.set_xlim(0, 1)
            ax3.tick_params(axis='x', top = False, labeltop = False)
    elif type == 'hist':
        ax.hist( x, bins = nbins, color = 'C' + str(color), zorder = 2 )
        labs[1] = 'Density [arb]'
    elif type == 'mod':
        bin_edges = np.linspace( np.min(x), np.max(x), nbins )
        bin_size = bin_edges[1] - bin_edges[0]
        digitized = np.digitize( x, bin_edges )  # Get data organised in bins
        bin_means = [y[digitized == i].mean() for i in range(1, len(bin_edges))]
        bin_edges = bin_edges[1:] - .5*bin_size
#        bin_means -= bin_means[0]
        bin_means -= np.array(bin_means).mean()
        labs[1] = '$\Delta$' + labs[1]
        ax.plot(bin_edges, bin_means, zorder = 2, lw = lw, ls = ls, color = 'C' + str(color))

    ax.tick_params( axis = 'both', labelsize = fs )
    ax.ticklabel_format( axis = 'both', style = 'sci', scilimits = (-2, 3), useOffset = False )
    ax.set_xlabel( labs[0], fontsize = fs )
    ax.set_ylabel( labs[1], fontsize = fs )
    if type == 'hist2d-hist':
        return [stdx, FWHMx, stdy, FWHMy]
    else:
        return [0,0,0,0]

def getFromSlurm( rowVar, line, show = False ):
    '''
    Returns target value if it is in given line.
    -rowVar : (dataframe) Row of dataframe with the variable info
    -line : (String) Line in which to look for value
    -show : (boolean) Show info if target found
    '''
    tgt = rowVar['var']
    if tgt == None and rowVar['kStart'] in line:
        start = line.find(rowVar['kStart'])
        lenStart = len(rowVar['kStart'])
        end = line.rfind(rowVar['kEnd'])
        tgt = float(line[start + lenStart : end])
        if show:
            print( rowVar['kStart'], tgt )
    return tgt
    
def importSlurm( fn, show = False ):
    '''
    Returns array of specific data read from the slurm file.
    -fname : (String) Foldername to read from
    -show  : (Boolean) Print info
    '''
    foundFile = 0
    for name in os.listdir( fn ):
        if 'slurm' in name:
            fn = str( fn + '/' + name)
            foundFile += 1
            if show:
                print('Found slurm file ', fn)
    if foundFile == 0:
        print('Error: no slurm file found')
        return [ ]
    if foundFile > 1:
        print('Error: more than one slurm file found')
        return [ ]
    
    # Variables that we want to get
    vars = [['rb', 'undulator is set at the point', '', None],
            ['lu', 'Undulator period = ', '', None],
            ['nP', 'Undulator length = ', '', None],
            ['K', 'Undulator parameter = ', '', None],
            ['gamma', 'Initial mean gamma of the bunch = ', '', None],
            ['dt', 'Time step for the field update is set to ', '', None]] 
    dfVars = pd.DataFrame( vars, columns = ['name','kStart', 'kEnd', 'var'] )
    dfVars = dfVars.set_index('name')

    # Get variables
    if show:
        print('Scanning ...')
    for line in open(fn):
        if not None in dfVars.values:
            break
        for i, row in dfVars.iterrows():
            tgt = getFromSlurm( row, line, show = show )
            if tgt !=  None:
                dfVars.loc[ [i], ['var'] ] = tgt

    dfVars = dfVars.drop( ['kStart', 'kEnd'], axis = 1 )
    return dfVars


def adjust_axes_limits( axs, axis = 'x' ):
    '''
    Adjusts all axis ranges to the one with the biggest range
    axs : (list or array af matplotlib axes)
    axis : (string) Indicates whether to adjust x or y axis
    '''
    crange = []
    for ax in axs:
        if axis == 'x':
            crange.append( np.ptp(ax.get_xlim()) )
        elif axis == 'y':
            crange.append( np.ptp(ax.get_ylim()) )

    def change_limits( ax, r, axis = 'x' ):
        if axis == 'x':
            ax.set_xlim( ax.get_xlim()[0] - r/2., ax.get_xlim()[1] + r/2.)
        elif axis == 'y':
            ax.set_ylim( ax.get_ylim()[0] - r/2., ax.get_ylim()[1] + r/2.)

    r = np.max(crange)
    for i, ax in enumerate(axs):
        change_limits( ax, r - crange[i], axis = axis  )

# WORK IN PROGRESS
# def getPotEnergy(x, y, z, px, py, pz):
#     N = len(x)
#     for i in range(N):
#         for j in range(i+1,N):
            
# def getKinEnergy(px, py, pz):
#     mc2 = .511 * 1e6  # eV/c2 electron mass
#     E = 0
#     for i, bg in enumerate( zip( px, py, pz) ):
#         g_i = np.sqrt( 1 + np.inner(bg,bg) )  # Get gamma of particle
#         E += mc2 * g_i 
#     return E

# def getEnergy(x, y, z, px, py, pz):
#     U = getPotEnergy(x,y,z,px,py,pz)
#     T = getKinEnergy(px,py,pz)
#     return [T,U]

def sliceError(file1, fileT, plot = False):
    '''
    Gives slice ernergy and energy spread errors between two files containing the slice energy
    '''
    df = pd.read_csv(file1, skiprows = 2, sep = ' ', header = None,
                     index_col = False, names = ['avgE', 'stdE'])
    dfT = pd.read_csv(fileT, skiprows = 2, sep = ' ', header = None,
                      index_col = False, names = ['avgE', 'stdE'])

    # Get rid of nans
    nanInds = []
    for i, row in df.iterrows():
        if pd.isna(row['avgE']):
            nanInds.append(i)
            print('nan row ', i, ' in file ', file1)
    for i, row in dfT.iterrows():
        if pd.isna(row['avgE']) and not (i in nanInds) :
            nanInds.append(i)
            print('nan row ', i, ' in file ', fileT)
    df = df.drop(nanInds)
    dfT = dfT.drop(nanInds)

    # Compute the errors
    errE = np.average(np.abs(df['avgE'] - dfT['avgE']))
    errEbar = np.std(np.abs(df['avgE'] - dfT['avgE']))
    errSpread = np.average(np.abs(df['stdE'] - dfT['stdE']))
    errSpreadbar = np.std(np.abs(df['stdE'] - dfT['stdE']))
    
    if plot:
        fig, ax = plt.subplots()
        x = np.arange(len(df['avgE']))
        ax.errorbar(x, df['avgE'], yerr = df['stdE'])
        ax.errorbar(x, dfT['avgE'], yerr = dfT['stdE'])
        ax.legend(['case', 'truth'])
        plt.show()
        
    return [errE, errSpread, errEbar, errSpreadbar]

def log_errorbary(ax, x, y, yerr, **kwargs):
    '''
    Plot errorbars but for log scale, where errors need to be transfromed:
    You are plotting x vs y, and in a cartesian coordinates plot you would use +-dy for error bars.
    But d(log(y)) != log(dy), which is what you usually get when plotting an errorbar in log scale.
    The correct way would be:
    d(log(y)) = 1 / ln(10) * dy / y
    '''
    dy = np.array(yerr)
    dy = 1 / np.log(10) * np.multiply(dy, 1/y)
    yerr = np.zeros([2,len(y)])
    yerr[0] = np.multiply( y, 1 - 1/(10**dy) )
    yerr[1] = np.multiply( y, 10**dy - 1 )

    ax.errorbar(x, y, yerr = yerr, **kwargs)

def getFWHM(hist, bin_edges, denom = 2):
    HM = np.max(hist) / denom
    right = bin_edges[-1]
    left = bin_edges[0]
    for i,edge in enumerate(bin_edges[1:-1]):
        if hist[i] < HM and hist[i+1] >= HM:
            left = edge
        elif hist[i] > HM and hist[i+1] <= HM:
            right = edge
            
    return right - left
    
def flatten_out_line(x,y):
    id2 = np.argmax(y)
    id1 = np.argmin(y)
    return (y[id2] - y[id1]) / (x[id2] - x[id1])


def getM (fn, crop = [], rot = 0, size_filter = 2, r = 0):
    '''
    Read .rimg file with filename fn
    Crop is optional, and should be 4 numbers between 0 and 1, meaning the percentage of image cut from left,right,top, bottom
    rot is how much the image is rotated in degrees
    size_filter is how many nearest neighbouring pixels are averaged
    returns the M matrix after neede transformations
    '''
    print('opening', fn)
    f = open(fn)
    for i, line in enumerate(f):
        if i == 0:
            txt = line.replace('\n', ' ')
            print(txt)
            start = txt.find('x:')
            end = txt[start:].find(' ')
            lenx = int(txt[start+2:start+end])
            start = txt.find('y:')
            end = txt[start:].find(' ')
            leny = int(txt[start+2:start+end])
            M = np.zeros((leny, lenx))
        else:
            txt = line.strip()
            txt = txt.split()
            for j in range(lenx):
                M[i-1,j] = txt[j]
    f.close()
    
    M = ndimage.rotate(M, rot)
    M = np.flip(M.T, axis = 1)
    M *= 1/np.max(M)
    M = ndimage.median_filter(M, size = size_filter)
    M -= np.mean(M)  # Subtract background
    M[M < 0] = 0
    print('shape after post-processing and cropping', M.shape)
    
    if len(crop) == 4:
        length = M.shape[1]
        M = M[:, int(crop[0] * length) : int(crop[1] * length)]
        height = M.shape[0]
        M = M[int(crop[2] * height) : int(crop[3] * height), :]
    else:
        # Automatic cropping
        height = M.shape[0]
        M = M[int(0.3 * height) : int(0.7 * height), :]
        length = M.shape[1]
        M = M[:, int(0.2 * length) : int(0.8 * length)]
        cg = ndimage.measurements.center_of_mass(M)
        cg = np.asarray(cg).astype(int)
        if r == 0:
            for r in range(5,np.min(M.shape)):
                if np.max(getR(M, r, cg)) <= 0:
                    break
        print('Cropped with r = ', r, 'pixels')
        crop = [(cg[1]-r) / length + 0.2,
                (cg[1]+r) / length + 0.2,
                (cg[0]-r) / height + 0.3,
                (cg[0]+r) / height + 0.3]
        print('Cropped at', '[{:.2f}, '.format(crop[0]), '{:.2f}, '.format(crop[1]),
              '{:.2f}, '.format(crop[2]), '{:.2f}]'.format(crop[3]))
        M = M[cg[0]-r:cg[0]+r, cg[1]-r:cg[1]+r]
    return M


def getR(M, r, cg):
    '''
    Get all pixels on the square at distance r from cg
    cg is the center of mass
    r is the radius around cg to start from
    '''
    length = M.shape[1]
    height = M.shape[0]
    if cg[1]-r < 0 or cg[1]+r >= length or cg[0]-r < 0 or cg[0]+r >= height:
        print('R scan has reached matrix limit')
        return np.array([0,0])
    R = []
    for i in range(cg[0]-r, cg[0]+r+1):
        R.append(M[i,cg[1]-r])
        R.append(M[i,cg[1]+r])
    for i in range(cg[1]-r, cg[1]+r+1):
        R.append(M[cg[0]-r,i])
        R.append(M[cg[0]+r,i])
    return np.array(R)

              
def integrateMatrix(M, axis = 0):
    '''
    Sum of all points on line or column 
    '''
    Mloc = M
    if axis == 1:
        Mloc = Mloc.T
    vec = np.zeros(Mloc.shape[1])
    for i in range(Mloc.shape[1]):
        vec[i] = np.sum(Mloc[:,i])
    return vec


def histAx2(ax, M, axis = 0, xlims = [], maxHH = .3, plot = True, shift = 0, cutoffs = [0,-1], show_cutoffs = False, **kwargs):
    '''
    get histogram of array M along an axis.
    xlims should be two (or four) points, which are the extents of the real-life distance that the matrix represents
    if plot, the histogram will be plotted as a line-plot on the matplotlib axis ax
    maxHH is a number from 0 to 1 saying how high theline plot should be when plotted
    shift is the number of matrix points to shift by the histogram
    - cutoffs is a 2 element list with cutoffs to consider in the histogram. Then the histogram is only returned within those cutoffs.
    The cutoffs should be in pixel number.
    - set show_cutoffs to True if you want to visualise the cutoffs
    '''

    def apply_cutoff(h,x,c):
        if c[1] == -1:
            h = h[c[0]:]
            x = x[c[0]:]
        else:
            h = h[c[0]:c[1]]
            x = x[c[0]:c[1]]
        return h,x
        
    hist = integrateMatrix(M, axis = axis)
    hist = ndimage.interpolation.shift(hist, shift, cval = 0.0)
    if axis == 0:
        xPoints = np.linspace(xlims[0], xlims[1], len(hist) + 1)[:-1]
        xPoints += .5 * (xPoints[1] - xPoints[0])
        hist, xPoints = apply_cutoff(hist,xPoints,cutoffs)
        if plot:
            ax.plot(xPoints, hist / np.max(hist) * maxHH * np.diff(xlims[-2:]) + xlims[2], **kwargs)
            if show_cutoffs and cutoffs != [0,-1]:
                ax.axvline(xPoints[0], lw = 3, color = 'orange')
                ax.axvline(xPoints[-1], lw = 3, color = 'orange')
    elif axis == 1:
        hist = np.flip(hist)
        xPoints = np.linspace(xlims[-2], xlims[-1], len(hist) + 1)[:-1]
        xPoints += .5 * (xPoints[1] - xPoints[0])
        hist, xPoints = apply_cutoff(hist,xPoints,cutoffs)
        if plot:
            ax.plot(0.999*xlims[1] - hist / np.max(hist) * maxHH * np.diff(xlims[:2]), xPoints, **kwargs)
            if show_cutoffs and cutoffs != [0,-1]:
                ax.axhline(xPoints[0], lw = 3, color = 'orange')
                ax.axhline(xPoints[-1], lw = 3, color = 'orange')
    return [xPoints, hist]


def histAx(ax, x, axis = 0, bins = 50, maxHH = .3, flip = True, **kwargs):
    '''
    Plots a histogram of x (an array of points) on ax, with maximum height maxHH (percentage)
    '''
    [hist,bs] = np.histogram(x, bins = bins)
    xlims = np.array([np.min(x), np.max(x)])
    if axis == 0:
        ax2 = ax.twinx()
        bs = bs[:-1]
        bs += .5 * (bs[1] - bs[0])
        ax2.plot(bs, hist / np.max(hist) * maxHH, **kwargs)
        ax2.set_ylim(0, 1)
        ax2.tick_params(axis='y', right = False, labelright = False)
    elif axis == 1:
        if flip:
            hist = np.flip(hist)
        ax2 = ax.twiny()
        bs = bs[:-1]
        bs += .5 * (bs[1] - bs[0])
        ax2.plot(1 - hist / np.max(hist) * maxHH, bs, **kwargs)
        ax2.set_xlim(0, 1)
        ax2.tick_params(axis='x', top = False, labeltop = False)
    return [bs, hist]


def getFWHM(x, hist, denom = 2):
    '''
    Returns FWHM of quantity x (an array of points)
    It also returns the right and left edges of the FWHM
    In fact, the returned quantity is the width of the distibution at maximum/denom. So for denom=2 we get FWHM
    '''
    idx = np.argmax(hist)
    HM = np.max(hist) / denom
    right = x[-1]
    left = x[0]
    for i in range(idx, len(hist)):
        if hist[i] < HM:
            right = x[i]
            break
    for i in range(idx, -1, -1):
        if hist[i] < HM:
            left = x[i]
            break
    FWHM = right - left
    return [FWHM,[right,left]]


def getFWHmean(x, hist, denom = 2):
    '''
    Same as getFWHM but uses the mean instead of the maximum
    '''
    mean = np.sum(hist * x) / np.sum(hist)
    idx = np.argmin(np.abs(x - mean))
    HM = hist[idx] / denom
    right = x[-1]
    left = x[0]
    for i in range(idx, len(hist)):
        if hist[i] < HM:
            right = x[i]
            break
    for i in range(idx, -1, -1):
        if hist[i] < HM:
            left = x[i]
            break
    FWHM = right - left
    return [FWHM,[right,left]]


def getRms(x, hist):
    '''
    Returns the rms of x (an array of points)
    '''
    mean = 0
    mom2 = 0
    N = 0
    for i, n in enumerate(hist):
        mean += x[i] * n
        mom2 += x[i]**2 * n
        N += n
    return np.sqrt(mom2 / N - (mean / N)**2)


def integrateCharge(fn, size_filter = 8, show = False):
    '''
    Reads a .rict file and integrates it to get the total charge
    Size filter is how many array points to average out to remove high_frequencies
    If show, it plots the .rict charge file
    '''
    factor = 1.25  # ICT calibration factor
    df = pd.read_csv(fn, skiprows = 1, sep = r'\s+', names = ['t', 'Q'])
    t = df['t']
    Q = df['Q']
    Q = ndimage.median_filter(Q, size = size_filter)
    Q -= np.mean(Q)
    idxmin = np.argmin(Q)
    lims = np.zeros(2)
    for i in range(idxmin, len(Q)-1):
        if Q[i] > 0:
            lims[1] = t[i]
            break
    for i in range(idxmin, -1, -1):
        if Q[i] > 0:
            lims[0] = t[i]
            break
    for i,ti in enumerate(t):
        if ti < lims[0] or ti > lims[1]:
            Q[i] = 0
    if show:
        fig, ax = plt.subplots()
        ax.plot(t,Q)
        ax.axvline(lims[0], color = 'green')
        ax.axvline(lims[1], color = 'green')
        plt.show()
    return integrate.trapz(x = t, y = Q) / factor


def trafo_dist_M(M1, M2, axis = 0, plot = False, shift = 0):
    '''
    - M1 and M2 are made to be equal along axis. If plot, we see the before and after images of the histograms
    Returns M, which has been modified based on M2
    M1 and M2 need to be in the same scale
    '''
    if axis == 0:
        M1 = M1.T
        M2 = M2.T

    # Get hists for zoom
    hist1 = integrateMatrix(M1, axis = 1)
    xs = np.linspace(0, 1, len(hist1), endpoint = False)
    [f1,_] = getFWHM(xs, hist1)
    hist2 = integrateMatrix(M2, axis = 1)
    xs = np.linspace(0, 1, len(hist2), endpoint = False)
    [f2,_] = getFWHM(xs, hist2)
    M1 = zoom_M(M1, f2/f1, axis = 1)
                        
    # Adjust hist2 to have same number of pixels as hist2
    hist1 = integrateMatrix(M1, axis = 1)
    xs = np.linspace(0,1,len(hist2), endpoint = False) + 1/(2*len(hist2))
    hist2 = np.append(hist2, np.array([hist2[0],hist2[-1]]))
    xs = np.append(xs, np.array([0.,1.]))
    h2 = interpolate.interp1d(xs , hist2, kind='quadratic')
    xs = np.linspace(0,1,len(hist1), endpoint = False) + 1/(2*len(hist1))
    hist2 = np.array([h2(i) for i in xs])
    
    # Shift hists to agree
    hist2 = ndimage.interpolation.shift(hist2, shift, cval = 0.0)
    
    # Add new hist
    for i,_ in enumerate(hist1):
        if hist1[i] != 0:
            M1[i,:] *= hist2[i] / hist1[i]

    if axis == 0:
        M1 = M1.T
        M2 = M2.T
        
    # Plot the hists
    if plot:
        hist2 = hist2 / hist2.max() * hist1.max()  # Normalise just for nicer plot
        figloc, axloc = plt.subplots()
        axloc.plot(hist1)
        axloc.plot(hist2)
        axloc.legend(['hist1', 'hist2'])
        
    return M1


def get_stats(fn, show = False):
    '''
    Get stats from the OPAL statfile
    returns the stats as a dataframe, and also the list of names and units as strings
    '''
    print(fn)
    text = open( fn, 'r' )
    names = []
    units = []
    for line in text:
        if 'name' in line:
            ind1 = line.find('=')
            ind2 = line.find(',')
            name = line[ind1+1:ind2]
        if 'units' in line:
            ind1 = line.find('=')
            ind2 = line.find(',')
            unit = line[ind1+1:ind2]
            if unit == '1':
                unit = ' '
            names.append( name )
            units.append( unit )
    if show:
        for i, n in enumerate(names):        
            print( i, n, ' [', units[i], ']' )
    stat = pd.read_csv( filepath_or_buffer = fn,
                        skiprows = 279, sep = '\\s+', names = names)
    return[stat, names, units]


def g_from_p(px,py,pz):
    'gamma from momentum vector'
    return np.sqrt(1 + px**2+py**2+pz**2)


def get_stats_mithra(fn, E, shift, K = 10.81):
    '''
    - fn: file with undulator.stat
    - E: energy in MeV of beam
    - sOpal: array of s from the OPAL stat file
    - sig_yOpal: array of sig_y from the OPAL stat file
    - K: Undulator strength parameters
    '''
    if os.path.isfile(fn) == True:
        # Get stats
        print('Getting stats from Mithra at', fn)
        statMi = pmf.importStat(fn, show = False)
        t = statMi['t']
        z = statMi['z']
        sigz = statMi['sig_z']
        pz = statMi['pz']
        sigpz = statMi['sig_pz']
        x = statMi['x']
        sigx = statMi['sig_x']
        px = statMi['px']
        sigpx = statMi['sig_px']
        y = statMi['y']
        sigy = statMi['sig_y']
        py = statMi['py']
        sigpy = statMi['sig_py']
        
        gamma_ = E / .511 / np.sqrt(1 + .5 * K**2)
        beta_ = np.sqrt(1 - 1 / gamma_**2)
        clight = 3e8
        
        # Lorentz transforms
        z = [gamma_ * (z[i] + beta_ * clight * t[i]) for i,_ in enumerate(t)]
        z = z - z[0] + shift
        
        stat = pd.DataFrame({'s': z, 'rms_x': sigx, 'rms_y': sigy})
    else:
        stat = pd.DataFrame()
        print('No Mithra file found')
    
    return stat


def get_transverse_params(fn, L, l = 0.05, plot = False, Qs = [], startpoint = 0.0, ax = []):
    '''
    Takes that statfile fn.
    Then it takes sigx,sigy,sigpx,sigpy at point startpoint. (startpoint needs to be within the range of the stat file simulation
    Then, using transfer matrices, it transports the transverse parameters for a length L
    l is the time-step, and it can be negative for back-tracking
    It uses drifts of length l, but can also do quadrupoles. 
    Qs is the list of Quadrupole positions and strengths in T/m (remark: There is a factor that might need adjusting in this function)
    Qs is a list of tuples or arrays, each of two elements = [pos Q, strength Q]
    It returns the final transverse beam parameters
    If plot is true it also makes a plot of the transverse parameters
    Give an ax if you want to use your own axes
    '''
    [stats,_,_] = get_stats(fn, show = False)
    
    
    steps = np.linspace(0.0, L,int(L // np.abs(l)))
    if l < 0.0:
        steps = np.flip(steps)
    if startpoint > 0.0:
        idx = np.argmin(abs(steps - startpoint))
        steps = steps[idx:]
        idx = np.argmin(abs(stats['s'] - startpoint))
    else:
        idx = 0
        
    # Get initial parameters
    gamma = stats['energy'][idx] / .511
    print('gamma', gamma)
    x = stats['rms_x'][idx]
    px = stats['rms_px'][idx] / gamma
    xpx = stats['xpx'][idx] * x * px
    y = stats['rms_y'][idx]
    py = stats['rms_py'][idx] / gamma
    ypy = stats['ypy'][idx] * y * py

    vec = np.array([[x**2, xpx, 0., 0.], [xpx, px**2, 0., 0.],
                  [0.,0.,y**2, ypy],[0.,0., ypy, py**2]])
    print('Initial state\n', vec)
    
    # Set up transfer matrix
    M = np.diag(np.ones(4)) + np.diag([l, 0., l], k = 1)
    print('Transfer matrix M\n', M)
    
    # Set up quad matrices
    factor = .6
    for i,Q in enumerate(Qs):
        Qs[i] = [Q[0], np.diag(np.ones(4)) + Q[1] * factor * np.diag([1., 0., -1.], k = -1)]
    if len(Qs) > 0:
        print('Example quadrupole matrix Q\n', Qs[0][1])
    
    # Time evolution of the bunch
    x = np.zeros([len(steps),3])
    y = np.zeros([len(steps),3])
    for i,step in enumerate(steps):
        x[i,:] = np.array([np.sqrt(vec[0,0]), np.sqrt(vec[1,1]) * gamma, 
                           vec[0,1] / np.sqrt(vec[0,0] * vec[1,1])])
        y[i,:] = np.array([np.sqrt(vec[2,2]), np.sqrt(vec[3,3]) * gamma, 
                           vec[2,3] / np.sqrt(vec[2,2] * vec[3,3])])     
        if len(Qs) > 0  and Qs[0][0] <= step:
            Q = Qs.pop(0)[1]
            vec = np.dot(np.dot(Q,vec),Q.T)
        else:
            vec = np.dot(np.dot(M,vec),M.T)
              
    # Plot
    if plot:
        lw = 5
        if len(ax) == 0:
            fig, ax = plt.subplots(1,3,figsize = (12,5))
            ax = ax.reshape(-1)
        ax[0].plot(steps, x[:,0], lw = lw-2, color = 'red')
        ax[0].plot(steps, y[:,0], lw = lw-2, color = 'blue')
        ax[0].plot(stats['s'][idx:], stats['rms_x'][idx:], lw = lw, ls = '--', color = 'green')
        ax[0].plot(stats['s'][idx:], stats['rms_y'][idx:], lw = lw, ls = '--', color = 'black')
        ax[0].set_ylim(top = 1.1*np.max(stats['rms_y']))
        
        ax[1].plot(steps, x[:,1], lw = lw-2, color = 'red')
        ax[1].plot(steps, y[:,1], lw = lw-2, color = 'blue')
        ax[1].plot(stats['s'][idx:], stats['rms_px'][idx:], lw = lw, ls = ':', color = 'green')
        ax[1].plot(stats['s'][idx:], stats['rms_py'][idx:], lw = lw, ls = ':', color = 'black')
        
        ax[2].plot(steps, x[:,2], lw = lw-2, color = 'red')
        ax[2].plot(steps, y[:,2], lw = lw-2, color = 'blue')
        ax[2].plot(stats['s'][idx:], stats['xpx'][idx:], lw = lw, ls = ':', color = 'green')
        ax[2].plot(stats['s'][idx:], stats['ypy'][idx:], lw = lw, ls = ':', color = 'black')
        
    return np.concatenate((x[-1,:],y[-1,:]))

        
def generate_transverse_phase_space(LPSFile, trans, outFile, E = 45.4, plot = False, 
                                    plotFile = 'dummy.png', lw = 3, color = 2):
    '''
    Takes as input a csv file of the LPS, i.e. z and pz list of particles
    It also takes the transverse beam parameters as an array sig[x,px,xpx,y,py,ypy]
    It assumes that x,y,px,py on average are 0
    It plots the resulting phase-space histogram in plotfile
    It returns an outFile with x,px,y,py,z,pz, and the number of particles in the first line (file ready for OPAL)
    At the moment it takes the energy, but actually it should compute it from pz
    '''
    
    df = pd.read_csv(LPSFile, sep = r"\s+", skiprows = 1, names = ['z', 'pz'])
    N = len(df['z'])
    print("number of particles is", N)
    gamma = E / .511

    [x,px,xpx,y,py,ypy] = trans
    cov = np.array([ [x**2, xpx * x * px],
                     [xpx * x * px, px**2] ])
    dat = np.random.multivariate_normal([0.0, 0.0], cov, N)
    df['x'] = dat[:,0]
    df['px'] = dat[:,1]

    cov = np.array([ [y**2, ypy * y * py],
                     [ypy * y * py, py**2] ])
    dat = np.random.multivariate_normal([0.0, 0.0], cov, N)
    df['y'] = dat[:,0]
    df['py'] = dat[:,1]
    
    if plot:
        fig,axs = plt.subplots(1, 3, figsize = (22,6))
        fig.subplots_adjust(wspace=.25)
        axs = axs.reshape(-1)
        dfloc = df.copy()
        pmf.plotScreenXY(axs[0], dfloc['x'], dfloc['px'], ['x', 'px'], 
                         type = 'hist2d-hist', factors = [1e3, 1], 
                         nbins = 200, color = color, maxHH = .2, enable_cbar = 0, lw = lw)
        pmf.plotScreenXY(axs[1], dfloc['y'], dfloc['py'], ['y', 'py'], 
                         type = 'hist2d-hist', factors = [1e3, 1], 
                         nbins = 200, color = color, maxHH = .2, enable_cbar = 0, lw = lw)
        pmf.plotScreenXY(axs[2], dfloc['z'], dfloc['pz'], ['z', 'pz'], 
                         type = 'hist2d-hist', factors = [1e3, 1], 
                         nbins = 200, color = color, maxHH = .2, enable_cbar = 0, lw = lw)
        
        for i in range(3):
            axs[i].tick_params(axis = 'both', labelsize = fs)
        
        plt.savefig((pltpath + '/' + plotFile),bbox_inches='tight')
        plt.show()

    # Save distribution in file
    ## Write particle number in first line
    print('\nSaving particles in file...')
    file = open(outFile, mode = 'w')
    file.write(str(N) + '\n')
    file.close()
    ## Write the distribution
    df = df[['x', 'px', 'y', 'py', 'z', 'pz']]
    df.to_csv(outFile, sep = '\t', header = False, index = False, mode = 'a')


def remove_bump(oldM, axis = 1, plot = False):
    '''
    Removes bump only from histogram on axis 1
    It returns the fixed M and the ratio of integration of the curve that has changed, so that the charge can be divided by this too
    '''

    if axis == 0:
        oldM = oldM.T
        
    # Get the hist
    extent = np.array([-0.5, oldM.shape[1] - 0.5, -0.5, oldM.shape[0] - 0.5])
    [_,oldHist] = histAx2(None, M = oldM, axis = 1, xlims = extent, plot = False)
    N = len(oldHist)
    print(N, 'and', oldM.shape)
    newHist = np.copy(oldHist)
    
    # Choose peak closest to centre
    [peaks, _] = signal.find_peaks(oldHist, prominence = 1)
    idx = 1
    peak = peaks[idx]
    print("Removing peak at index", peak)
    [valleys, _] = signal.find_peaks(-oldHist, prominence = 1)
    w2 = valleys[-1] - peak
    keep_idx = [idx for idx in range(N) if idx < peak-w2-5 or idx > peak+w2+5]
    
    # Interpolate new points at place were peak is removed
    f = interpolate.interp1d(keep_idx, oldHist[keep_idx], kind='quadratic')
    
    # Create new hist
    for i in range(N):
        if i not in keep_idx:
            newHist[i] = f(i)
            
    # Compute ratio between integrals
    ratio = np.sum(newHist) / np.sum(oldHist)
    
    # Create newM with correct distribution on axis 1, and dummy disto on the axis 0
    newM = np.zeros(oldM.shape)
    newM[:,0] = np.flip(newHist)
    
    # Now adjust matrix to have newHist
    newM = trafo_dist_M(oldM, newM, axis = 1, plot = plot)

    if axis == 0:
        newM = newM.T

    return [newM, ratio]


def average_matrix(Ms, Qs):
    '''
    First of all the centres of mass of the Ms are aligned
    Then they are averaged, with weights. If Q is close to -300 pCthe weight is larger
    Important: All images must have the same extent and size!!!
    '''
    Qs = np.array(Qs)
    Ms = np.array(Ms)
    newM = Ms[0,:,:]
    cm = ndimage.measurements.center_of_mass(newM)
    cm = np.asarray(cm).astype(int)

    # Align all images around the same centre
    for i in range(Ms.shape[0]):
        cmloc = ndimage.measurements.center_of_mass(Ms[i,:,:])
        cmloc = np.asarray(cmloc).astype(int)
        Ms[i,:,:] = ndimage.interpolation.shift(Ms[i,:,:], cm-cmloc, cval = 0.0)
    
    # Now do a weighted average
    newM = np.average(Ms, axis = 0, weights = np.abs(Qs+300)**(-1))
    newQ = np.average(Qs, weights = np.abs(Qs+300)**(-1))
            
    return newM, newQ


def save_matrix(M, extent, fn):
    with open(fn, mode = 'w') as f:
        np.savetxt(f,extent)
        np.savetxt(f,M)


def load_matrix(fn):
    with open(fn, mode = 'r') as f:
        extent = np.zeros(4)
        for i in range(4):
            extent[i] = float(f.readline())
            
    M = np.loadtxt(fn, skiprows = 4)
    return M, extent


def sample_3Ms(axs, LPS_fn, specton_fn, TDCon_fn, out_fn, Npart, casename = '',
               E = 45.4, TDC = 2.872, spec = 365, show_info = False,
               plot_trafos = False, sample = False):

    maxHH = 0.3
    axs = axs.reshape(-1)

    print("Getting LPS")
    MLPS, extent = load_matrix(LPS_fn)
    MLPS = MLPS.T
    extent = np.take_along_axis(extent, np.array([2,3,0,1]), axis = 0)
    extent[2:] *= E/spec
    extent[2:] -= np.mean(extent[2:])
    extent[2:] += E
    extent[:2] *= 1/TDC
    extentLPS = extent.copy()
    MLPS = centre_matrix(MLPS)

    axs[0].imshow(MLPS, extent = extent, aspect = 'auto')
    axs[0].tick_params(axis = 'both', labelsize = fs)
    [x,hist] = histAx2(axs[0], M = MLPS, axis = 0, xlims = extent,
                           maxHH = maxHH, color = 'orange', lw = 3)
    rmsz = getRms(x, hist)
    [x,hist] = histAx2(axs[0], M = MLPS, axis = 1, xlims = extent,
                           maxHH = maxHH, color = 'orange', lw = 3)
    [fwhmE,_] = getFWHM(x, hist)
    axs[0].set_ylabel('E [MeV]', fontsize = fs)
    axs[0].set_xlabel(r'z [mm]', fontsize = fs)
    axs[0].tick_params(axis = 'both', labelsize = fs)
    if show_info:
        axs[0].text(.01, .5, casename + '\n' +
                   'FWHM$_E$ = {:.2f} MeV\n'.format(fwhmE) +
                   '$\sigma_z$ = {:.2f} $\mu$m'.format(rmsz*1e3),
                   fontsize = fs, color = 'orange', transform = axs[0].transAxes)

    print("Getting Spect only")
    Mspec, extent = load_matrix(specton_fn)
    Mspec = Mspec.T
    extent = np.take_along_axis(extent, np.array([2,3,0,1]), axis = 0)
    extent[2:] *= E/spec
    extent[2:] -= np.mean(extent[2:])
    extent[2:] += E
    Mspec = centre_matrix(Mspec)
    Mspec = zoom_M(Mspec, (extent[3]-extent[2])/(extentLPS[3]-extentLPS[2]), axis = 1)
    extent[2:] = extentLPS[2:]
    
    axs[1].imshow(Mspec, extent = extent, aspect = 'auto')
    axs[1].tick_params(axis = 'both', labelsize = fs)
    [x,hist] = histAx2(axs[1], M = Mspec, axis = 1, xlims = extent,
                           maxHH = maxHH, color = 'orange', lw = 3)
    [fwhmE,_] = getFWHM(x, hist)
    axs[1].set_ylabel('E [MeV]', fontsize = fs)
    axs[1].set_xlabel(r'y [mm]', fontsize = fs)
    axs[1].tick_params(axis = 'both', labelsize = fs)
    if show_info:
        axs[1].text(.01, .5, casename + '\n' +
                   'FWHM$_E$ = {:.2f} MeV\n'.format(fwhmE),
                   fontsize = fs, color = 'orange', transform = axs[1].transAxes)
        
    print("Getting TDC only")
    MTDC, extent = load_matrix(TDCon_fn)
    MTDC = MTDC.T
    extent = np.take_along_axis(extent, np.array([2,3,0,1]), axis = 0)
    extent[:2] *= 1/TDC
    MTDC = centre_matrix(MTDC)
    MTDC = zoom_M(MTDC, (extent[1]-extent[0])/(extentLPS[1]-extentLPS[0]), axis = 0)
    extent[:2] = extentLPS[:2]
    
    axs[2].imshow(MTDC, extent = extent, aspect = 'auto')
    axs[2].tick_params(axis = 'both', labelsize = fs)
    [x,hist] = histAx2(axs[2], M = MTDC, axis = 0, xlims = extent,
                           maxHH = maxHH, color = 'orange', lw = 3)
    rmsz = getRms(x, hist)
    axs[2].set_ylabel('x [mm]', fontsize = fs)
    axs[2].set_xlabel(r'z [mm]', fontsize = fs)
    axs[2].tick_params(axis = 'both', labelsize = fs)
    if show_info:
        axs[2].text(.01, .5, casename + '\n' +
                   '$\sigma_z$ = {:.2f} $\mu$m'.format(rmsz*1e3),
                   fontsize = fs, color = 'orange', transform = axs[2].transAxes)
        

    # Replace E distribution in MLPS with distribution from Mspec
    print('Replacing the E distribution with the one from the spectrometer')
    MLPS = trafo_dist_M(MLPS, Mspec, axis = 1, plot = plot_trafos)
    MLPS = trafo_dist_M(MLPS, MTDC, axis = 0, plot = plot_trafos)
    MLPS = trafo_dist_M(MLPS, Mspec, axis = 1, plot = plot_trafos)
    MLPS = trafo_dist_M(MLPS, MTDC, axis = 0, plot = plot_trafos)

    if not sample:
        axs[3].imshow(MLPS, extent = extentLPS, aspect = 'auto')
        axs[3].tick_params(axis = 'both', labelsize = fs)
        [x,hist] = histAx2(axs[3], M = MLPS, axis = 0, xlims = extentLPS,
                           maxHH = maxHH, color = 'orange', lw = 3)
        rmsz = getRms(x, hist)
        [x,hist] = histAx2(axs[3], M = MLPS, axis = 1, xlims = extentLPS,
                           maxHH = maxHH, color = 'orange', lw = 3)
        [fwhmE,_] = getFWHM(x, hist)
        axs[3].set_ylabel('E [MeV]', fontsize = fs)
        axs[3].set_xlabel(r'z [mm]', fontsize = fs)
        axs[3].tick_params(axis = 'both', labelsize = fs)
        if show_info:
            axs[3].text(.01, .5, casename + '\n' +
                        'FWHM$_E$ = {:.2f} MeV\n'.format(fwhmE) +
                        '$\sigma_z$ = {:.2f} $\mu$m'.format(rmsz*1e3),
                        fontsize = fs, color = 'orange', transform = axs[3].transAxes)
    elif sample:

        # Now get a density function by interpolating the gridpoints
        print('\n Interpolating...')
        dist = np.zeros([Npart, 2])
        x = np.linspace(0, 1, MLPS.shape[1])
        y = np.linspace(0, 1, MLPS.shape[0])
        maximum = np.max(MLPS)
        density = interpolate.interp2d(x, y, MLPS)

        # Now generate Npart particles using the density function
        print('generating particles...')
        realNpart = 0
        for j in range(Npart):
            if j%50000 == 0:
                print(j, 'particles have been generated')
            N_attempts = 0
            while True:
                x = np.random.rand()
                y = np.random.rand()
                N_attempts += 1
                if np.random.rand() * maximum < density(x,y):
                    dist[realNpart,0] = x
                    dist[realNpart,1] = 1 - y
                    realNpart += 1
                    break
                elif N_attempts > 200:
                    break
        dist = dist[:realNpart]
        print("Finally we have", realNpart, 'particles')
                    
        # Rescale obtained distribution
        dist[:,0] *= extentLPS[1] - extentLPS[0]
        dist[:,0] -= np.mean(dist[:,0])
        dist[:,0] *= 1e-3  # mm to metres
        dist[:,1] *= extentLPS[3] - extentLPS[2]
        dist[:,1] += extentLPS[2]

        # Save distribution for later use
        dist[:,1] *= 1 / .511  # MeV to unitless momentum (necessary for OPAL)
        df = pd.DataFrame(dist, columns = ['z', 'pz'])
        df.to_csv(out_fn, sep = '\t', index = False, mode = 'w')
        dist[:,1] *= 511e3  # back to eV for the plot

        # Plot newly generated distro
        [hist, bins] = np.histogram(dist[:,0], bins = 50)
        [FWHM,_] = getFWHM(bins[1:], hist, denom = 2)
        FWHM *= 1e6
        rms = getRms(bins[1:], hist) * 1e6
        [hist, bins] = np.histogram(dist[:,1], bins = 50)
        [FWHME,_] = getFWHM(bins[1:], hist, denom = 2)
        FWHME *= 1e-6
        rmsE = getRms(bins[1:], hist) * 1e-6
        plotScreenXY(axs[3], dist[:,0], dist[:,1], ['z', 'E'], type = 'hist2d-hist', factors = [1e3, 1e-6], nbins = 200, color = 3, maxHH = maxHH, enable_cbar = 0)
        # Text with info
        axs[3].text(.01, .45, 'Generated {:.2e} particle distribution\n'.format(realNpart) +
                   'E = {:.1f} MeV, Q = 300 pC\n'.format(E) + 
                   'FWHM$_z$ = {:.2f} $\mu$m,\n$\sigma_z$ = {:.2f} $\mu$m\n'.format(FWHM, rms) +
                   'FWHM$_E$ = {:.2f} MeV,\n$\sigma_E$ = {:.2f} MeV\n'.format(FWHME, rmsE),
                   fontsize = 15, color = 'black', transform = axs[3].transAxes)
        axs[3].set_xlim(extentLPS[:2] - np.mean(extentLPS[:2]))
        axs[3].set_ylim(extentLPS[2:])

# # Adjust axes to have same scale
# pmf.adjust_axes_limits([ax[0], ax[2], ax[3]], 'x')
# pmf.adjust_axes_limits([ax[0], ax[1], ax[3]], 'y')


# # Add axis labels when necessary
# ax[0].set_ylabel('E [MeV]', fontsize = fs)
# ax[2].set_xlabel(r'z [$\mu$m]', fontsize = fs)
# ax[1].tick_params(axis = 'y', labelsize = fs)
# ax[1].set_xticks([])
# ax[2].tick_params(axis = 'x', labelsize = fs)
# ax[2].set_yticks([])

# # Custom legend
# custom_lines = [Line2D([0], [0], color = 'orange', lw = 4, ls = lsspe),
#                 Line2D([0], [0], color = 'orange', lw = 4, ls = lstdc),
#                 Line2D([0], [0], color = 'red', lw = 4),]
# fig.legend(custom_lines, ['Spectrometer only', 'TDC only', '1D projection current plot', 'Histogram before removing bump'], fontsize = fs)

# # Save the image
# plt.savefig((pltpath + '/YAG_to_OPAL_distro_case3_noBump.png'),bbox_inches='tight')
# plt.show()


def centre_matrix(M):
    cent = np.array([M.shape[0]//2, M.shape[1]//2])
    cm = ndimage.measurements.center_of_mass(M)
    cm = np.asarray(cm).astype(int)
    return ndimage.interpolation.shift(M, cent-cm, cval = 0.0)


def zoom_M(M, z, axis):
    # Zoom and maintain the same number of pixels
    if axis == 0:
        M = M.T
    old = M.shape    
    L0 = old[0]
    M = ndimage.zoom(M, [z,1.0])
    Lf = M.shape[0]
    diff1 = (Lf-L0)//2
    diff2 = Lf - L0 - diff1
    if diff1 > 0:
        M = M[diff1:-diff2,:]
    elif diff1 < 0:
        extra1 = np.zeros((-diff1,M.shape[1]))
        extra2 = np.zeros((-diff2,M.shape[1]))
        M = np.concatenate((extra1,M,extra2), axis = 0)

    if axis == 0:
        M = M.T
             
    return M