import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from decimal import Decimal
import os
from datetime import date

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
            ax2.plot(xPoints[:-1], hist / np.max(hist) * maxHH, lw = 3, color = 'C' + str(color))
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
            ax3.plot(1 - hist / np.max(hist) * maxHH, xPoints, lw = 3, color = 'C' + str(color))
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
#        bin_means -= bin_means[0]
        bin_means -= np.array(bin_means).mean()
        labs[1] = '$\Delta$' + labs[1]
        ax.plot( bin_edges[1:] - .5*bin_size, bin_means, zorder = 2, lw = lw, ls = ls)

    ax.tick_params( axis = 'both', labelsize = fs )
    ax.ticklabel_format( axis = 'both', style = 'sci', scilimits = (-1, 3), useOffset = False )
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
    
