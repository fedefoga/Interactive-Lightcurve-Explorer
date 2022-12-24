import numpy as np
    
## =============================================
## Mission dependent channel/energy conversion
## =============================================

def kev2chan(kev, mission):

    if mission.lower()=='nustar':
        return int(25*(kev-1.6))

    elif mission.lower()=='nicer':
        return int(100*kev)

    elif mission.lower()=='xmm':
        return int(1000*kev)

    else:
        print(f'mission not recognized. Returning {kev} keV.\n')
        return kev
    
    
def chan2kev(chan, mission):

    if mission.lower()=='nustar':
        return chan/25+1.6

    elif mission.lower()=='nicer':
        return chan/100

    elif mission.lower()=='xmm':
        return chan/1000

    else:
        print(f'Mission not recognized. Returning {chan} channel.\n')
        return chan


## ============================================
## Find index of array to given nearest value
## ============================================

def find_nearest(array, value):
    return np.argmin(np.abs(array-value))
    

## -----------------------------------------------------------------------------------
## https://stackoverflow.com/questions/25210723/matplotlib-set-data-for-errorbar-plot
## -----------------------------------------------------------------------------------
    
def update_errorbar(errobj, x, y, xerr=None, yerr=None):
    
    ln, caps, bars = errobj

    if len(bars) == 2:
        assert xerr is not None and yerr is not None, \
        "Your errorbar object has 2 dimension of error bars defined. \
        You must provide xerr and yerr."
        barsx, barsy = bars  # bars always exist (?)
        try:  # caps are optional
            errx_top, errx_bot, erry_top, erry_bot = caps
        except ValueError:  # in case there is no caps
            pass

    elif len(bars) == 1:
        assert (xerr is     None and yerr is not None) or\
               (xerr is not None and yerr is     None),  \
               "Your errorbar object has 1 dimension of error bars defined. \
               You must provide xerr or yerr."

        if xerr is not None:
            barsx, = bars  # bars always exist (?)
            try:
                errx_top, errx_bot = caps
            except ValueError:  # in case there is no caps
                pass
        else:
            barsy, = bars  # bars always exist (?)
            try:
                erry_top, erry_bot = caps
            except ValueError:  # in case there is no caps
                pass

    ln.set_data(x,y)

    try:
        errx_top.set_xdata(x + xerr)
        errx_bot.set_xdata(x - xerr)
        errx_top.set_ydata(y)
        errx_bot.set_ydata(y)
    except NameError:
        pass
    
    try:
        barsx.set_segments([np.array([[xt, y], [xb, y]]) for xt, xb, y in zip(x + xerr, x - xerr, y)])
    except NameError:
        pass

    try:
        erry_top.set_xdata(x)
        erry_bot.set_xdata(x)
        erry_top.set_ydata(y + yerr)
        erry_bot.set_ydata(y - yerr)
    except NameError:
        pass
    
    try:
        barsy.set_segments([np.array([[x, yt], [x, yb]]) for x, yt, yb in zip(x, y + yerr, y - yerr)])
    except NameError:
        pass

## -----------------------------------------------------------------------------
## -----------------------------------------------------------------------------
    
