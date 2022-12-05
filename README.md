# Interactive-HID

This tool makes light curve visualization more simple. 
It takes a cleaned event list as input, from three common HEASARC missions: NICER, NUSTAR and XMM-Newton.

By selecting any time interval from the full spectrum lightcurve (top panel) it will display that time segment on a secondary axis, 
where the time resolution of that segment can be increased or decreaced at will with the help of a slider ('time bin slider').
The full spectrum Emin-Emax is mission dependent.

On the bttom right panel it will display the hardness-intensity diagram (HID), with the hardness defined as the ratio of hard (Ecut-Emax) to soft (Emin-Emax)
lightcurves, using the time bin defined on the 'time bin slider'. This HID also can be modified by moving the 'Cutoff Energy slider' to see how the spectral shape changes.

Widgets events key (ex, 'Esc' to reset) where left to their default behaviour according to matplotlib docs.

# Prerequisites
- Python 3.6 or higher
- Stingray : https://docs.stingray.science/install.html
- Numpy
