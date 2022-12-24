# Interactive Lightcurve Explorer

![Preview](https://github.com/fedefoga/Interactive-Lightcurve-Explorer/fig2.png)

This tool makes light curve visualization more simple. 
It takes a cleaned event list as input, from three common NASA/ESA missions: *NICER*, *NUSTAR* and *XMM-Newton*.

By selecting any time interval from the full spectrum lightcurve (top panel) it will display that time segment on the bottom left axis, 
where the time resolution of that selected interval can be increased or decreaced with the help of a slider ('time bin slider').
The full spectrum *Emin/Emax* is mission dependent.

On the bottom right panel it will display the hardness-intensity diagram (HID), with the hardness defined as the ratio of hard (*Ecut/Emax*) to soft (*Emin/Ecut*)
lightcurves, using the time bin defined on the 'time bin slider'. This HID also can be modified by moving the 'Cutoff Energy slider' to see how the spectral shape changes.

# Prerequisites

- Python 3.6+
- Matplotlib 
- Numpy
- [Stingray](https://docs.stingray.science/install.html) 


# Usage
`python3 hid.py evt1 [evt2]`

where **evt1** or **evt2** are cleaned event lists from the mentioned xray missions.
When two eventlists **from the same mission** are given, they will be joined.

