import os, sys
from utils import update_errorbar

	## -----------------------------------------
	## Check for necessary libraries
	## -----------------------------------------

try:
	import numpy as np
except():
	print('\nPlease install NumPy. Exiting.')
	print('pip install numpy\n')
	sys.exit()

try:
	import matplotlib as mpl
	import matplotlib.gridspec as gridspec
	import matplotlib.pyplot as plt
	from matplotlib.widgets import SpanSelector, Slider, Button, CheckButtons
	from matplotlib.collections import LineCollection
except():
	print('\nPlease install Matplotlib.')
	print('pip install matplotlib\n')
	sys.exit()

try:
	from stingray.events import EventList
	from stingray.gti import create_gti_from_condition
except():
	print('\nPlease install Stingray.')
	print('pip install stingray\n')
	sys.exit()

try:
	import seaborn as sns
	sns.set_context('paper')
#	sn.set_theme(context='paper', style='darkgrid', palette='deep', font='sans', font_scale=1)
except():
	print('\nSeaborn not installed. Using default matplotlib styles.\n')
	plt.style.use('seaborn-darkgrid')


	## --------------------------------------------------------------------------
	## Functions
	## --------------------------------------------------------------------------

def update_plot(pararameters): 

	## Read parameter list
	## --------------------------

	tbin = parameters[0]
	ecut = parameters[1]
	tmin = parameters[2]
	tmax = parameters[3]

	## Create hard/soft events and lightcurves
	## --------------------------------------------

	soft_events = events.filter_energy_range([emin,ecut])
	hard_events = events.filter_energy_range([ecut,emax])

	hard_lc = hard_events.to_lc(tbin)
	soft_lc = soft_events.to_lc(tbin)

	local_time = 0.5*(hard_lc.time+soft_lc.time)/1000
	local_time -= static_tmin

	## Filters for time and high quality hardness
	## -------------------------------------------

	mask1 = (local_time>tmin) & (local_time<tmax)
	mask2 = (hard_lc.counts>1) & (soft_lc.counts>1)
	mask = mask1 & mask2
	indx = np.where(mask)[0]

	time = local_time[indx]-min(local_time[indx])  
	intensity = hard_lc.countrate[indx] + soft_lc.countrate[indx]
	intensity_err = np.sqrt( hard_lc.counts_err[indx]**2 + soft_lc.counts_err[indx]**2 ) / tbin

	hardness = hard_lc.counts[indx] / soft_lc.counts[indx]

	#    hardness = (hard_lc.counts[indx] - soft_lc.counts[indx]) / intensity
	#    hardness_err = intensity_err * np.sqrt( (1/(hard_lc.counts[indx] - soft_lc.counts[indx]))**2 
	#                                            + (1/intensity)**2 )

	sigma = np.sqrt( (soft_lc.counts_err[indx] / soft_lc.counts[indx])**2 + \
		              (hard_lc.counts_err[indx] / hard_lc.counts[indx])**2   )
	hardness_err = hardness * sigma

	#    intensity /= tbin
	#    intensity_err /= tbin


	## Create colormap from time array
	## ---------------------------------

	norm = mpl.colors.Normalize(vmin=min(time), vmax=max(time))
	colors = norm(time)

	## Plot lines
	## -------------------

	line1.set_data(time,intensity)
	line2.set_data(hardness,intensity)

	update_errorbar(err1, time, intensity, None, intensity_err)
	update_errorbar(err2, hardness, intensity, hardness_err, intensity_err)


	#    lines = [np.column_stack([xi, yi]) for xi, yi in zip(hardness,intensity)]
	#    # Now we update the lines in our LineCollection
	#    line2.set_segments(lines)
	#    line2.set_array(colors)

	## Plot colored scatter data points
	## ---------------------------------

	offsets = np.c_[time,intensity]
	scat1.set_offsets(offsets)
	scat1.set_array(colors)

	offsets = np.c_[hardness,intensity]
	scat2.set_offsets(offsets)
	scat2.set_array(colors)

	## Set axis limits
	## ----------------

	ax1.set_xlim(-0.1, 1.05*time.max())


	ax1.set_ylim(0.9*(intensity.min()-intensity_err[np.argmin(intensity)]), 
	1.025*(intensity.max()+intensity_err[np.argmax(intensity)]))

	ax2.set_xlim(0.95*hardness.min(), 1.05*hardness.max())

	## Set axis tick labels
	## ---------------------

	ticks = np.linspace(hardness.min(), hardness.max(), 5, endpoint=True)
	ax2.set_xticks(ticks)
	ax2.set_xticklabels([f'${x:.2f}$' for x in ticks])

	ticks = np.linspace(0, time.max(), 5, endpoint=True)
	ax1.set_xticks(ticks)
	ax1.set_xticklabels([f'${x:.1f}$' for x in ticks])

	## Update title with lightcurve basic stats
	## -------------------------------------------

#	wavg = np.average(intensity, weights=intensity_err)
#	wstd = np.sqrt(np.average((intensity-wavg)**2, weights=intensity_err))

#	ax1_title.set_text(f'Min:${min(intensity)/tbin:.1f}$ c/s  Max:${max(intensity)/tbin:.1f}$ c/s \
#	Avg:${wavg/tbin:.1f}\\pm{wstd/tbin:.1f}$ c/s')

	## Update time axis with reference time
	## --------------------------------------

#	ax1_xlabel.set_text(f'Time (ks since {min(local_time[indx]):.1f} ks)')

	fig.canvas.draw_idle()

	## ---------------------------------------------------------------------------


def read_span_selector(xmin,xmax):

	## ---------------------------------------------
	## Función asociada al widget "SPAN SELECTOR"
	## ---------------------------------------------
	## Dados xmin,xmax leidos de la seleccion de tiempo
	## Busca los indices kmin,kmax de los elementos más cercanos
	## Actualiza los valores de tiempo en la lista de parámetros
	## Lee y actualiza los parámetros de los sliders
	## Actualiza los labels de los sliders
	## Ejecuta update_plot
	## ----------------------------------------------

	kmin, kmax = np.searchsorted(static_time, (xmin, xmax))
	kmax = min(len(static_time) - 1, kmax)

	parameters[2] = static_time[kmin]
	parameters[3] = static_time[kmax]

	parameters[0] = timebins[sld_tbin.val]
	parameters[1] = energies[sld_ecut.val]

#	tbin_lab.set_text(f'Bin time : ${parameters[0]:.1f}$ s')
#	ecut_lab.set_text(f'Cutoff energy : ${parameters[1]:.1f}$ keV')

	update_plot(parameters)

	## ---------------------------------------------------------------------------


def read_sliders(val):

	## -----------------------------------------------
	## Función asociada a los widgets "SLIDERS"
	## -----------------------------------------------
	## 1. Lee las posiciones actuales de los sliders
	## 2. Actualiza los valores en la lista de parametros
	## 3. Actualiza las etiquetas de los sliders
	## 4. Llama a la función "update_plot"
	## -----------------------------------------------

	parameters[0] = timebins[sld_tbin.val]
	parameters[1] = energies[sld_ecut.val]

#	tbin_lab.set_text(f'Bin time : ${parameters[0]:.1f}$ s')
#	ecut_lab.set_text(f'Cutoff energy : ${parameters[1]:.1f}$ keV')

	ax1_xlabel.set_text(f'Time (ks; ${parameters[0]:.1f}$ s bin)')
	ax2_xlabel.set_text(f'Hardness (${parameters[1]:.1f}$ keV cutoff)')

	update_plot(parameters) 


	## ---------------------------------------------------------------------------


#def clear_axes(event):

#	## -----------------------------------------------
#	## Función asociada al widget "BUTTON"
#	## -----------------------------------------------
#	## Remueve los "scatter" y "line" plots
#	## Remueve el "span selector"
#	## Remueve etiquetas dinámicas
#	## Reposiciona los "sliders" a valores por defecto
#	## ------------------------------------------------

#	timespan.set_visible(False)
#	sld_tbin.reset()
#	sld_ecut.reset()

#	parameters[0] = timebins[sld_tbin.val]
#	parameters[1] = energies[sld_ecut.val]

#	tbin_lab.set_text(f'Bin time : ${parameters[0]:.1f}$ s')
#	ecut_lab.set_text(f'Cutoff energy : ${parameters[1]:.1f}$ keV')

#	ax1_title.set_text('')
#	ax1_xlabel.set_text('Time (ks)')


#	line1.set_data([],[])
#	line2.set_data([],[])

#	update_errorbar(err1, np.empty(1),np.empty(1),None,np.empty(1))
#	update_errorbar(err2, np.empty(1),np.empty(1),np.empty(1),np.empty(1))

#	offsets = np.c_[[],[]]

#	scat1.set_offsets(offsets)
#	scat1.set_array([])

#	scat2.set_offsets(offsets)
#	scat2.set_array([])

	## ---------------------------------------------------------------------------




if __name__ == "__main__":

	## ===========================================================================
	## --------------- Bloque Principal ------------------------------------------
	## ===========================================================================
	##  1. Lectura de datos.
	##  2. Definición de GTI maestro y de parámetros de misiones.
	##  3. Diseño de los paneles a graficar, títulos y etiquetas
	##  4. Creación de la curva de luz completa (estática) del panel superior
	##  5. Se definen los "snap_times" y el "tmin" de referencia
	##  6. Se definen los contenedores para los "scatter" y "line" plots
	##  7. Se definen y configuran todos los widgets a utilizar 
	##      (nombres, ejes, etiquetas, funciones, etc)
	## ===========================================================================


	## Read event data
	## -------------------

	input_data = sys.argv[1:]

	if len(input_data)==1:
	   
		events = EventList.read(sys.argv[1], 'hea')

	elif len(input_data)==2:
	   
		evA = EventList.read(sys.argv[1], 'hea')
		evB = EventList.read(sys.argv[2], 'hea')
	  
		if evA.mission != evB.mission : 
		    print('\nEvent lists from different missions!')
		    print('Cannot join. Exit.\n')
		    sys.exit()
	  
		events = evA.join(evB)

	else:
		print('\nPlease enter up to 2 HEASoft event list from the same mission. Exit.\n')
		sys.exit()
		

	## Create lightcurve to generate master GTI
	## to avoid null gaps from Earth occultation
	## Only applicable to low orbit satellites (NICER & NUSTAR)
	## ----------------------------------------------------------
	lc_raw = events.to_lc(100)
	gti0 = create_gti_from_condition(lc_raw.time, lc_raw.counts > 1, safe_interval=1)
	events.gti = gti0


	## Mission dependent parameters
	## --------------------------------

	mission = events.mission                        # MISSION name
	instrument = events.instr                       # INSTRUMENT name
	tstart = events.mjdref + min(lc_raw.time)/86400 # REFERENCE TIME

	nbins = 20  # Number of bins for array of time bins and energy bins

	if mission.lower()=='nustar':
		emin = 3
		emax = 79
		bintime = 30            											# binning for static lightcurve
		energies = np.geomspace(5, 30, nbins, endpoint=True)    	# energy array
		timebins = np.geomspace(10,150, nbins, endpoint=True) 	# time bins array


	elif mission.lower()=='nicer':
		emin = 0.4
		emax = 12
		bintime = 10
		energies = np.geomspace(1,10, nbins, endpoint=True)
		timebins = np.geomspace(1,50, nbins, endpoint=True) # time bins array

	elif mission.lower()=='xmm':
		emin = 0.4
		emax = 12
		bintime = 30
		energies = np.geomspace(1, 10,  nbins, endpoint=True)
		timebins = np.geomspace(10,150,  nbins, endpoint=True) # time bins array

	else:
		print('Please enter XMM, NICER or NUSTAR event list. Exit.')
		sys.exit()
		
	## -------------------------------------
	## Plot design
	## -------------------------------------

	## Colormap for future time axis on HID
	## -------------------------------------
	cm = plt.cm.get_cmap('viridis')
	norm = mpl.colors.Normalize(vmin=0, vmax=1)


	## Figure creation and design
	## -----------------------------
	fig = plt.figure(figsize=plt.figaspect(0.6), dpi=170, constrained_layout=True)
	gs = gridspec.GridSpec(2,2, wspace=0.02, hspace=0.02, height_ratios=[0.5,1], width_ratios=[1,1]) 

	ax0 = fig.add_subplot(gs[0, :])             ## Top axis
	ax1 = fig.add_subplot(gs[1, 0])             ## Middle left
	ax2 = fig.add_subplot(gs[1, 1], sharey=ax1) ## Middle right
	## Bottom left,right: sliders. 

#	for ax in [ax0,ax1,ax2]:
#		#ax.patch.set_color('palegoldenrod') # or whatever color you like
#		ax.patch.set_alpha(.5)


	tick_kwargs = dict(which='both', size=3, direction='in', top=True, right=True, labelsize='small')

	ax0.tick_params(**tick_kwargs, labelbottom=False, labeltop=False)
	ax1.tick_params(**tick_kwargs, labelleft=True, labelright=False)
	ax2.tick_params(**tick_kwargs, labelleft=False, labelright=False)

	ax0_top = ax0.secondary_xaxis('top')
	ax0_top.tick_params(pad=1, size=0, which='both', labelsize='small')
	ax0_top.set_xlabel(f'Time (ks since ${tstart:.4f}$ MJD; ${bintime:g}$s bin)', fontsize='small')

	ax0.annotate(f'{mission.upper():10s}\n${emin:g}-{emax:g}$ keV', xy=(1.03,0.5), 
	xycoords='axes fraction', rotation=-90, fontsize='small', va='center', ha='center')

	ax0.set_ylabel('Count rate (cts/sec)', fontsize='small')
	ax1.set_ylabel('Count rate (cts/sec)', fontsize='small')

	ax1_xlabel = ax1.set_xlabel('Time (ks)', fontsize='small')
	ax2_xlabel = ax2.set_xlabel('Hardness', fontsize='small')

	ax2.set_xscale('log')
	ax2.minorticks_off()
	ax2.axvline(1, lw=1.25, c='k', zorder=1)  


	## -------------------------------------------------------------------------------------
	## LIGHTCURVE
	## -------------------------------------------------------------------------------------
	## 1. Create general light curve
	## 2. Create GTIs from null gaps 
	## 3. Split lightcurve in consecutives segments
	## 4. Save min,max times of GTIs to create "snap_times" for "SPAN SELECTOR"
	##      If min,max equals TSTART,TSTOP then no "snap_times" is passed to "SPAN SELECTOR"
	##      This is useful for low orbit missions.
	## ------------------------------------------------------------------------------------    

	total_events = events.filter_energy_range([emin,emax])

	total_lc = total_events.to_lc(bintime)

	gti = create_gti_from_condition(total_lc.time, total_lc.counts>1, safe_interval=1)
	total_lc.gti = gti
	total_lc.apply_gtis()

	static_time  = total_lc.time/1000
	static_tmin  = min(static_time)
	static_time -= static_tmin

	split_lc = total_lc.split(min_gap=bintime, min_points=3)
	tmin = min(split_lc[0].time)/1000

	times = []

	for k in range(len(split_lc)):

#		if len(split_lc[k].time) < 3 : continue
		
		ax0.step(split_lc[k].time/1000-tmin, split_lc[k].countrate, where='mid', lw=1, c='k', zorder=2)
		
		times.append([split_lc[k].time[0], split_lc[k].time[-1]])


	snap_times = np.unique(np.c_[np.asarray(times)/1000-tmin])
	if len(snap_times) < 3: snap_times = None
#	snap_times = None


	## ------------------------------------------------
	## Initiate parameter list 
	## Initiate lines, errorbar & scatter containers
	## Inititate title container for zoom statistics
	## ------------------------------------------------

	parameters = np.zeros(4)

	line1, = ax1.step([],[], c='k', where='mid', lw=1.25, zorder=2, alpha=0.2)
	line2, = ax2.plot([],[], c='k', lw=1.25, zorder=2, alpha=0.2)

	scat1 = ax1.scatter([],[], c=[], cmap=cm, marker='o', s=10, zorder=3)
	scat2 = ax2.scatter([],[], c=[], cmap=cm, marker='o', s=10, zorder=3)

	err_kwargs = dict(visible=True, c='k', lw=1.25, ls='', zorder=2, alpha=0.15)
	err1 = ax1.errorbar([],[],xerr=None,yerr=[], **err_kwargs)
	err2 = ax2.errorbar([],[],xerr=[],  yerr=[], **err_kwargs)

#	ax1_title = ax1.set_title('', fontsize='small', loc='center', pad=1)

	#    lines = []
	#    line2 = LineCollection(lines, cmap="viridis", lw=1.25, alpha=0.4)
	#    ax2.add_collection(line2)

	## ============================
	## Widgets
	## ============================

	## Time Span Selector  
	## ----------------------------    
	timespan = SpanSelector(ax0, read_span_selector, direction="horizontal",
		    useblit=True, props=dict(alpha=0.05, facecolor="forestgreen"), 
		    handle_props=dict(lw=1, ls='-'), interactive=True, 
		    drag_from_anywhere=True, ignore_event_outside=True,
		    grab_range=10, snap_values=snap_times )


	## Energy Cutoff Slider 
	## ----------------------------    

#	ax_ecut = fig.add_subplot(gs[1, 3])     ## Bottom right axis
#	ax_ecut = fig.add_axes([0.917, 0.13, 0.015, 0.45])
	ax_ecut = fig.add_axes([0.58, 0.01, 0.26, 0.015])

	sld_ecut = Slider( ax_ecut, "", valmin=0, valmax=nbins-1, valinit=nbins//2, 
	valstep=1, valfmt='%.1f', track_color='silver', color="k", initcolor=None, 
	orientation='horizontal', handle_style={'size':5, '':'D', 'edgecolor':'k'})
		
	sld_ecut.on_changed(read_sliders)

	sld_ecut.valtext.set_visible(False) # Deactive slider labels to control from annotation

#	ecut_lab = ax_ecut.annotate(f'Cutoff energy : ${energies[nbins//2]:.1f}$ keV', 
#	xy=(1,0.5), xycoords='axes fraction', rotation=-90, fontsize='small', va='center')


	## Time Bin Slider 
	## ----------------------------    

#	ax_tbin = fig.add_subplot(gs[1, 0])     ## Bottom left axis
#	ax_tbin = fig.add_axes([0.03, 0.13, 0.015, 0.45])
	ax_tbin = fig.add_axes([0.18, 0.01, 0.26, 0.015])


	sld_tbin = Slider( ax_tbin, "", valmin=0, valmax=nbins-1, valinit=nbins//2,
	valstep=1, valfmt='%.1f', track_color='silver', color="k", initcolor=None, 
	orientation='horizontal', handle_style={'size':5, '':'D', 'edgecolor':'k'})

	sld_tbin.on_changed(read_sliders)

	sld_tbin.valtext.set_visible(False) # Deactive slider labels to control from annotation

#	tbin_lab = ax_tbin.annotate(f'Bin time : ${timebins[nbins//2]:.1f}$ s',
#	 xy=(-1,0.5), xycoords='axes fraction', rotation=90, fontsize='small', va='center')


	## Reset button
	## ------------- 

#	ax_reset = fig.add_axes([0.49, 0.01, 0.05, 0.05])

#	clear_button = Button(ax_reset, 'Clear', color='silver', hovercolor='skyblue')

#	#    for r in clear_button.rectangles:
#	#        r.set_facecolor("silver") 
#	#        r.set_edgecolor("k")
#	#        r.set_alpha(0.1) 

#	clear_button.label.set_fontsize('small')

#	clear_button.on_clicked(clear_axes)

	## Check Buttons
	## https://stackoverflow.com/questions/42421363/customize-check-buttons-in-matplotlib
	## -----------------

#	def update_check(label):
#		for x in [err1,err2]:
#		    line,cap,bars = x
#		    [bar.set_visible(not bar.get_visible()) for bar in bars]
#		fig.canvas.draw_idle()


#	ax_check_errors = fig.add_axes([0.85, 0.01, 0.1, 0.05])

#	check_error = CheckButtons(ax_check_errors, ['Show errors'], [False])

#	check_error.on_clicked(update_check)

#	for r in check_error.rectangles:
#		r.set_facecolor("silver") 
#		r.set_edgecolor("k")
#		r.set_alpha(0.1) 
#		r.set_hatch('o') 
#		r.set_capstyle('round')

#	check_error.labels[0].set_rotation(0)
#	check_error.labels[0].set_fontsize('x-small')

#	[ll.set_color("k") for l in check_error.lines for ll in l]
#	[ll.set_linewidth(1) for l in check_error.lines for ll in l]


	## ---------------------------

	plt.show()


	## ====================================================================
	## END OF PROGRAM
	## ====================================================================

