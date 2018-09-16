""" The new plotting library works like this: If you are comparing
two different datasets, you'd use the intraproject plotting. If you are
just looking at one project, you'd do interproject plotting. In either
case you're going to create a figure at the outset and pass the axes
along with pandas dataframes to a plotting method. No data is contained
in the CFG files that is used in this plotting library with the
exception of project name. Future revisions of this file will be split
into PlotTimeseries.py, PlotHistogram.py, and Plot2DHistogram.py. """

# Contains the approach all plots will be generated with.
# http://matplotlib.org/examples/pylab_examples/subplots_demo.html

import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
import numpy as np
import pandas as pd
from itertools import product
import operator
from matplotlib.patches import Rectangle, Circle
from matplotlib.collections import PatchCollection

nullfmt   = mpl.ticker.NullFormatter()

def get_cmap(N):
    '''Returns a function that maps each index in 0, 1, ... N-1 to a distinct
    RGB color.'''
    color_norm  = mpl.colors.Normalize(vmin=0, vmax=N-1)
    scalar_map = mpl.cm.ScalarMappable(norm=color_norm, cmap='Set1')
    def map_index_to_rgb_color(index):
        return scalar_map.to_rgba(index)
    return map_index_to_rgb_color

def proxy_fill_between(x, y1, y2=0, ax=None, flipxy=False, **kwargs):
    """Plot filled region between `y1` and `y2`.

    This function works exactly the same as matplotlib's fill_between, except
    that it also plots a proxy artist (specifically, a rectangle of 0 size)
    so that it can be added it appears on a legend.
    """
    ax = ax if ax is not None else plt.gca()
    if flipxy:
        ax.fill_betweenx(x, y1, y2, **kwargs)
    else:
        ax.fill_between(x, y1, y2, **kwargs)
    p = Rectangle((0, 0), 0, 0, **kwargs)
    ax.add_patch(p)
    return p

def plot_ord_ions(seqstr, grect, ion_col, sites):
    """ Does the heavy lifting of plotting ions on ordering boxes
    """
    ions = []
    # TODO, do not hardcode?
    ec,s,e,el,lt,cc = seqstr
    for site_num, site in enumerate([ec+s, e, el, lt, cc]):
        if len(site) == 1:
            ions.append(Circle(grect + sites[site_num],
                                        0.02, fc=ion_col[site[0]], lw=0))

        elif len(site) == 2:
            ions.append(Circle(grect + sites[site_num] + [0.015,0.015],
                               0.02, fc=ion_col[site[0]], lw=0))
            ions.append(Circle(grect + sites[site_num] - [0.015,0.015],
                               0.02, fc=ion_col[site[1]], lw=0))

        elif len(site) > 2:
            ions.append(Circle(grect + sites[site_num] + [-0.015,-0.015],
                               0.02, fc=ion_col[site[0]], lw=0))
            ions.append(Circle(grect + sites[site_num] + [0.015,-0.015],
                               0.02, fc=ion_col[site[1]], lw=0))
            ions.append(Circle(grect + sites[site_num] + [0,+0.015],
                               0.02, fc=ion_col[site[2]], lw=0))

    return ions

def plot_orderings(ax, popdata, rows=2, ords_per_row=8,
                   ion_col={"N":"blue","K":"green"},
                   sites={0:[0.045, 0.30], 1:[0.045, 0.20],
                          2:[0.045, 0.10], 3:[0.045, 0.0],
                          4:[0.045, -0.09]}):
    """
    Occupancy orderings, even for mixed ions, as generated in popdata dict.

    Parameters
    ----------
    ax : matplotlib.axes.AxesSubplot
        Plotting is performed on this axis.
    popdata: dict
        Dictionary of counts for all orderings
    rows : int
        Number of rows of orderings to render
    ords_per_row : int
        Number of orderings per row to render
    ion_col : dict
        For all one-letter atom codes, map to a color
    sites : dict
        For all sites, map to a position in a grid.
    """

    total_ords = rows*ords_per_row
    total_frames = np.sum(popdata.values())
    top_ords = sorted(popdata.items(), key=operator.itemgetter(1), reverse=True)[0:total_ords]

    grid = np.roll(np.roll(np.mgrid[0.2:0.8:complex(0,rows),
                                    0.0:1.0:complex(0,ords_per_row)].reshape(2, -1).T, 1, axis=1),
                   total_ords)

    patches = []
    for grect,popdat in zip(grid, top_ords):
        patches.append(Circle(grect + [0.045, -0.085], 0.06, ec="black", fc="white", ls="--", lw=6))
        patches.append(Rectangle(grect, 0.09, 0.2, ec="black", fc="white", lw=6))
        patches.append(Rectangle(grect - [0, 0.05], 0.09, 0.095, ec="black", fc="white", lw=6))
        patches.append(Rectangle(grect + [0, 0.15], 0.09, 0.095, ec="black", fc="white", lw=6))
        patches.extend(plot_ord_ions(popdat[0], grect, ion_col, sites))
        pop_percent = str(100*np.round(float(popdat[1])/total_frames,3))+"%"
        print(pop_percent, total_frames, 100*np.round(float(popdat[1])/total_frames,3))
        ax.text(grect[0] + 0.045, grect[1] - 0.20, pop_percent,
                ha="center", family='sans-serif', size=30)

    collection = PatchCollection(patches, match_original=True, alpha=1.0)
    ax.add_collection(collection)
    return ax

def plot_mode_occbarchart(ax, xvals, traj_mean, traj_sem, ptitle=None,
                          yoffset=None,
                          xlim=None, ylim=[0,1.0], color="black"):
    """
    Occupancy breakdowns are shown for each mode. Very basic bar chart
    is all that is required.

    Parameters
    ----------
    ax : matplotlib.axes.AxesSubplot
        Plotting is performed on this axis.
    x_vals : list
        Floats of the plotting indices
    traj_mean : list
        Floats to plot
    traj_sem : list
        Error bars to plot
    ptitle : str
        Axis title
    yoffset : list
        Floating point y-adjustments for multiplotting.
    """

    #if xlim != None:
    #    ind = np.arange(len(traj_mean))
    #else:
    #    ind = np.arange(0, xlim[1]-xlim[0])
    ind = xvals

    width = 0.5
    yvals = traj_mean.values
    yerrs = traj_sem.values

    if yoffset is None:
        rects2 = ax.bar(ind+width/2, yvals, width, color=color, yerr=yerrs)
    else:
        if yoffset.sum() > 0:
            rects2 = ax.bar(ind+width/2, yvals, width, color="grey", yerr=yerrs,
                            bottom=yoffset.fillna(0))
        else:
            rects2 = ax.bar(ind+width/2, yvals, width, color=color, yerr=yerrs)

    if ptitle != None:
        ax.set_title(ptitle, fontsize=8)

    _plot_labelling(ax, xlim, ylim, ticky=[0.1,0.2], tickx=[5,10], gridx=False)

    return ax

def plot_ts_mean(ax, s, yerr, dt=0.02, xlim=None, ylim=[0,3],
                 x_label=None, y_label=None,
                 skip=10, ts_vline=None, color="black"):
    """
    Uses fill_between to make shaded regions above and below a timeseries
    (+- SEM or STDDEV) on a matplotlib axis. Figure generation is done
    outside of this function. If a secondary hist_ax is specified then
    an additional histogram of the timeseries is performed.

    Parameters
    ----------
    ax : matplotlib.axes.AxesSubplot
        Plotting is performed on this axis.
    s : pandas.core.series.Series
        Any column of a dataframe or series to be plotted on an axis.
    yerr : pandas.core.series.Series
        Error in the series data.
    dt : float
        The conversion of step to time units (typically nanoseconds).
    xlim : list
        Minimum and maximum x-value for the axis range.
    ylim : list
        Minimum and maximum y-value for the axis range.
    x_label : str
        Optional x-label for plotting.
    y_label : str
        Optional y-label for plotting.
    skip : int
        Number of nanoseconds to skip between datapoints.
    ts_vline : int
        x-axis location in nanoseconds for a thick vertical line.
    color : str
        color of the line and shading 

    """

    # Add shaded background to the time series, indicating equilibration
    if ts_vline != None:
        ax.axvspan(0, ts_vline, facecolor='k', alpha=0.25, linewidth=0)

    ax.plot(s.index[::skip]*dt, s.values[::skip], linewidth=1.0,
            color=color)

    # We don't want to draw a zillion rectangles, so, arbitarily
    # reduce data using a heuristic...
    ax.fill_between(s.index[::int(0.5*skip/dt)]*dt,
                    (s.values-yerr.values)[::int(0.5*skip/dt)],
                    (s.values+yerr.values)[::int(0.5*skip/dt)],
                    alpha=0.5, linewidth=0, lw=0, color=color)

    _plot_labelling(ax, xlim, ylim, x_label, y_label)

    return ax

def plot_ts_histogram(ax, s, binwidth=0.1, dt=0.02, xlim=None, ylim=[0,3],
                      x_label=None, y_label=None, start=0, show_mean=True):

    """
    A histogram of the dataseries using specified limits,
    plotted on the matplotlib axis argument.

    Parameters
    ----------
    ax : matplotlib.axes.AxesSubplot
        Plotting is performed on this axis.
    s : pandas.core.series.Series
        Any column of a dataframe or series to be plotted on an axis.
    dt : float
        The conversion of step to time units (typically nanoseconds).
    binwidth : float
        Histogram binwidth.
    xlim : list
        Minimum and maximum x-value for the axis range.
    ylim : list
        Minimum and maximum y-value for the axis range.
    x_label : str
        Optional x-label for plotting.
    y_label : str
        Optional y-label for plotting.
    start : int
        Time in nanoseconds to exclude from analysis.
    show_mean : bool
        Flag that toggles the display of a shaded mean value region

    """

    # Add shaded background to the time series, indicating equilibration
    if show_mean:
        ax.axhline(s.mean(), linewidth=2.0, color='k')
        ax.axhspan(s.mean()-s.std(), s.mean()+s.std(),
                   facecolor='k', alpha=0.25, linewidth=0)

    bins = np.arange(np.floor(s.min()), np.ceil(s.max()) + binwidth, binwidth)

    ax.hist(s.values[int(start/dt):], bins=bins, orientation='horizontal')

    _plot_labelling(ax, xlim, ylim, x_label, y_label, gridx=False)

    return ax

def plot_order_histograms(ax, df, prime=False, bins=300, xlim=[-20,20], ylim=[0,1], x_label=None,
                          y_label=None, norm_factor=1.0, hist_vline=None):

    """
    A timeseries with multi-color ion representations but ranked
    by descent into the channel rather than colored by ResID.
    TODO: Remove Order checking from inside this function to Coordination.

    Parameters
    ----------
    ax : list
        A list of matplotlib.axes.AxesSubplot objects, larger than needed.
    df : pandas.core.series.DataFrame
        A dataframe with column ResidID.
    prime : bool
        Should we plot only states where an inner ion is unbound.
    bins : int
        Number of histogram bins.
    xlim : list
        Minimum and maximum x-value for the axis range.
    ylim : list
        Minimum and maximum y-value for the axis range.
    x_label : str
        Optional x-label for plotting.
    y_label : str
        Optional y-label for plotting.
    norm_factor : float
        Factor to multiply normalized histograms by.
    hist_vline : list
        x-axis locations for thick vertical lines.
    """

    # Add vertical line to histogram, useful for showing dividers
    if hist_vline != None:
        for xval in hist_vline:
            ax.axvline(xval, color='red', alpha=0.25, linewidth=5)

    total_size = sum(df["Order"] == 0)

    if prime:
        gdf_size = df[df["Prime"]==True]
    else:
        gdf_size = df[df["Prime"]==False]

    # This is the old way of computing occupancy number, which doesn't check the blue ion
    #size_prime_gdf = gdf_size.groupby(["TrajNum","Time"]).size().reset_index(name="Occupancy")
    #occ_percent = size_prime_gdf.groupby(["Occupancy"]).size() / total_size
    # This way takes that into account (here, and noted below)

    occ_percent = gdf_size[gdf_size["Order"] == 0].groupby(["OccMacrostate"]).size() / total_size
    print(occ_percent)
    #occ_cut = (occ_percent[occ_percent > 0.04]).to_dict()
    occ_cut = (occ_percent[occ_percent > 0.01]).to_dict()

    ordcolors=["red","green","blue","purple","orange","pink","aqua","maroon"]
    for axind, (occid, occval) in enumerate(occ_cut.iteritems()):

        target_ax = ax[axind]
        target_ax.text(xlim[1]-9.5, ylim[1]-0.065,
                       "Occ. "+str(occid)+": "+str(np.around(100*occval,1))+"%",
                       fontsize=16)

        #for orderedatom, data in gdf_size[gdf_size["Occupancy"]==occid].groupby("Order"):
        for orderedatom, data in gdf_size[gdf_size["OccMacrostate"]==occid].groupby("Order"):
            histogram, edges = np.histogram(data["Z"], bins=bins, range=[-20,20], normed=True)
            proxy_fill_between(edges[1:], 0, histogram,
                               ax=target_ax,
                               linewidth=0.5, facecolor=ordcolors[orderedatom], alpha=0.75)
            #histogram, edges = np.histogram(data["Z"], bins=bins, range=[-20,20], normed=False)
            #proxy_fill_between(edges[1:], 0, histogram*norm_factor,
            #                   ax=target_ax,
            #                   linewidth=0.5, facecolor=ordcolors[orderedatom], alpha=0.75)

        _plot_labelling(target_ax, xlim, ylim, x_label, y_label,
                        tickx=[1.0,5.0], ticky=[0.05,0.1])

    return ax

def plot_order_2d_histograms(ax, df, prime=False, bins=300, 
                             xlim=[-20,20], ylim=[0,1], x_label=None,
                             y_label=None, norm_factor=1.0, hist_vline=None):

    """
    A timeseries with multi-color ion representations but ranked
    by descent into the channel rather than colored by ResID.
    TODO: Remove Order checking from inside this function to Coordination.

    Parameters
    ----------
    ax : list
        A list of matplotlib.axes.AxesSubplot objects, larger than needed.
    df : pandas.core.series.DataFrame
        A dataframe with column ResidID.
    prime : bool
        Should we plot only states where an inner ion is unbound.
    bins : int
        Number of histogram bins.
    xlim : list
        Minimum and maximum x-value for the axis range.
    ylim : list
        Minimum and maximum y-value for the axis range.
    x_label : str
        Optional x-label for plotting.
    y_label : str
        Optional y-label for plotting.
    norm_factor : float
        Factor to multiply normalized histograms by.
    hist_vline : list
        x-axis locations for thick vertical lines.
    """

    # Add vertical line to histogram, useful for showing dividers
    if hist_vline != None:
        for xval in hist_vline:
            ax.axvline(xval, color='red', alpha=0.25, linewidth=5)

    gdf_size_lowestorder = df
    total_size = sum(gdf_size_lowestorder["Order"] == 0)

    if prime:
        gdf_size = gdf_size_lowestorder[gdf_size_lowestorder["Prime"]==True]
    else:
        gdf_size = gdf_size_lowestorder[gdf_size_lowestorder["Prime"]==False]

    size_prime_gdf = gdf_size.groupby(["TrajNum","Time"]).size().reset_index(name="Occupancy")
    occ_percent = size_prime_gdf.groupby(["Occupancy"]).size() / total_size
    print(occ_percent)
    occ_cut = (occ_percent[occ_percent > 0.04]).to_dict()

    ordcolors=["red","green","blue","purple","orange","pink","aqua","maroon"]
    for axind, (occid, occval) in enumerate(occ_cut.iteritems()):

        target_ax = ax[axind]
        target_ax.text(xlim[1]-0.3, ylim[1]-0.05,
                       "Occ. "+str(occid)+": "+str(np.around(100*occval,1))+"%",
                       fontsize=16)

        for orderedatom, data in gdf_size[gdf_size["Occupancy"]==occid].groupby("Order"):
            histogram, edges = np.histogram(data["Z"], bins=bins, range=[-20,20], normed=True)
            proxy_fill_between(edges[1:], 0, histogram,
                               ax=target_ax,
                               linewidth=0.5, facecolor=ordcolors[orderedatom], alpha=0.75)

        _plot_labelling(target_ax, xlim, ylim, x_label, y_label,
                        tickx=[1.0,5.0], ticky=[0.05,0.1])

    return ax

def plot_pmf(ax, df, xlim=[-20,20], ylim=[0,3], color="#000000",
             bin_col="Bin", mean_col="PMF", sem_col="SEM",
             x_label=None, y_label=None, hist_vline=None, flipxy=False):

    """
    A potential of mean force, basically just a line plot with shading
    plotted on the matplotlib axis argument.

    Parameters
    ----------
    ax : matplotlib.axes.AxesSubplot
        Plotting is performed on this axis.
    df : pandas.core.frame.DataFrame
        A coordination dataframe to be processed.
    xlim : list
        min and max x values
    ylim : list
        min and max y values
    x_label : str
        Optional x-label for plotting.
    y_label : str
        Optional y-label for plotting.
    hist_vline : list
        x-axis locations for thick vertical lines.
    flipxy : bool
        reverses x and y if you want to plot differently
    """

    # Add vertical line to histogram, useful for showing dividers
    if hist_vline != None:
        for xval in hist_vline:
            ax.axvline(xval, color='red', alpha=0.25, linewidth=5)

    x = df[bin_col]
    y = df[mean_col]
    e = df[sem_col]
    if flipxy:
        ax.plot(y, x, linewidth=2.0, color=color)
        proxy_fill_between(x, y-e, y+e,
                           ax=ax, flipxy=flipxy,
                           linewidth=0, facecolor=color, alpha=0.30)
        _plot_labelling(ax, ylim, xlim, y_label, x_label,
                        ticky=[1.0,5.0],tickx=[0.5,1.0])
    else:
        ax.plot(x, y, linewidth=2.0, color=color)
        proxy_fill_between(x, y-e, y+e,
                           ax=ax, flipxy=flipxy,
                           linewidth=0, facecolor=color, alpha=0.30)
        _plot_labelling(ax, xlim, ylim, x_label, y_label,
                        tickx=[1.0,5.0],ticky=[0.5,1.0])

    return ax

def plot_coord_histogram(ax, df, bins=300, dt=0.02, xlim=[-20,20], ylim=[0,3],
                         x_label=None, y_label=None,
                         coord_colnames = ["S178","E177","L176","T175"],
                         norm_factor=1.0, hist_vline=None):

    """
    A histogram of the dataseries using specified limits,
    plotted on the matplotlib axis argument.

    Parameters
    ----------
    ax : matplotlib.axes.AxesSubplot
        Plotting is performed on this axis.
    df : pandas.core.frame.DataFrame
        A coordination dataframe to be processed.
    dt : float
        The conversion of step to time units (typically nanoseconds).
    bins : int
        Number of histogram bins.
    xlim : list
        Minimum and maximum x-value for the axis range.
    ylim : list
        Minimum and maximum y-value for the axis range.
    x_label : str
        Optional x-label for plotting.
    y_label : str
        Optional y-label for plotting.
    coord_colnames : list
        A list of strings that indicates what column names to plot.
    norm_factor : float
        Factor to multiply normalized histograms by.
    hist_vline : list
        x-axis locations for thick vertical lines.
    """

    # Add vertical line to histogram, useful for showing dividers
    if hist_vline != None:
        for xval in hist_vline:
            ax.axvline(xval, color='red', alpha=0.25, linewidth=5)

    zero_label = "0"*len(coord_colnames)
    coord_chars = [c[0] for c in coord_colnames]

    # Enumerate all possible 0/1 values for each coordination column.
    all_binding_modes = product("01",repeat=len(coord_colnames))
    all_binding_strs = ["".join(mode) for mode in all_binding_modes]
    all_binding_formatted = {coordlabel:"".join([coord_chars[i] for i,c in enumerate(coordlabel) if c == "1"])
                             for coordlabel in all_binding_strs}
    all_binding_formatted[zero_label]="Unbound"

    # Get a non-random unique color for each coordination label
    cmap = get_cmap(len(all_binding_strs))
    unique_color_list = {coordlabel: cmap(ind) for ind, coordlabel in enumerate(all_binding_strs)}

    # Start plotting some fake Rectangles for the line plots
    proxy_rects = [Rectangle((0, 0), 0, 0, color="black"), Rectangle((0, 0), 0, 0, color="red")]
    proxy_names = ["Bound", "Unbound"]

    # Subsets of the data for unbound and bound vales.
    unbound_df = df[df["CoordLabel"]==zero_label]
    bound_df = df[df["CoordLabel"]!=zero_label]

    bound_histo, edges = np.histogram(bound_df["Z"], bins=bins, range=[-20,20], normed=False)
    ax.plot(edges[1:], bound_histo*norm_factor, linewidth=2.0, color="#000000")

    unbound_histo, edges = np.histogram(unbound_df["Z"], bins=bins, range=[-20,20], normed=False)
    ax.plot(edges[1:], unbound_histo*norm_factor, linewidth=2.0, color=unique_color_list[zero_label])

    for coordlabel, v in df.groupby("CoordLabel"):
        if coordlabel != zero_label:
            histogram, edges = np.histogram(v["Z"], bins=bins, range=[-20,20], normed=False)
            proxy_rects.append(proxy_fill_between(edges[1:], 0, histogram*norm_factor,
                                                  ax=ax,
                                                  linewidth=0.5,
                                                  facecolor=unique_color_list[coordlabel],
                                                  alpha=0.75))
            proxy_names.append(all_binding_formatted[coordlabel])

    #ax.legend(proxy_rects, proxy_names, bbox_to_anchor=(0., 1.05, 1.0, .102),
    #          ncol=6, mode="expand", borderaxespad=0.)

    _plot_labelling(ax, xlim, ylim, x_label, y_label,
                    tickx=[1.0,5.0],ticky=[0.05,0.1])

    return ax

def plot_mode_histogram(ax, df, bins=300, dt=0.02, xlim=[-20,20], ylim=[0,3],
                         x_label=None, y_label=None,
                         mode_colname = "ModeLabel",
                         all_binding_modes = ["EC","CC","S","E","EL","L","LT","NONE"],
                         norm_factor=1.0, hist_vline=None, sumdist_color="black"):

    """
    A histogram of the dataseries using specified limits,
    plotted on the matplotlib axis argument.

    Parameters
    ----------
    ax : matplotlib.axes.AxesSubplot
        Plotting is performed on this axis.
    df : pandas.core.frame.DataFrame
        A coordination dataframe to be processed.
    dt : float
        The conversion of step to time units (typically nanoseconds).
    bins : int
        Number of histogram bins.
    xlim : list
        Minimum and maximum x-value for the axis range.
    ylim : list
        Minimum and maximum y-value for the axis range.
    x_label : str
        Optional x-label for plotting.
    y_label : str
        Optional y-label for plotting.
    coord_colnames : list
        A list of strings that indicates what column names to plot.
    norm_factor : float
        Factor to multiply normalized histograms by.
    hist_vline : list
        x-axis locations for thick vertical lines.
    sumdist_color : string
        color code used for drawing the total distribution
    """

    # Add vertical line to histogram, useful for showing dividers
    if hist_vline != None:
        for xval in hist_vline:
            ax.axvline(xval, color='red', alpha=0.25, linewidth=5)

    # Get a non-random unique color for each coordination label
    cmap = get_cmap(len(all_binding_modes))
    unique_color_list = {coordlabel: cmap(ind) for ind, coordlabel in enumerate(all_binding_modes)}

    # Start plotting some fake Rectangles for the line plots
    proxy_rects = []
    proxy_names = []

    for coordlabel, v in df.groupby(mode_colname):
        histogram, edges = np.histogram(v["Z"], bins=bins, range=[-20,20], normed=False)
        proxy_rects.append(proxy_fill_between(edges[1:], 0, histogram*norm_factor,
                                              ax=ax,
                                              linewidth=0.5,
                                              facecolor=unique_color_list[coordlabel],
                                              alpha=0.75))
        proxy_names.append(coordlabel)

    # Plot the total distribution
    histogram, edges = np.histogram(df["Z"], bins=bins, range=[-20,20], normed=False)
    ax.plot(edges[1:], histogram*norm_factor, linewidth=2.0, color=sumdist_color)

    ax.legend(proxy_rects, proxy_names, bbox_to_anchor=(0., 1.05, 1.0, .102),
              ncol=6, mode="expand", borderaxespad=0.)

    _plot_labelling(ax, xlim, ylim, x_label, y_label,
                    tickx=[1.0,5.0],ticky=[0.05,0.1])

    return ax

def plot_avgcoord_histogram(ax, df, axis=2, bins=300, xlim=[-20,20], ylim=[0,3],
                         x_label=None, y_label=None, colors={}, line_style="solid",
                         coord_colnames = ["S178","E177","L176","T175","W"],
                         hist_vline=None):

    """
    A histogram showing average coordination of ions as they traverse
    a specified channel axis.

    Parameters
    ----------
    ax : matplotlib.axes.AxesSubplot
        Plotting is performed on this axis.
    df : pandas.core.frame.DataFrame
        A coordination dataframe to be processed.
    bins : int
        Number of histogram bins.
    xlim : list
        Minimum and maximum x-value for the axis range.
    ylim : list
        Minimum and maximum y-value for the axis range.
    x_label : str
        Optional x-label for plotting.
    y_label : str
        Optional y-label for plotting.
    coord_colnames : list
        A list of strings that indicates what column names to plot.
    colors : dict
        A dictionary that matches column headers with colors for plotting.
    hist_vline : list
        x-axis locations for thick vertical lines.
    """

    edges = np.linspace(xlim[0], xlim[1], num=bins)

    z_bins = np.digitize(df["Z"], edges)
    coord_gb = df.groupby(z_bins)

    #if len(np.unique(z_bins)) != bins:
    #    print "Insufficient data along the Z axis: ", len(np.unique(z_bins)), bins

    cmap = get_cmap(len(coord_colnames))
    unique_color_list = {clbl: cmap(ind) for ind, clbl in enumerate(coord_colnames)}

    x = edges+((edges[1]-edges[0])*0.5)
    sum_y = np.zeros(bins+1)

    for coordlabel in coord_colnames:
        y_sampled = coord_gb[coordlabel].mean()

        # We may not sample all y values, so all non-sampled
        # data is set to zero!
        y = np.zeros(bins+1)
        y[y_sampled.index] = y_sampled

        sum_y[y_sampled.index] += y_sampled

        # We skip the first and last values since they correspond to a mean
        # of all data outside our range (but it should be zero anyway)
        if coordlabel in colors:
            ax.plot(x[:-1], y[1:-1], linewidth=2.0, color=colors[coordlabel], ls=line_style)
        else:
            ax.plot(x[:-1], y[1:-1], linewidth=2.0, color=unique_color_list[coordlabel], ls=line_style)

    ax.plot(x[:-1], sum_y[1:-1], color="#000000", linewidth=2.0, ls=line_style)

    _plot_labelling(ax, xlim, ylim, x_label, y_label,
                    tickx=[1.0,5.0],ticky=[0.50,1.0])
                    #tickx=[1.0,5.0],ticky=[1.00,2.0])

    return ax

def plot_coordcount_histogram(ax, df, bins=300, xlim=[-20,20], ylim=[0,3],
                              x_label=None, y_label=None,
                              coord_min=None, coord_max=None, coord_colname="W",
                              norm_factor=1.0, 
                              hist_vline=None):

    """
    A histogram showing average coordination of ions as they traverse
    a specified channel axis.

    Parameters
    ----------
    ax : matplotlib.axes.AxesSubplot
        Plotting is performed on this axis.
    df : pandas.core.frame.DataFrame
        A coordination dataframe to be processed.
    bins : int
        Number of histogram bins.
    xlim : list
        Minimum and maximum x-value for the axis range.
    ylim : list
        Minimum and maximum y-value for the axis range.
    x_label : str
        Optional x-label for plotting.
    y_label : str
        Optional y-label for plotting.
    coord_colnames : list
        A list of strings that indicates what column names to plot.
    colors : dict
        A dictionary that matches column headers with colors for plotting.
    hist_vline : list
        x-axis locations for thick vertical lines.
    """

    if coord_min == None:
        coord_min = df[coord_colname].min()
    if coord_max == None:
        # Typically, the actual coordination max is low population... so -1
        coord_max = df[coord_colname].max()-1

    if coord_max > 10:
        cmap = get_cmap(20)
    else:
        cmap = get_cmap(10)

    unique_color_list = {clbl: cmap(ind) for ind, clbl in enumerate(range(coord_max+1))}

    x = np.linspace(xlim[0], xlim[1], num=bins)

    for coord in range(coord_min, coord_max+1):
        all_z = df[df[coord_colname]==coord]["Z"]
        histogram, edges = np.histogram(all_z, bins=bins, range=xlim, normed=False)
        ax.plot(edges[1:], histogram*norm_factor, linewidth=2.0,
                label=str(coord), color=unique_color_list[coord])

    _plot_labelling(ax, xlim, ylim, x_label, y_label,
                    #tickx=[1.0,5.0],ticky=[0.50,1.0])
                    tickx=[1.0,5.0],ticky=[1.00,2.0])
    ax.legend()

    return ax

def plot_multi_ts(ax, df, dt=0.02, xlim=None, ylim=[0,3], x_label="Time (ns)",
                  y_label="Y", skip=10, ts_vline=None):
    """
    Traditional plot of all timeseries (without error bars) but with
    low opacity on all lines. Useful for visualizing the spread of
    timeseries data without the mean value. Only makes sense for
    continous data like positions or angles.

    Parameters
    ----------
    ax : matplotlib.axes.AxesSubplot
        Plotting is performed on this axis.
    df : pandas.core.frame.DataFrame
        Dataframe wherein we will plot a timeseries for each column key
    dt : float
        The conversion of timestep to units (typically nanoseconds)
    x_label : str
        Optional x-label for plotting.
    y_label : str
        Optional y-label for plotting.
    skip : int
        Number of datapoints to skip (varies depending on dataset size)
    ts_vline : int
        x-axis location in nanoseconds for a thick vertical line.

    """

    # Add shaded background to the time series, indicating equilibration
    #if ts_vline != None:
    #    ax.axvspan(0, ts_vline, facecolor='k', alpha=0.25, linewidth=0)

    for ts in df.columns:
        ax.plot(df[ts].index[::skip]*dt, df[ts].values[::skip], linewidth=0.5,
            color="#000000", alpha=0.1)

    _plot_labelling(ax, xlim, ylim, x_label, y_label)

    return ax

def _plot_labelling(ax, xlim, ylim, x_label=None, y_label=None, tickx=None, ticky=None, gridx=True, gridy=True):
    """
    Helper method for axis modifications (labels, ticks, and range).
    Internal use only.

    Parameters
    ----------
    ax : matplotlib.axes.AxesSubplot
        Plotting is performed on this axis.
    xlim : list
        Minimum and maximum x-value for the axis range.
    ylim : list
        Minimum and maximum y-value for the axis range.
    x_label : str
        Optional x-label for plotting.
    y_label : str
        Optional y-label for plotting.
    tickx : list
        Tick frequency on x-axis (minor, major)
    ticky : list
        Tick frequency on y-axis (minor, major)
    gridx : bool
        Toogle gridlines on the x-axis.
    gridy : bool
        Toggle gridlines on the y-axis.
    """

    if xlim != None:
        ax.set_xlim([xlim[0], xlim[1]])

    if ylim != None:
        ax.set_ylim([ylim[0], ylim[1]])

    if x_label != None:
        ax.set_xlabel(x_label)
    else:
        ax.xaxis.set_major_formatter(nullfmt)

    if y_label != None:
        ax.set_ylabel(y_label)
    else:
        ax.yaxis.set_major_formatter(nullfmt)

    x = ax.get_xaxis()
    y = ax.get_yaxis()

    if tickx != None:
        x.set_minor_locator(mpl.ticker.MultipleLocator(tickx[0]))
        x.set_major_locator(mpl.ticker.MultipleLocator(tickx[1]))
    if ticky != None:
        y.set_minor_locator(mpl.ticker.MultipleLocator(ticky[0]))
        y.set_major_locator(mpl.ticker.MultipleLocator(ticky[1]))

    ax.set_axisbelow(True)
    if gridx:
        x.grid(True, which='minor', linewidth=1, linestyle=':', alpha=0.5, color="grey")
        x.grid(True, which='major', linewidth=2, linestyle=':', alpha=0.8, color="grey")
    if gridy:
        y.grid(True, which='minor', linewidth=1, linestyle=':', alpha=0.5, color="grey")
        y.grid(True, which='major', linewidth=2, linestyle=':', alpha=0.8, color="grey")

    return ax

def plot_histogram_from_ts(ax, s, bins=300, xlim=[-20,20], color="#000000",
                           ylim=[0,10], hist_range=[-20,20],
                           x_label=None, y_label=None, norm_factor=1.0,
                           hist_vline=None, fill=False):

    """
    A histogram of the dataseries using specified limits,
    plotted on the matplotlib axis argument.

    Parameters
    ----------
    ax : matplotlib.axes.AxesSubplot
        Plotting is performed on this axis.
    s : pandas.core.series.Series
        Any column of a dataframe or series to be plotted on an axis.
    bins : int
        Number of histogram bins.
    hist_range : list
        Minimum and maximum x-value for histogram.
    color : str
        Hexadecimal color code for histogram line.
    xlim : list
        Minimum and maximum x-value for the axis range.
    ylim : list
        Minimum and maximum y-value for the axis range.
    x_label : str
        Optional x-label for plotting.
    y_label : str
        Optional y-label for plotting.
    norm_factor : float
        Factor to multiply normalized histogram.
    hist_vline : list
        x-axis locations for thick vertical lines .
    """

    # Add vertical line to histogram, useful for showing dividers
    if hist_vline != None:
        for xval in hist_vline:
            ax.axvline(xval, color='red', alpha=0.25, linewidth=5)

    histogram, edges = np.histogram(s.values, bins=bins, range=hist_range, normed=False)
    if fill:
        proxy_fill_between(edges[1:], 0, norm_factor*histogram,
                           ax=ax,
                           linewidth=0.5, facecolor=color, alpha=0.75)
    else:
        ax.plot(edges[1:], norm_factor*histogram, linewidth=2.0, color=color)

    _plot_labelling(ax, xlim, ylim, x_label, y_label, tickx=[1.0,5.0], ticky=[0.05,0.1])

    return ax

def plot_logseries_with_error(ax, s, xlim=[-20,20], color="#000000",
                           ylim=[0,10],
                           x_label=None, y_label=None,
                           ts_vline=None):

    """
    A histogram of the dataseries using specified limits,
    plotted on the matplotlib axis argument.

    Parameters
    ----------
    ax : matplotlib.axes.AxesSubplot
        Plotting is performed on this axis.
    s : pandas.core.series.Series
        Any column of a dataframe or series to be plotted on an axis.
    color : str
        Hexadecimal color code for histogram line.
    xlim : list
        Minimum and maximum x-value for the axis range.
    ylim : list
        Minimum and maximum y-value for the axis range.
    x_label : str
        Optional x-label for plotting.
    y_label : str
        Optional y-label for plotting.
    norm_factor : float
        Factor to multiply normalized histogram.
    ts_vline : list
        x-axis locations for thick vertical lines .
    """

    # Add vertical line to histogram, useful for showing dividers
    if ts_vline != None:
        for xval in ts_vline:
            ax.axvline(xval, color='red', alpha=0.25, linewidth=5)

    ax.semilogy(s.index, s["Data"], linewidth=2.0, color=color)
    ax.fill_between(s.index, s["Data"]-s["Error"], s["Data"]+s["Error"],
                    alpha=0.5, linewidth=0, color=color)

    _plot_labelling(ax, xlim, ylim, x_label, y_label, tickx=[2.0, 10.0])
    ax.yaxis.set_major_formatter(mpl.ticker.ScalarFormatter())

    return ax

def plot_ts_by_resid_with_trans(ax, df, trans=None, trans_columns=None,
                           state_colors=None, xlim=[-20,20],
                           ylim=[0,10],
                           x_label=None, y_label=None,
                           ts_hline=None, skip=5,
                           limit_state_draw=-1):

    """
    A timeseries with multi-color ion representations but state
    transitions can be passed as well, which will be color-blocked
    onto the timeseries by state name (in the background)

    Parameters
    ----------
    ax : matplotlib.axes.AxesSubplot
        Plotting is performed on this axis.
    df : pandas.core.series.DataFrame
        A dataframe with column ResidID.
    xlim : list
        Minimum and maximum x-value for the axis range.
    ylim : list
        Minimum and maximum y-value for the axis range.
    x_label : str
        Optional x-label for plotting.
    y_label : str
        Optional y-label for plotting.
    ts_hline : list
        x-axis locations for thick vertical lines.
    limit_state_draw : int
        a time value that provides an upper-bound for state lengths
    """

    # Add horizontal line to histogram, useful for showing dividers
    if ts_hline != None:
        for yval in ts_hline:
            ax.axhline(yval, color='black', linewidth=2)

    unique_resid_list = df["ResidID"].unique()
    cmap = get_cmap(len(unique_resid_list))
    unique_color_list = {resid: cmap(ind) for ind, resid in enumerate(unique_resid_list)}

    # We're going to color the background behind the plot but we don't know
    # how many states there are in this trajectory.
    if state_colors != None:
        unique_state_colors = state_colors
    elif (trans != None and trans_columns != None):
        unique_state_list = np.unique(trans[trans_columns].unstack().values)
        cmap = get_cmap(len(unique_state_list))
        unique_state_colors = {state: cmap(ind) for ind, state in enumerate(unique_state_list)}

    gdf = df.sort_values(by=["TrajNum","Time","Z"], ascending=(False,True,False))
    sizes = gdf.groupby(["TrajNum","Time"]).size().values
    gdf["Order"] = np.arange(sizes.sum()) - np.repeat(sizes.cumsum() - sizes, sizes)

    # Plot by type
    typecolors={"NA+":"#2E3192", "K+":"#00A651"}
    for resid, data in df.groupby(["ResidID"]):
        points = data[["Time","Z"]].values
        atomtype = data["RowID"].iloc[0]
        ax.scatter(points[:,0][::skip], points[:,1][::skip], color=typecolors[atomtype], s=1.75)
    '''
    # Plot by ion order
    ordcolors=["red","green","blue","purple","orange","pink","aqua","maroon"]
    for ordatom, data in gdf.groupby("Order"):
        points = data[["Time","Z"]].values
        ax.scatter(points[:,0][::skip], points[:,1][::skip], color=ordcolors[ordatom], s=1.75)
    # Plot by ResID color
    for resid, data in df.groupby(["ResidID"]):
        resid_col = unique_color_list[resid]
        points = data[["Time","Z"]].values
        if len(points) > 20:
            ax.scatter(points[:,0][::skip], points[:,1][::skip], color=resid_col, s=1.75)
            #ax.plot(points[:,0], pd.rolling_mean(points[:,1], 20), color=resid_col, linewidth=0.5)
    '''

    if trans is not None and trans_columns is not None:

        # blocks drawn prevents overlapping states from being drawn
        blocks_drawn = []

        for row in trans.iterrows():
            # iterrows returns (index, Series) tuples, so take only the Series.
            # also keep in mind that there's an extra few columns (TrajNum is there)
            # so we omit the last bundle.
            srow = [row[1][i:i+3].values for i in range(0, len(row[1]), 3)][:-1]
            for block in srow:
                print row[1][-1], block
                if tuple(block) not in blocks_drawn:
                    blocks_drawn.append(tuple(block))
                    state_start = block[1]
                    state_length = block[1]+block[2]
                    if (limit_state_draw > 0):
                        if (state_start < limit_state_draw) and (state_length < limit_state_draw):
                            ax.axvspan(state_start, state_length, ymin=0, ymax=0.05,
                                       color=unique_state_colors[block[0]], alpha=0.9)
                        elif (state_start < limit_state_draw) and (state_length > limit_state_draw):
                            ax.axvspan(state_start, limit_state_draw, ymin=0, ymax=0.05,
                                       color=unique_state_colors[block[0]], alpha=0.9)
                    else:
                        ax.axvspan(state_start, state_length, ymin=0, ymax=0.05,
                                   color=unique_state_colors[block[0]], alpha=0.9)

            #trans_sub = trans[trans["ResidID"]==resid]
            #for row in trans_sub.iterrows():
            #    state_start = row[1]["Frame_Start"]
            #    state_switch = state_start + row[1]["Dwell_Start"]
            #    state_end = row[1]["Frame_End"]+row[1]["Dwell_End"]
            #    #ax.axvline(state_switch, alpha=0.8, linestyle="--",
            #    #            linewidth=2, color=resid_col)

    _plot_labelling(ax, xlim, ylim, x_label, y_label,
                    tickx=[50,100], ticky=[1.0,2.0])

    return ax

def plot_ts_by_order(ax, df, xlim=[-20,20], ylim=[0,10], x_label=None,
                    y_label=None, ts_hline=None, skip=5):

    """
    A timeseries with multi-color ion representations but ranked
    by descent into the channel rather than colored by ResID.

    Parameters
    ----------
    ax : matplotlib.axes.AxesSubplot
        Plotting is performed on this axis.
    df : pandas.core.series.DataFrame
        A dataframe with column ResidID.
    xlim : list
        Minimum and maximum x-value for the axis range.
    ylim : list
        Minimum and maximum y-value for the axis range.
    x_label : str
        Optional x-label for plotting.
    y_label : str
        Optional y-label for plotting.
    ts_hline : list
        x-axis locations for thick vertical lines.
    skip : int
        scatter plotting, so skip this many points (good for rich data)
    """

    # Add horizontal line to histogram, useful for showing dividers
    if ts_hline != None:
        for yval in ts_hline:
            ax.axhline(yval, color='black', linewidth=2)

    gdf = df.sort_values(by=["TrajNum","Time","Z"], ascending=(False,True,False))

    sizes = gdf.groupby(["TrajNum","Time"]).size().values
    gdf["Order"] = np.arange(sizes.sum()) - np.repeat(sizes.cumsum() - sizes, sizes)

    ordcolors=["red","green","blue","purple","orange","pink","aqua","maroon"]
    for ordatom, data in gdf.groupby("Order"):
        points = data[["Time","Z"]].values
        ax.scatter(points[:,0][::skip], points[:,1][::skip], color=ordcolors[ordatom], s=2.0)

    _plot_labelling(ax, xlim, ylim, x_label, y_label,
                    #tickx=[50,100], ticky=[0.2,0.4])
                    tickx=[50,100], ticky=[2.0,4.0])

    return ax

def plot_2d_pmfs(ax, hist,
                 hist_range=[0,360,0,360],
                 x_label="Chi1 (degrees)",
                 y_label="Chi2 (degrees)",
                 x_ticks=[30,60],
                 y_ticks=[30,60],
                 vrange=[0,6],cmap="CMRmap",
                 ):
    """
    Using a 2D histogram, then use this function
    to plot this PMF as a heatmap, with optional text labels

    Parameters
    ----------
    ax : matplotlib.axes.AxesSubplot
        Plotting is performed on this axis.
    df : pandas.core.series.DataFrame
        A dataframe with column ResidID.
    """

    cax = ax.imshow(hist.transpose(), vmin=vrange[0], vmax=vrange[1],
                     extent=hist_range, origin="lower", aspect='auto',
                     interpolation='nearest', cmap=cmap)
    ax.autoscale(False)
    _plot_labelling(ax, hist_range[0:2], hist_range[2:4], x_label, y_label,
                    tickx=x_ticks, ticky=y_ticks)

    return cax

def plot_macrostate_2d_pmfs(ax, axl, axb, coord_df, axis_columns,
                            axis_colors=None,
                            hist_range=[-10,10,-10,10],
                            x_label="axial coordinate",
                            y_label="axial coordinate",
                            x_ticks=[2,4],
                            y_ticks=[2,4],
                            vrange=[0,5], cmap="CMRmap",
                            norm_factor=1.0,
                            residual_pmfs=False,
                            ):
    """
    Using a 2D histogram, then use this function
    to plot this PMF with residuals on the axes

    Parameters
    ----------
    ax : matplotlib.axes.AxesSubplot
        Plotting is performed on this axis.
    df : pandas.core.series.DataFrame
        A dataframe with column ResidID.
    residual_pmfs : bool
        Instead of plotting histograms, plot PMFs on X,Y axes
    """

    if axis_colors == None:
        axis_colors=["red","green",
                     "blue","purple",
                     "orange"][:len(axis_columns)]

    # TODO: relative import?
    from ChannelAnalysis2.Coordination import mean_2d_pmf

    pmf = mean_2d_pmf(coord_df, axis_columns=axis_columns,
                      histrange=hist_range[:2],
                      histbins=100,min_binval=5)
    cax=plot_2d_pmfs(ax, pmf.transpose(),
                     hist_range=hist_range,
                     x_label=x_label,
                     y_label=y_label,
                     x_ticks=x_ticks,
                     y_ticks=y_ticks, vrange=vrange)

    if residual_pmfs:
        from ChannelAnalysis2.Coordination import mean_axial_pmf
        from ChannelAnalysis2.Plot import plot_pmf
        pmfl = mean_axial_pmf(coord_df[axis_columns[0]+["TrajNum"]], min_binval=5,
                              axis_column=axis_columns[0], histrange=hist_range[:2], left_shift=False)
        pmfb = mean_axial_pmf(coord_df[axis_columns[1]+["TrajNum"]], min_binval=5,
                              axis_column=axis_columns[1], histrange=hist_range[:2], left_shift=False)
        plot_pmf(axl, pmfl,
                 color=axis_colors[0],
                 xlim=hist_range[:2], ylim=[-1,5], flipxy=True)
        plot_pmf(axl, pmfb,
                 color=axis_colors[1],
                 xlim=hist_range[:2], ylim=[-1,5], flipxy=True)
        plot_pmf(axb, pmfl,
                 color=axis_colors[0],
                 xlim=hist_range[:2], ylim=[-1,5])
        plot_pmf(axb, pmfb,
                 color=axis_colors[1],
                 xlim=hist_range[:2], ylim=[-1,5])
    else:
        xhisto, xedges = np.histogram(coord_df[axis_columns[0]], bins=250, range=hist_range[:2], normed=True)
        yhisto, yedges = np.histogram(coord_df[axis_columns[1]], bins=250, range=hist_range[:2], normed=True)
        axb.plot(yedges[1:], yhisto*norm_factor, color=axis_colors[1])
        axl.plot(xhisto*norm_factor, xedges[1:], color=axis_colors[0])
        axb.set_ylim([0,1.0])
        axl.set_xlim([0,1.0])

    return cax,axb,axl
