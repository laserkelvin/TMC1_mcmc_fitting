import numpy as np
import matplotlib.pylab as pl
import corner
from scipy.stats import mode


def label_offset(ax, axis):
    if axis == "y":
        offsetText = ax.yaxis.offsetText
        ax.yaxis.offsetText.set_visible(False)
        set_label = ax.set_ylabel
        label = ax.get_ylabel()

    elif axis == "x":
        offsetText = ax.xaxis.offsetText
        ax.xaxis.offsetText.set_visible(False)
        set_label = ax.set_xlabel
        label = ax.get_xlabel()

    def update_label(old_label, exponent_text):
        if exponent_text == "":
            return old_label
        
        try:
            units = old_label[old_label.index("[") + 1:old_label.rindex("]")]
        except ValueError:
            units = ""
        label = old_label.replace("[{}]".format(units), "")
        
        exponent_text = exponent_text.replace("\\times", "")
        
        return "{} [{} {}]".format(label, exponent_text, units)

    ax.figure.canvas.draw()
    set_label(update_label(label, offsetText.get_text()))
    return


samples = np.load("cnn/1-cyanonaphthalene/analyzed_chain.npy").T
samples[4:8,:] *= 1e11

#samples = samples[:,samples[13]<0.16]

print(np.percentile(samples,50, axis=1))

f= open("1CNN_formatted_results.txt","w+")

f.write("Source Size 1 = " + '{0:.1f}'.format(np.percentile(samples,50,axis=1)[0]) + " +/- " + '{0:.1f}, {1:.1f}'.format((np.percentile(samples,84,axis=1)[0]-np.percentile(samples,50,axis=1)[0]), (np.percentile(samples,50,axis=1)[0]-np.percentile(samples,16,axis=1)[0])) + " arcsec\n")

f.write("Source Size 2 = " + '{0:.1f}'.format(np.percentile(samples,50,axis=1)[1]) + " +/- " + '{0:.1f}, {1:.1f}'.format((np.percentile(samples,84,axis=1)[1]-np.percentile(samples,50,axis=1)[1]), (np.percentile(samples,50,axis=1)[1]-np.percentile(samples,16,axis=1)[1])) + " arcsec\n")

f.write("Source Size 3 = " + '{0:.1f}'.format(np.percentile(samples,50,axis=1)[2]) + " +/- " + '{0:.1f}, {1:.1f}'.format((np.percentile(samples,84,axis=1)[2]-np.percentile(samples,50,axis=1)[2]), (np.percentile(samples,50,axis=1)[2]-np.percentile(samples,16,axis=1)[2])) + " arcsec\n")

f.write("Source Size 4 = " + '{0:.1f}'.format(np.percentile(samples,50,axis=1)[3]) + " +/- " + '{0:.1f}, {1:.1f}'.format((np.percentile(samples,84,axis=1)[3]-np.percentile(samples,50,axis=1)[3]), (np.percentile(samples,50,axis=1)[3]-np.percentile(samples,16,axis=1)[3])) + " arcsec\n")


f.write("Column Density 1 = " + '{0:.2e}'.format(np.percentile(samples,50,axis=1)[4]) + " +/- " + '{0:.2e}, {1:.2e}'.format((np.percentile(samples,84,axis=1)[4]-np.percentile(samples,50,axis=1)[4]), (np.percentile(samples,50,axis=1)[4]-np.percentile(samples,16,axis=1)[4])) + " cm^-2\n")

f.write("Column Density 2 = " + '{0:.2e}'.format(np.percentile(samples,50,axis=1)[5]) + " +/- " + '{0:.2e}, {1:.2e}'.format((np.percentile(samples,84,axis=1)[5]-np.percentile(samples,50,axis=1)[5]), (np.percentile(samples,50,axis=1)[5]-np.percentile(samples,16,axis=1)[5])) + " cm^-2\n")

f.write("Column Density 3 = " + '{0:.2e}'.format(np.percentile(samples,50,axis=1)[6]) + " +/- " + '{0:.2e}, {1:.2e}'.format((np.percentile(samples,84,axis=1)[6]-np.percentile(samples,50,axis=1)[6]), (np.percentile(samples,50,axis=1)[6]-np.percentile(samples,16,axis=1)[6])) + " cm^-2\n")

f.write("Column Density 4 = " + '{0:.2e}'.format(np.percentile(samples,50,axis=1)[7]) + " +/- " + '{0:.2e}, {1:.2e}'.format((np.percentile(samples,84,axis=1)[7]-np.percentile(samples,50,axis=1)[7]), (np.percentile(samples,50,axis=1)[7]-np.percentile(samples,16,axis=1)[7])) + " cm^-2\n")


f.write("T_ex = " + '{0:.2f}'.format(np.percentile(samples,50,axis=1)[8]) + " +/- " + '{0:.2f}, {1:.2f}'.format((np.percentile(samples,84,axis=1)[8]-np.percentile(samples,50,axis=1)[8]), (np.percentile(samples,50,axis=1)[8]-np.percentile(samples,16,axis=1)[8])) + " K\n")


f.write("V_lsrk 1 = " + '{0:.3f}'.format(np.percentile(samples,50,axis=1)[9]) + " +/- " + '{0:.3f}, {1:.3f}'.format((np.percentile(samples,84,axis=1)[9]-np.percentile(samples,50,axis=1)[9]), (np.percentile(samples,50,axis=1)[9]-np.percentile(samples,16,axis=1)[9])) + " km/s\n")

f.write("V_lsrk 2 = " + '{0:.3f}'.format(np.percentile(samples,50,axis=1)[10]) + " +/- " + '{0:.3f}, {1:.3f}'.format((np.percentile(samples,84,axis=1)[10]-np.percentile(samples,50,axis=1)[10]), (np.percentile(samples,50,axis=1)[10]-np.percentile(samples,16,axis=1)[10])) + " km/s\n")

f.write("V_lsrk 3 = " + '{0:.3f}'.format(np.percentile(samples,50,axis=1)[11]) + " +/- " + '{0:.3f}, {1:.3f}'.format((np.percentile(samples,84,axis=1)[11]-np.percentile(samples,50,axis=1)[11]), (np.percentile(samples,50,axis=1)[11]-np.percentile(samples,16,axis=1)[11])) + " km/s\n")

f.write("V_lsrk 4 = " + '{0:.3f}'.format(np.percentile(samples,50,axis=1)[12]) + " +/- " + '{0:.3f}, {1:.3f}'.format((np.percentile(samples,84,axis=1)[12]-np.percentile(samples,50,axis=1)[12]), (np.percentile(samples,50,axis=1)[12]-np.percentile(samples,16,axis=1)[12])) + " km/s\n")


f.write("dV = " + '{0:.3f}'.format(np.percentile(samples,50,axis=1)[13]) + " +/- " + '{0:.3f}, {1:.3f}'.format((np.percentile(samples,84,axis=1)[13]-np.percentile(samples,50,axis=1)[13]), (np.percentile(samples,50,axis=1)[13]-np.percentile(samples,16,axis=1)[13])) + " km/s\n")

f.close()


f= open("1CNN_results.txt","w+")

f.write("1CNN_hfs.cat\n")

for i in range(14):
    f.write('{0:.5e}'.format(np.percentile(samples,50,axis=1)[i]) + ' ' + '{0:.5e}'.format(np.percentile(samples,84,axis=1)[i]-np.percentile(samples,50,axis=1)[i]) + ' ' + '{0:.5e}'.format(np.percentile(samples,50,axis=1)[i]-np.percentile(samples,16,axis=1)[i]) + "\n")

f.close()


#fig = corner.corner(samples.T, labels=[r'Source Size #1 ["]', r'Source Size #2 ["]', r'Source Size #3 ["]', r'Source Size #4 ["]', r"N$_{\mathrm{col}}$ #1 [cm$^{-2}$]", r"N$_{\mathrm{col}}$ #2 [cm$^{-2}$]", r"N$_{\mathrm{col}}$ #3 [cm$^{-2}$]", r"N$_{\mathrm{col}}$ #4 [cm$^{-2}$]", r"T$_{\mathrm{ex}}$ [K]", r"V$_{\mathrm{LSRK}}$ #1 [km s$^{-1}$]", r"V$_{\mathrm{LSRK}}$ #2 [km s$^{-1}$]", r"V$_{\mathrm{LSRK}}$ #3 [km s$^{-1}$]", r"V$_{\mathrm{LSRK}}$ #4 [km s$^{-1}$]", r"dV [km s$^{-1}$]"], quantiles=[0.16, 0.5, 0.84], show_titles=True, title_kwargs={"fontsize": 11}, title_fmt=[".1f", ".1f", ".1f", ".1f", ".1e", ".1e", ".1e", ".1e", ".2f", ".2f", ".2f", ".2f", ".2f", ".2f"], range=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
#
#leftmost_ax_idx = np.arange(14)*14
#for i in leftmost_ax_idx:
#    fig.axes[i].yaxis.major.formatter._useMathText = True
#    label_offset(fig.axes[i], "y")
#fig.savefig("1CNN_corner.jpg")
