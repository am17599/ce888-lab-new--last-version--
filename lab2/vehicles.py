import matplotlib
matplotlib.use('Agg')


import pandas as pd              
import random 
import matplotlib.pyplot as plt  
import seaborn as sns            
import numpy as np 
  



def mad(arr):
    """ Median Absolute Deviation: a "Robust" version of standard deviation.
        Indices variabililty of the sample.
        https://en.wikipedia.org/wiki/Median_absolute_deviation 
        http://stackoverflow.com/questions/8930370/where-can-i-find-mad-mean-absolute-deviation-in-scipy
    """
    arr = np.ma.array(arr).compressed()   
    med = np.median(arr)
    return np.median(np.abs(arr - med))


if __name__ == "__main__":
	df = pd.read_csv('./vehicles.csv')
	df = df.dropna() 
	print (df.columns)
	
	sns_plot = sns.lmplot(df.columns[0], df.columns[1], data=df, fit_reg=False)   

	sns_plot.axes[0,0].set_ylim(0,)
	sns_plot.axes[0,0].set_xlim(0,)

	plt.xlabel('Proposed Fleet')  
	plt.ylabel('Current Fleet')   

	sns_plot.savefig("v_scaterplot.png",bbox_inches='tight')    
	sns_plot.savefig("v_scaterplot.pdf",bbox_inches='tight')  
 
	data = df.values.T[1]
	
	
	print (("Mean: %f")%(np.mean(data)))
	print (("Median: %f")%(np.median(data)))
	print (("Var: %f")%(np.var(data)))
	print (("std: %f")%(np.std(data)))
	print (("MAD: %f")%(mad(data)))

	plt.clf()
	sns_plot2 = sns.distplot(data, bins=20, kde=False, rug=True).get_figure()

	axes = plt.gca()
	axes.set_xlabel('Proposed fleet') 
	axes.set_ylabel('Current fleet')

	sns_plot2.savefig("v_histogram.png",bbox_inches='tight')
	sns_plot2.savefig("v_histogram.pdf",bbox_inches='tight')


	
