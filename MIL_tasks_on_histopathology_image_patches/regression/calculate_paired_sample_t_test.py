import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind, ttest_rel, normaltest
from mpl_toolkits.axes_grid1 import make_axes_locatable
from statsmodels.stats.contingency_tables import mcnemar

mil_filter_list = ["distribution","mean","attention","max"]
num_filters = len(mil_filter_list)

folder_path_list = [ 
"test_metrics/2020_12_04__13_09_06__4920/test",
"test_metrics/2020_05_24__08_01_28__3160/test",
"test_metrics/2020_05_25__12_41_00__4000/test",
"test_metrics/2020_05_25__12_41_45__4000/test",
]

p_values_arr = np.ones((num_filters,num_filters))
for i in range(num_filters):
	# print(mil_filter_list[i])

	folder1 = folder_path_list[i]
	file1 = '{}/image_predictions_mpp.txt'.format(folder1)
	data1 = np.loadtxt(file1, delimiter='\t', comments='#',dtype=str)
	truth1 = np.asarray(data1[:,1],dtype=float)
	predicted1 = np.asarray(data1[:,2],dtype=float)

	abs_error1 = np.abs(truth1 - predicted1)

	for j in range(i+1,num_filters):
		print('{} - {}'.format(mil_filter_list[i],mil_filter_list[j]))

		folder2 = folder_path_list[j]
		file2 = '{}/image_predictions_mpp.txt'.format(folder2)
		data2 = np.loadtxt(file2, delimiter='\t', comments='#',dtype=str)
		truth2 = np.asarray(data2[:,1],dtype=float)
		predicted2 = np.asarray(data2[:,2],dtype=float)

		abs_error2 = np.abs(truth2 - predicted2)

		difference = abs_error1 - abs_error2

		fig = plt.figure()
		n, bins, patches = plt.hist(difference, 20, density=True, facecolor='g', alpha=0.75)
		plt.xlabel('acc')
		plt.ylabel('probability')
		plt.grid(True)
		# plt.show()

		fig.tight_layout()
		fig_filename = 'test_metrics/{}__{}_histogram.png'.format(mil_filter_list[i],mil_filter_list[j])
		fig.savefig(fig_filename)

		plt.close('all')
		
		#paired sample t-test
		result = ttest_rel(abs_error1, abs_error2)

		# summarize the finding
		print('statistic=%.3f, p-value=%.3f' % (result.statistic, result.pvalue))

		# interpret the p-value
		alpha = 0.05
		if result.pvalue > alpha:
			print('Fail to reject H0')
		else:
			print('Reject H0')

		p_values_arr[i,j] = result.pvalue
		p_values_arr[j,i] = result.pvalue


p_values_filename = 'test_metrics/p_values.txt'
np.savetxt(p_values_filename, p_values_arr, delimiter='\t', comments='# ', header='\t'.join(mil_filter_list)) #fmt='%.4f',


p_values_color_coded = p_values_arr.copy()
p_values_color_coded[p_values_arr > 0.05] = 1
p_values_color_coded[p_values_arr <= 0.05] = 0.67
p_values_color_coded[p_values_arr <= 0.01] = 0.34
p_values_color_coded[p_values_arr <= 0.001] = 0.0


plt.rcParams.update({'font.size':8, 'font.family':'Times New Roman'})

fig,ax=plt.subplots(figsize=(2,1.5))

im = ax.imshow(p_values_color_coded, interpolation='nearest', cmap=plt.cm.viridis)
tick_marks = np.arange(len(mil_filter_list))
ax.set_xticks(tick_marks)
ax.set_yticks(tick_marks)
ax.set_xticklabels(mil_filter_list, rotation=90)
ax.set_yticklabels(mil_filter_list)
ax.set_ylim( (len(mil_filter_list)-0.5, -0.5) )

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = fig.colorbar(im, cax=cax)
cbar.set_ticks([0,0.34,0.67,1.])
cbar.ax.set_yticklabels(['p<=0.001','p<=0.01','p<=0.05','p>0.05'])
# cbar.ax.set_ylabel('p-value')

fig.tight_layout()
fig_filename = 'test_metrics/p_values_color_coded.png'
fig.savefig(fig_filename, dpi=200)


fig,ax=plt.subplots(figsize=(2,1.5))
im = ax.imshow(p_values_arr, interpolation='nearest', cmap=plt.cm.viridis)
tick_marks = np.arange(len(mil_filter_list))
ax.set_xticks(tick_marks)
ax.set_yticks(tick_marks)
ax.set_xticklabels(mil_filter_list, rotation=90)
ax.set_yticklabels(mil_filter_list)
ax.set_ylim( (len(mil_filter_list)-0.5, -0.5) )

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = fig.colorbar(im, cax=cax)
cbar.set_ticks([0,0.34,0.67,1.])
# cbar.ax.set_ylabel('p-value')

fig.tight_layout()
fig_filename = 'test_metrics/p_values.png'
fig.savefig(fig_filename, dpi=200)

# plt.show()









