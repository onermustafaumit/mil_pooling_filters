import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind, normaltest
from mpl_toolkits.axes_grid1 import make_axes_locatable
from statsmodels.stats.contingency_tables import mcnemar

mil_filter_list = ["distribution","mean","attention","max"]
num_filters = len(mil_filter_list)

folder_path_list = [ 
"test_metrics/2020_12_01__13_13_18__500/test",
"test_metrics/2020_05_22__11_40_36__2500/test",
"test_metrics/2020_05_23__04_00_00__2500/test",
"test_metrics/2020_05_23__20_19_35__750/test",
]

p_values_arr = np.ones((num_filters,num_filters))
for i in range(num_filters):
	# print(mil_filter_list[i])

	folder1 = folder_path_list[i]
	file1 = '{}/image_predictions_mpp.txt'.format(folder1)
	data1 = np.loadtxt(file1, delimiter='\t', comments='#',dtype=str)
	truth1 = np.asarray(data1[:,1],dtype=int)
	predicted1 = np.asarray(data1[:,2],dtype=int)

	correct_preds1 = (truth1 == predicted1)

	for j in range(i+1,num_filters):
		print('{} - {}'.format(mil_filter_list[i],mil_filter_list[j]))

		folder2 = folder_path_list[j]
		file2 = '{}/image_predictions_mpp.txt'.format(folder2)
		data2 = np.loadtxt(file2, delimiter='\t', comments='#',dtype=str)
		truth2 = np.asarray(data2[:,1],dtype=int)
		predicted2 = np.asarray(data2[:,2],dtype=int)

		correct_preds2 = (truth2 == predicted2)

		contingency_table = np.zeros((2,2), dtype=int)

		# wrong-wrong
		contingency_table[0,0] = np.sum(~correct_preds1 & ~correct_preds2)

		# wrong-correct
		contingency_table[0,1] = np.sum(~correct_preds1 & correct_preds2)

		# correct-wrong
		contingency_table[1,0] = np.sum(correct_preds1 & ~correct_preds2)

		# correct-correct
		contingency_table[1,1] = np.sum(correct_preds1 & correct_preds2)

		print(contingency_table)
		
		#mcnemar test
		result = mcnemar(contingency_table, exact=False, correction=True)

		# summarize the finding
		print('statistic=%.3f, p-value=%.3f' % (result.statistic, result.pvalue))

		# interpret the p-value
		alpha = 0.05
		if result.pvalue > alpha:
			print('Same proportions of errors (fail to reject H0)')
		else:
			print('Different proportions of errors (reject H0)')

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









