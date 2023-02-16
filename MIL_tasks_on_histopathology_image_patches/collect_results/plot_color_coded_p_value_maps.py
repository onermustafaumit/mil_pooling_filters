import numpy as np
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind, normaltest
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.font_manager as font_manager

filepath_list = ["positive_negative_classification/test_metrics/p_values.txt",
"ucc_classification/test_metrics/p_values.txt",
"multi_class_classification/test_metrics/p_values.txt",
"multi_task_classification/test_metrics/p_values_normal.txt",
"multi_task_classification/test_metrics/p_values_metastases.txt",
"regression/test_metrics/p_values.txt",]

mil_filter_list = ["distribution","mean","attention","max"]
num_filters = len(mil_filter_list)

title_list = ['+ve/-ve',
'ucc',
'3-class',
'2-task (N)',
'2-task (M)',
'% metastases']

plt.rcParams.update({'font.size':8, 'font.family':'Times New Roman'})

fig, ax2 = plt.subplots(1,6,figsize=(5.5, 1.1))
for i in range(len(filepath_list)):
	filepath = '../{}'.format(filepath_list[i])

	p_values_arr = np.loadtxt(filepath, delimiter='\t', comments='#', dtype=float)

	# p_values_color_coded = np.ones((4,4,3),dtype=np.uint8)*255
	p_values_color_coded = np.ones((num_filters-1,num_filters-1,3),dtype=np.uint8)*255
	for m in range(num_filters-1):
		for n in range(m+1,num_filters):
			if p_values_arr[m,n] <= 0.001:
				p_values_color_coded[m,n-1,:]=[188,0,0]
			elif p_values_arr[m,n] <= 0.01:
				p_values_color_coded[m,n-1,:]=[230,75,53]
			elif p_values_arr[m,n] <= 0.05:
				p_values_color_coded[m,n-1,:]=[243,155,127]
			else:
				p_values_color_coded[m,n-1,:]=[145,209,194]

	ax = ax2[i]

	im = ax.imshow(p_values_color_coded)
	tick_marks = np.arange(len(mil_filter_list)-1)
	ax.set_title(title_list[i],fontsize=8)
	ax.set_xticks(tick_marks)
	ax.set_xticklabels(mil_filter_list[1:], rotation=90)
	ax.set_ylim( (len(mil_filter_list)-1-0.5, -0.5) )
	ax.set_yticks(tick_marks)

	if i==0:
		ax.set_yticklabels(mil_filter_list[:-1])

	else:
		ax.set_yticklabels(['','','','',''])

	# Minor ticks
	ax.tick_params(which='minor', length=0, color='w')
	ax.set_xticks(np.arange(-.5, num_filters-1, 1), minor=True);
	ax.set_yticks(np.arange(-.5, num_filters-1, 1), minor=True);

	# Gridlines based on minor ticks
	ax.grid(which='minor', color='w', linestyle='-', linewidth=2)

rect1 = mlines.Line2D([], [], marker="s", markersize=5, linewidth=0, color=np.array([188,0,0])/255)
rect2 = mlines.Line2D([], [], marker="s", markersize=5, linewidth=0, color=np.array([230,75,53])/255)
rect3 = mlines.Line2D([], [], marker="s", markersize=5, linewidth=0, color=np.array([243,155,127])/255)
rect4 = mlines.Line2D([], [], marker="s", markersize=5, linewidth=0, color=np.array([145,209,194])/255)

fig.legend((rect4, rect3, rect2, rect1), ('$p>0.05$','$p\leq0.05$','$p\leq0.01$','$p\leq0.001$'), loc='upper right', bbox_to_anchor=(1.0, 0.93), fancybox=True, shadow=False, ncol=1,fontsize=6,frameon=True)

fig.subplots_adjust(left=0.11, bottom=0.46, right=0.85, top=0.85, wspace=0.20 ,hspace=0.20 )

# fig.savefig('color_coded_p_value_maps.png', dpi=200)
fig.savefig('color_coded_p_value_maps.pdf', dpi=200)

plt.show()










