import numpy as np

folder_list_positive_negative_classification = [
"2020_12_01__13_13_18__500",
"2020_05_22__11_40_36__2500",
"2020_05_23__04_00_00__2500",
"2020_05_23__20_19_35__750"
]
folder_list_ucc_classification = [
"2020_12_03__15_06_33__1720",
"2020_05_22__17_34_02__750",
"2020_05_24__07_06_41__2500",
"2020_05_25__01_53_56__3500"
]
folder_list_multi_class_classification = [
"2020_11_30__13_26_13__2440",
"2020_05_19__07_13_10__640",
"2020_05_23__07_00_50__1750",
"2020_05_18__08_59_07__200"
]
folder_list_multi_task_classification = [
"2020_11_28__14_10_14__3000",
"2020_05_23__15_50_06__3080",
"2020_05_17__02_26_00__900",
"2020_05_17__15_29_39__380"
]
folder_list_regression = [
"2020_12_04__13_09_06__4920",
"2020_05_24__08_01_28__3160",
"2020_05_25__12_41_00__4000",
"2020_05_25__12_41_45__4000"
]

folder_list_dict = {	
'positive_negative_classification':folder_list_positive_negative_classification,
'ucc_classification':folder_list_ucc_classification,
'multi_class_classification':folder_list_multi_class_classification,
'multi_task_classification':folder_list_multi_task_classification,
'regression':folder_list_regression,
}

mil_task_list = [
'positive_negative_classification',
'ucc_classification',
'multi_class_classification',
'multi_task_classification',
'regression',
]

mil_pooling_filter_list = ["distribution","mean","attention","max"]

title_list = ['+ve/-ve','ucc','3-class','2-task (N)','2-task (M)','regression']

out_file = 'collected_acc_values.txt'
with open(out_file,'w') as f_out_file:
	f_out_file.write('# mil_pooling_filter\t' + '\t'.join(title_list) + '\n')

	for i,mil_pooling_filter in enumerate(mil_pooling_filter_list):
		f_out_file.write('{}\t'.format(mil_pooling_filter))

		for j, mil_task in enumerate(mil_task_list):
			folder_name = folder_list_dict[mil_task][i]

			if mil_task == 'multi_task_classification':
				file_path = '../{}/test_metrics/{}/test/image_level_statistics_mpp_normal.txt'.format(mil_task,folder_name)
				acc = np.loadtxt(file_path, delimiter='\t', comments='#', dtype=float)
				acc = acc.reshape((1,-1))

				f_out_file.write('{}\t'.format(acc[0,0]))

				file_path = '../{}/test_metrics/{}/test/image_level_statistics_mpp_metastases.txt'.format(mil_task,folder_name)
				acc = np.loadtxt(file_path, delimiter='\t', comments='#', dtype=float)
				acc = acc.reshape((1,-1))

				f_out_file.write('{}\t'.format(acc[0,0]))
				
			else:
				file_path = '../{}/test_metrics/{}/test/image_level_statistics_mpp.txt'.format(mil_task,folder_name)
				acc = np.loadtxt(file_path, delimiter='\t', comments='#', dtype=float)
				acc = acc.reshape((1,-1))

				f_out_file.write('{}\t'.format(acc[0,0]))


		f_out_file.write('\n')
				






