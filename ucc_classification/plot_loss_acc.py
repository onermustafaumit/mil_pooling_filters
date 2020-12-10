import argparse
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser(description='Plot the loss vs iteration and accuracy vs iteration for given data file')
parser.add_argument('--data_file', help='Data file path', dest='data_file')
parser.add_argument('--step_size', default=1, type=int, help='Data file path', dest='step_size')
parser.add_argument('--filter_size', default=1, type=int, help='Data file path', dest='filter_size')
FLAGS = parser.parse_args()

w = FLAGS.filter_size

data_arr = np.loadtxt(FLAGS.data_file, dtype='float', comments='#', delimiter='\t')

# steps = data_arr[:,0]
steps = np.arange(data_arr.shape[0])
train_acc = data_arr[:,1]
train_loss = data_arr[:,2]
val_acc = data_arr[:,3]
val_loss = data_arr[:,4]


def moving_avg_filter(data_arr, w):
	data_arr_cumsum = np.cumsum(data_arr)
	data_arr_cumsum[w:] = (data_arr_cumsum[w:] - data_arr_cumsum[:-w])
	data_arr_filtered = data_arr_cumsum[w-1:]/w

	return data_arr_filtered

if w>1:
	steps = steps[w-1:]
	train_acc = moving_avg_filter(train_acc,w)
	train_loss = moving_avg_filter(train_loss,w)
	val_acc = moving_avg_filter(val_acc,w)
	val_loss = moving_avg_filter(val_loss,w)


ind_start = 0
ind_step = FLAGS.step_size
# ind_end = min(110,len(steps))
ind_end = len(steps)

fig = plt.figure(1)
plt.plot(steps[ind_start:ind_end:ind_step], train_loss[ind_start:ind_end:ind_step], 'r', label="train")
plt.plot(steps[ind_start:ind_end:ind_step], val_loss[ind_start:ind_end:ind_step], 'b', label="val")
plt.title('loss vs epoch')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.grid(linestyle='--')
plt.legend()
# fig.savefig('{}__loss.png'.format(FLAGS.data_file[:-4]), bbox_inches='tight')
# plt.show()

fig = plt.figure(2)
plt.plot(steps[ind_start:ind_end:ind_step], train_acc[ind_start:ind_end:ind_step], 'r', label="train")
plt.plot(steps[ind_start:ind_end:ind_step], val_acc[ind_start:ind_end:ind_step], 'b', label="val")
plt.title('acc vs epoch')
plt.xlabel('epoch')
plt.ylabel('acc')
plt.grid(linestyle='--')
plt.legend()
# fig.savefig('{}__acc.png'.format(FLAGS.data_file[:-4]), bbox_inches='tight')
plt.show()

	
