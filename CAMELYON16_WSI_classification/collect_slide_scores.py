import argparse
import numpy as np


parser = argparse.ArgumentParser(description='')

parser.add_argument('--slide_list_filename', default='./dataset/test.txt', help='slide list test', dest='slide_list_filename')
parser.add_argument('--metrics_dir', default='', help='Text file to write metrics', dest='metrics_dir')

FLAGS = parser.parse_args()
    

# read slide list
data_arr = np.loadtxt(FLAGS.slide_list_filename, delimiter='\t', comments='#', dtype=str)
slide_ids = data_arr[:,0]
labels = np.asarray(data_arr[:,1], dtype=int)
num_slides = slide_ids.shape[0]
print('num_slides:{}'.format(num_slides))

out_file = '{}/slide_scores.txt'.format(FLAGS.metrics_dir)
with open(out_file, 'w') as f_out_file:
    f_out_file.write('# slide_id\tslide_label\tslide_score_pos\n')


    for s, slide_id in enumerate(slide_ids):
        print('slide {}/{}: {}'.format(s+1,num_slides,slide_id))

        slide_label = labels[s]

        # read metrics file 
        # slide_id\tbag_id\tslide_label\tprediction\tscore_negative\tscore_positive\n
        metrics_file = '{}/test_scores__{}.txt'.format(FLAGS.metrics_dir,slide_id)

        data = np.loadtxt(metrics_file, delimiter='\t', comments='#', dtype=str)
        data = data.reshape((-1,6))
        score_pos_arr = np.asarray(data[:,-1], dtype=float)

        slide_score_pos = np.amax(score_pos_arr)

        f_out_file.write('{}\t{}\t{}\n'.format(slide_id,slide_label,slide_score_pos))





        