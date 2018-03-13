import re
import tensorflow as tf
import struct
import operator
import collections
import os
from os import listdir
from os.path import isfile, join
from tensorflow.core.example import example_pb2
from numpy.random import shuffle as random_shuffle


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('in_directories', '', 'path to directory')
tf.app.flags.DEFINE_string('out_files', '', 'comma separated paths to files')
tf.app.flags.DEFINE_string('split', '', 'comma separated fractions of data')

VOCAB_LIMIT = 200000
CHUNK_SIZE = 1000 # num examples per chunk, for the chunked data

def text_to_binary(input_directories, output_filenames, split_fractions):
    filenames = get_filenames(input_directories)
    random_shuffle(filenames)
    start_from_index = 0
    counter = collections.Counter()	# for the vocab counts
    for index, output_filename in enumerate(output_filenames):
        sample_count = int(len(filenames) * split_fractions[index])
        print(output_filename + ': ' + str(sample_count))
        end_index = min(start_from_index + sample_count, len(filenames))
        convert_files_to_binary(filenames[start_from_index:end_index], output_filename, counter)
        start_from_index = end_index

    # create vocab file
    with open('vocab', 'w+') as vocab_f:
        for word, count in counter.most_common(VOCAB_LIMIT):
            vocab_f.write(word + ' ' + str(count) + '\n')
        vocab_f.write('<s> 0\n')
        vocab_f.write('</s> 0\n')
        vocab_f.write('<UNK> 0\n')
        vocab_f.write('<PAD> 0\n')

def convert_files_to_binary(input_filenames, output_filename, counter):
	with open(output_filename, 'wb') as serialized_f:
		for filename in input_filenames:
			with open(filename, 'r') as input_f:
				pattern = re.compile(r'<HEADLINE>\n([\w\W]+?)\n</HEADLINE>[\w\W]+?<P>\n([\w\W]+?)\n</P>[\w\W]+?<P>\n([\w\W]+?)\n</P>')
				for match in pattern.findall(input_f.read()):
					abstract, s1, s2 = match

					# split & count words
					counter.update(' '.join([abstract, s1, s2]).split())

					# then create serialized version of abstract/article for training
					abstract = bytes(abstract, 'utf-8')
					article = bytes(' '.join([s1, s2]), 'utf-8')
					tf_example = example_pb2.Example()
					tf_example.features.feature['article'].bytes_list.value.extend([article])
					tf_example.features.feature['abstract'].bytes_list.value.extend([abstract])
					tf_example_str = tf_example.SerializeToString()
					str_len = len(tf_example_str)
					serialized_f.write(struct.pack('q', str_len))
					serialized_f.write(struct.pack('%ds' % str_len, tf_example_str))


def get_filenames(input_directories):
  filenames = []
  for directory_name in input_directories:
    filenames.extend([join(directory_name, f) for f in listdir(directory_name) if isfile(join(directory_name, f))])
  return filenames

def clear_files(output_filenames):
    for output_file in output_filenames:
        try:
            os.remove(output_file)
        except OSError:
            pass

    try:
        os.remove('vocab')
    except OSError:
        pass

def main(unused_args):
    assert FLAGS.in_directories and FLAGS.out_files
    output_filenames = FLAGS.out_files.split(',')
    input_directories = FLAGS.in_directories.split(',')

    clear_files(output_filenames)
    assert FLAGS.split
    split_fractions = [float(s) for s in FLAGS.split.split(',')]
    assert len(output_filenames) == len(split_fractions)
    text_to_binary(input_directories, output_filenames, split_fractions)


if __name__ == '__main__':
	tf.app.run()
