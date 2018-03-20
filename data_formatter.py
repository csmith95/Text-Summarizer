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
from lexrank import lexrank
import pickle


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('in_directories', '', 'path to directory')
tf.app.flags.DEFINE_string('out_files', '', 'comma separated paths to files')
tf.app.flags.DEFINE_string('split', '', 'comma separated fractions of data')
tf.app.flags.DEFINE_string('lexrank', '', 'indicates to use lexrank')
tf.app.flags.DEFINE_boolean('glove', False, 'whether to use glove pretrained word vectors')

VOCAB_LIMIT = 200000
SENTENCE_START = '<s>'
SENTENCE_END = '</s>'


def lexrankSentences(match):
    abstract, text = match
    pattern2 = re.compile(r'<P>([\w\W]+?)</P>')
    pattern3 = re.compile(r'\n')
    text = pattern3.sub(" ",text)
    sentences = pattern2.findall(text)
    result = lexrank(sentences)
    print(result)
    return abstract, s1, s2

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
        for word, count in counter.most_common(VOCAB_LIMIT-4):
            vocab_f.write(word + ' ' + str(count) + '\n')
        vocab_f.write('<s> 0\n')
        vocab_f.write('</s> 0\n')
        vocab_f.write('<UNK> 0\n')
        vocab_f.write('<PAD> 0\n')

def modify(s):
    s = re.sub(r"\s+", " ", s).lower()
    s = re.sub(r"[()\",\[\_:;\]]", "", s)  # strip parentheses, quotations, commas
    if len(s) > 0 and s[-1] in ['.', '?', '!']:
          s = s[:-1]
    return s

def convert_files_to_binary(input_filenames, output_filename, counter):
    with open(output_filename, 'wb') as serialized_f:
        for filename in input_filenames:
            with open(filename, 'r') as input_f:
                pattern = re.compile(r'<HEADLINE>\n([\w\W]+?)\n</HEADLINE>[\w\W]+?<P>\n([\w\W]+?)\n</P>[\w\W]+?<P>\n([\w\W]+?)\n</P>')
                if FLAGS.lexrank == True:
                    pattern = re.compile(r'<HEADLINE>\n([\w\W]+?)\n</HEADLINE>[\w\W]+?<TEXT>\n([\w\W]+?)\n</TEXT>')
                for match in pattern.findall(input_f.read()):

                    print("h")
                    if FLAGS.lexrank == True: 
                        print("lexrank true")
                        abstract, s1, s2 = lexrankSentences(match)
                    else: 
                        abstract, s1, s2 = match

                    # split & count words
                    abstract = modify(abstract)
                    s1 = modify(s1)
                    s2 = modify(s2)
                    if len(abstract) == 0 or len(s1) == 0 or len(s2) == 0: continue
                    counter.update(' '.join([abstract, s1, s2]).split())

					# then create serialized version of abstract/article for training
                    article = ' '.join([s1, s2])
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

def writeGloveEmbeddings():
    d = {}
    glove_vocab = []
    with open('glove_pretrained/glove.6B.100d.txt', 'r') as f:
        content = f.readlines()
        content = [x.split(" ") for x in content]
        for wordVector in content:
            glove_vocab.append(wordVector[0])
            wordVector[-1] = wordVector[-1].replace('\n', '')
            d[wordVector[0]] = wordVector[1:]
    if not os.path.exists('glove_embeddings.p'):
        pickle.dump(d, open("glove_embeddings.p", "wb" ), protocol=2)
    if not os.path.exists('glove_vocab.txt'):
        target2 = open('glove_vocab.txt', 'a')
        for w in glove_vocab:
            target2.write("%s\n" % w)


def main(unused_args):
    if FLAGS.glove == True: 
        writeGloveEmbeddings()
        # return # don't overwrite other files
    assert FLAGS.split
    assert FLAGS.in_directories and FLAGS.out_files
    output_filenames = FLAGS.out_files.split(',')
    input_directories = FLAGS.in_directories.split(',')
    clear_files(output_filenames)
    split_fractions = [float(s) for s in FLAGS.split.split(',')]
    assert len(output_filenames) == len(split_fractions)
    text_to_binary(input_directories, output_filenames, split_fractions)


if __name__ == '__main__':
	tf.app.run()
