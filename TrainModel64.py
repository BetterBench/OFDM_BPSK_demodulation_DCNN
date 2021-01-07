import tensorflow as tf
import numpy as np
#from dcnn.data_reader import AsciiSignalSource, RealDataSource, add_noise
# from dcnn.model import DCNN, Logreg
from dcnn.model64 import DCNN
import os
import datetime
import scipy.io as sio

flags = tf.flags
flags.DEFINE_string('data_dir', 'data',
                    'data directory. Should contain train_text..txt, valid_text.txt, test_text.txt')
flags.DEFINE_string('train_dir', 'cv', 'training directory (models and summaries are saved there periodically)')
flags.DEFINE_string('summaries_dir', 'summaries',
                    'directory to store tensorboard summaries')
flags.DEFINE_string('load_model', None,
                    '(optional) filename of the model to load. Useful for re-starting training from a checkpoint')
flags.DEFINE_integer('test_signal_id', None, 'Id of the signal to be used as a test set')
flags.DEFINE_integer('valid_signal_id', None, 'Id of the signal to be used as a validation set')
flags.DEFINE_string('suffix', datetime.datetime.today().strftime("%Y-%m-%d-%H.%M.%S"), 'Suffix of the model')
FLAGS = flags.FLAGS 

F1 = 984.0                                                                                                                              
F2 = 966
FS = 14648
BR = 2
BIT_LEN = int(FS / BR)


if __name__ == '__main__':
    SNR_levels = [None]
    mat_name = 'bpsk-data100.mat'
    matfile = sio.loadmat(mat_name)
    #训练集和训练标签
    train_features = matfile['rxTraining']
    train_labels = matfile['rxTrainingLabel']
    #验证集和验证标签
    valid_features = matfile['rxValidation']
    valid_labels = matfile['rxTrainingLabel']
    #测试集和测试标签
    test_features = matfile['rxTest']
    test_labels = matfile['rxTestLabel']


    training_set = tf.data.Dataset.zip((tf.data.Dataset.from_tensor_slices(train_features), tf.data.Dataset.from_tensor_slices(train_labels))).batch(32)
    validation_set = tf.data.Dataset.zip((tf.data.Dataset.from_tensor_slices(valid_features), tf.data.Dataset.from_tensor_slices(valid_labels))).batch(32)
    test_set = tf.data.Dataset.zip((tf.data.Dataset.from_tensor_slices(test_features), tf.data.Dataset.from_tensor_slices(test_labels))).batch(32)
    iterator = tf.data.Iterator.from_structure(training_set.output_types, training_set.output_shapes)
    next_element = iterator.get_next()

    # next_element[0]是训练集，next_element[0]是训练标签
    demodulation_cnn = DCNN(next_element[0], next_element[1])  
    #将所有summary全部保存到磁盘，以便tensorboard显示。如果没有特殊要求，一般用这一句就可一显示训练时的各种信息了。     
    merged = tf.summary.merge_all()
    step = 0
    loss_per_epoch = tf.placeholder(dtype=tf.float32)
    acc_per_epoch = tf.placeholder(dtype=tf.float32)
    loss_summary_per_epoch = tf.summary.scalar('loss_per_epoch', loss_per_epoch)
    acc_summary_per_epoch = tf.summary.scalar('acc_per_epoch', acc_per_epoch)
    overfit_indicator = 0
    best_valid_loss = 10000
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        #将训练过程数据保存在filewriter指定的文件中
        train_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/train_' + FLAGS.suffix,sess.graph)
        valid_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/valid_' + FLAGS.suffix)

        for epoch in range(10000):
            sess.run(iterator.make_initializer(training_set))
            train_loss = 0
            train_correct_predictions = 0
            while True:
                try:
                    _, loss, correct_prediction, summary_value = sess.run([demodulation_cnn.train_op, demodulation_cnn.loss, demodulation_cnn.correct_prediction, merged])
                    step += 1
                    train_loss += loss
                    train_correct_predictions += correct_prediction
                    #调用train_writer的add_summary方法将训练过程以及训练步数保存
                    train_writer.add_summary(summary_value, step)
                except tf.errors.OutOfRangeError:
                    break
            #训练验证集
            sess.run(iterator.make_initializer(validation_set))
            valid_loss = 0
            valid_correct_predictions = 0
            while True:
                try:
                    loss, correct_prediction, summary_value = sess.run([demodulation_cnn.loss, demodulation_cnn.correct_prediction, merged])
                    valid_loss += loss
                    valid_correct_predictions += correct_prediction
                except tf.errors.OutOfRangeError:
                    break
            #运行测试集
            sess.run(iterator.make_initializer(test_set))
            test_loss = 0
            test_correct_predictions = 0
            while True:
                try:
                    loss, correct_prediction, _ = sess.run([demodulation_cnn.loss, demodulation_cnn.correct_prediction, merged])
                    test_loss += loss
                    test_correct_predictions += correct_prediction
                except tf.errors.OutOfRangeError:
                    break
            train_writer.add_summary(
                loss_summary_per_epoch.eval(feed_dict={loss_per_epoch: train_loss / train_features.shape[0]}),
                epoch)
            train_writer.add_summary(
                acc_summary_per_epoch.eval(feed_dict={acc_per_epoch: train_correct_predictions / train_features.shape[0]}),
                epoch)
            valid_writer.add_summary(
                loss_summary_per_epoch.eval(feed_dict={loss_per_epoch: valid_loss / valid_features.shape[0]}),
                epoch)
            valid_writer.add_summary(
                acc_summary_per_epoch.eval(feed_dict={acc_per_epoch: valid_correct_predictions / valid_features.shape[0]}),
                epoch)
            print("Epoch: {}. Step: {} Train Loss: {} Valid Loss: {} Train acc.: {} Valid acc.: {} Test acc.:{}".format(epoch, step, train_loss / train_features.shape[0],
                        valid_loss / valid_features.shape[0], train_correct_predictions / train_features.shape[0], valid_correct_predictions / valid_features.shape[0], test_correct_predictions / test_features.shape[0]))
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                best_test_loss = test_loss
                best_test_acc = test_correct_predictions / test_features.shape[0]
                best_valid_acc = valid_correct_predictions / valid_features.shape[0]
            if valid_loss > best_valid_loss:
                overfit_indicator += 1
                print('+1')
            else:
                overfit_indicator = 0
            if overfit_indicator > 50:
                print('结束了')
                break
        print("Finished! Best_valid_loss: {} Valid_acc.@best_valid_loss: {} Test_loss@Best_valid_loss: {} Test_acc.@Best_valid_loss: {}".format(best_valid_loss, best_valid_acc, best_test_loss, best_test_acc))
