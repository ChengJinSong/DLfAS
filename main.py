""" cnn model based on SAT problem """
# pylint: disable=C0103,C0111
import os
import sys
import numpy as np
import pickle as pk
import tensorflow as tf

import Config as Cfg
import AslibHandler as Ash
import Network as Net

tf.logging.set_verbosity(tf.logging.INFO)


def get_data(scen, config):
    ashan = Ash.AslibHander(config)
    ashan.load_scenarios(scen)
    data = ashan.get_finst()
    # data: { scen:[ [{inst: ....}, #inst...], #fold] }
    # if sys.platform == 'linux':
    #     file_prefix = config['lin_prefix']
    # else:
    #     file_prefix = "C:\\Users\\MN"
    # name = os.path.join(file_prefix, 'Desktop', 'data.pk')
    # print(name)
    # with open(name, 'wb') as f:
    #     pk.dump(data[scen], f)
    return data[scen]


# if __name__ == '__main__':
#     scen = "SAT12-HAND"
#     config = Cfg.Config().get_config_dic()
#     get_data(scen, config)


def cross_validation(scen, config, data, net, shuffle=False):
    """
    :param data: [ [{inst: ....}, #inst...], #fold]
    :return:
    """
    if sys.platform == 'linux':
        file_prefix = config['lin_prefix']
    else:
        file_prefix = config['win_prefix']

    def extract_info(instSet):
        # instSet : [{inst:[]}...]
        instList = []
        feature = []
        label = []
        runtime = []
        for inst in instSet:
            instList.append(inst[0])
            feature.append(inst[2]['image_array'])
            label.append(inst[2]['status_index'])
            runtime.append(inst[2]['run_time'])
        return np.asarray(instList), np.asarray(feature), np.asarray(label), np.asarray(runtime)

    allPre = []

    testCount = 1
    for fold in data:
        index = data.index(fold)

        # if not testCount == index:
        #     continue

        testInstSet = fold
        evalInstSet = data[(index + 1) % 10]
        trainInstSet = []
        for i in range(10):
            if i == index or i == (index + 1) % 10:
                continue
            trainInstSet.extend(data[i])

        testSet = extract_info(testInstSet)
        evalSet = extract_info(evalInstSet)
        trainSet = extract_info(trainInstSet)

        PARA = dict({'batch_size': config['mini_batch_of_training'],
                     'num_label': config['label_dim'], 'batch_size_of_eval': len(evalInstSet)})
        runConf = tf.estimator.RunConfig(
            save_summary_steps=100, log_step_count_steps=100)

        def my_input_fn():
            tds = tf.data.Dataset.from_tensor_slices(trainSet[1:3])
            number_epoch = int(config['total_steps'] /
                               config['number_batch_of_epoch'])
            # iterator = tds.shuffle(1000).repeat(number_epoch).batch(
            #     config['mini_batch_of_training']).make_one_shot_iterator()
            iterator = tds.shuffle(1000).repeat(number_epoch).apply(
                tf.contrib.data.batch_and_drop_remainder(config['mini_batch_of_training'])).make_one_shot_iterator()
            feature, label = iterator.get_next()
            features = {'fea': feature}
            return features, label

        # TODO add data
        tmp_model_name = os.path.join(
            file_prefix, config['tmp_model_path'], "ver_2th", "{}-{}".format(scen, index))
        sat_clf = tf.estimator.Estimator(
            model_fn=net.cnn_model,
            model_dir=tmp_model_name,
            params=PARA,
            config=runConf)

        # Set up logging for predictions
        tensors_to_log = {
            "learning_rate": "learning_rate", "momentum": "momentum"}
        logging_hook = tf.train.LoggingTensorHook(
            tensors=tensors_to_log, every_n_iter=100)

        # training the model
        # sat_clf.train(input_fn=my_input_fn, hooks=[logging_hook])
        sat_clf.train(input_fn=my_input_fn)

        # return
        # evaluate the model
        eval_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={'fea': evalSet[1],
               'runtime': evalSet[3]},
            y=evalSet[2],
            batch_size=len(evalInstSet),
            num_epochs=1,
            shuffle=False)
        eval_res = sat_clf.evaluate(input_fn=eval_input_fn)
        print(eval_res)

        test_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={'fea': testSet[1]},
            batch_size=len(testInstSet),
            num_epochs=1,
            shuffle=False
        )
        test_res = sat_clf.predict(input_fn=test_input_fn)
        tr = list(test_res)
        # pre_name = '{}-pre-{}.pk'.format(os.path.join(config['tmp_model_path'], scen), index)
        # with open(pre_name, 'wb') as f:
        #     pk.dump(tr, f)
        # print(tr)

        # TODO evaluating the test result
        test_batch = len(tr)
        test_index = np.asarray([tr[i]['index'] for i in range(test_batch)])
        test_output = np.asarray([tr[i]['real_output']
                                  for i in range(test_batch)])
        test_par10 = net.PAR10(
            testSet[2], testSet[3], test_index, test_batch, config['label_dim'])
        test_misclas = net.Mis(
            testSet[2], test_output, config['misclas_threshold'], test_batch)
        test_pencen = net.Percentage(
            testSet[2], test_output, config['label_dim'])
        with tf.Session() as ss:
            ss.run(tf.local_variables_initializer())
            # print(ss.run([test_misclas[1], test_misclas[0]]))
            pa, _, pe, mi = ss.run(
                [test_par10[1], test_misclas[1], test_pencen[1], test_misclas[0]])
        test_res = {'Percentage_solverd': pe,
                    'Misclassified_solver': mi, 'PAR10': pa}
        print(test_res)
        allPre.append(test_res)
        res_name = '{}-res-{}.pk'.format(
            os.path.join(file_prefix, config['tmp_model_path'], "ver_2th", scen), index)
        with open(res_name, 'wb') as f:
            pk.dump(allPre, f)

        testCount += 1
        # if testCount == 1:
        #     break


def main():
    scen = "SAT12-HAND"
    config = Cfg.Config().get_config_dic()
    net = Net.Network(config)
    data = get_data(scen, config)
    cross_validation(scen, config, data, net, True)


if __name__ == '__main__':
    main()

    # TODO save train evalution result and test result
