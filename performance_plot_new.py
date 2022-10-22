def performance_plot() -> None:
    """
    :return: trains each model using train_model in a batch style and plots the loss over epochs, displays
             the accuracy metrics
    """

    models['DNN'].fit(X_train, y_train)
    models['CNN'].fit(X_train, y_train)
    models['LSTM'].fit(np.array(X_train)[:,:, np.newaxis], np.array(y_train))
    models['RNN'].fit(np.array(X_train).reshape([-1, X_train.shape[1], 1]), y_train)


    print('after training metrics')
    dnn_metrics = evaluation_metrics(dnn, X_eval, y_eval)
    cnn_metrics = evaluation_metrics(cnn, X_eval, y_eval)
    lstm_metrics = evaluation_metrics(models['LSTM'], np.array(X_eval)[:,:, np.newaxis], np.array(y_eval))
    rnn_metrics = evaluation_metrics(rnn, np.array(X_eval).reshape([-1, X_eval.shape[1], 1]), y_eval)

    print('dnn', dnn_metrics)
    print('cnn', cnn_metrics)
    print('lstm', lstm_metrics)
    print('rnn', rnn_metrics)

    plt.figure()

    fig, ax = plt.subplots()
    x = np.arange(len(dnn_metrics))
    width = 0.2  # the width of the bars

    rects1 = ax.bar(x - width, list(dnn_metrics.values()), width, label='DNN')
    rects2 = ax.bar(x + width , list(cnn_metrics.values()), width, label='CNN')
    rects3 = ax.bar(x, list(lstm_metrics.values()), width, label='LSTM')
    rects4 = ax.bar(x - 2 * width, list(rnn_metrics.values()), width, label='RNN')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Scores', fontsize=15)
    ax.set_title('Model Performance metrics', fontsize=15)
    ax.set_xticks(x, list(dnn_metrics.keys()), fontsize=15)
    ax.set_ylim(0, 1)
    ax.legend()

    ax.bar_label(rects1, padding=3)
    ax.bar_label(rects2, padding=3)
    ax.bar_label(rects3, padding=3)
    ax.bar_label(rects4, padding=3)

    fig.tight_layout()

    #plt.figure()
    #m = tf.keras.metrics.AUC(curve='ROC')
    #m.update_state(y_eval, models['DNN'].predict(X_eval))

    #plot_roc_curve(dnn, X_eval, y_eval)

    plt.show()


def myround(predictions):
    return np.array([1 if t > 0.5 else 0 for t in predictions])

def evaluation_metrics(model, X_eval, y_eval):
    """
    :param model:
    :param X_eval:
    :param y_eval:
    :return:
    """
    assert hasattr(model, 'predict')

    y_pred = myround(model.predict(X_eval))
    tn, fp, fn, tp = confusion_matrix(y_eval, y_pred, labels=[0, 1], normalize='all').ravel()

    return {'true negative': tn, 'false positive': fp, 'false negative': fn, 'true positive': tp}

X_train, X_eval, y_train, y_eval = preprocess_UNSW()

n = X_train.shape[1]

dnn = CustomNN(n, tf.keras.initializers.he_uniform)

cnn = CNN_Model(n, 2)

lstm = SequeClassifier(128)
lstm.LSTM_model()

rnn = RNN_base()

dnn.compile(optimizer='Adam', loss=tf.losses.binary_crossentropy, metrics=[tf.metrics.TruePositives()])
cnn.compile(loss=tf.losses.binary_crossentropy, optimizer='adam', metrics=[tf.metrics.TruePositives()])
lstm.model.compile(loss=tf.keras.losses.binary_crossentropy, optimizer='adam', metrics=[tf.metrics.TruePositives()])
rnn.compile(loss=tf.losses.binary_crossentropy, optimizer='adam', metrics=[tf.metrics.TruePositives()])

models = {'DNN': dnn, 'CNN': cnn, 'LSTM':lstm.model, 'RNN':rnn}
loss_funcs = {'DNN': log_loss, 'CNN': log_loss, 'LSTM':log_loss, 'RNN':log_loss}

X_train = X_train.iloc[0:50, :]        # for speed
y_train = y_train.iloc[0:50, :]
X_eval = X_eval.iloc[0:50, :]
y_eval = y_eval.iloc[0:50, :]

performance_plot()