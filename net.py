import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


def squash(cap_input):
    with tf.name_scope('SQUASH'):
        # compute norm square of inputs with the last axis, keep dims for broadcasting
        # ||s_j||^2 in paper
        input_norm_square = tf.reduce_sum(tf.square(cap_input), axis=-1, keep_dims=True)
        # ||s_j||^2 / (1. + ||s_j||^2) * (s_j / ||s_j||)
        scaling = input_norm_square / (1. + input_norm_square) / tf.sqrt(input_norm_square)
    return cap_input * scaling

def dynamic_routing(u_ij):
    '''

    :param u_ij: [None,1152,10,16]
    :return:
    '''
    with tf.name_scope("ROUTING"):
        # Prior [10,1152]
        b_IJ = tf.get_variable('b_IJ', [1152,10], dtype=tf.float32, initializer=tf.zeros_initializer(),
                               trainable=False)


        def update_prior(iter,b_IJ,u_ij,vj):
            c_IJ = tf.nn.softmax(b_IJ, dim=0, name="c_IJ")
            #[1152,10,1]
            c_IJ = tf.expand_dims(c_IJ,axis=-1)
            # batch x 1152x10x16 -> batch x 10 x 16
            sj = tf.reduce_sum(tf.multiply(u_ij, c_IJ), axis=-3)
            # do squash [None,10,16]
            vj = squash(sj)
            # [None,10,1,16]
            vj_expand = tf.expand_dims(vj, axis=-3)
            # u_ij [None,1152,10,16] dot vj_expand [None,1,10,16] = [None,1152,10,16]
            # sum over the last dimension [None,1152,10]
            dot_product = tf.reduce_sum(tf.multiply(u_ij,vj_expand),axis=-1,name="dot_product")
            # [None,10,1152] reduce_mean [1152,10]
            b_IJ_delta = tf.reduce_mean(dot_product,axis=0)
            # update Prior [1152,10]
            b_IJ = b_IJ + b_IJ_delta
            return iter-1, b_IJ, u_ij,vj

        _,_,_,vj = update_prior(3,b_IJ,u_IJ,None)
        cond = lambda i,b,u,v: i>0
        _,_,_,vj = tf.while_loop(cond,update_prior,[2,b_IJ,u_ij,vj])

    return vj

tf.reset_default_graph()
graph = tf.get_default_graph()

with graph.as_default():

    with tf.variable_scope("INPUT"):
        X = tf.placeholder(tf.float32, [None, 784], name="X")
        X_img = tf.reshape(X,[-1,28,28])
        y = tf.placeholder(tf.int32, (None), name="y")
        y_onehot = tf.one_hot(y, 10,dtype=tf.float32)

    with tf.variable_scope("CONV1"):
        conv1 = tf.contrib.layers.conv2d(X_img, 256, kernel_size=9, stride=1, padding='VALID')
        conv1 = tf.identity(conv1, name="CONV1")

    with tf.variable_scope("PRIMARY_CAP"):
        conv2 = tf.contrib.layers.conv2d(conv1, 256, kernel_size=9, stride=2, padding='VALID')
        # [None, 6*6*256]
        conv2 = tf.identity(conv2, name="CONV2")
        # [None, 1152,1,8]
        u_I = squash(tf.reshape(conv2, (-1, 1152,1,8)))

    with tf.variable_scope("ROUTING"):
        W_JI = tf.get_variable('W_JI', [10,1152,8,16], dtype=tf.float32, initializer=tf.random_normal_initializer())
        # [None,10,1152,1,16]
        u_JI = tf.map_fn(lambda ui: tf.map_fn(lambda wi:tf.matmul(ui,wi), W_JI),u_I)
        # [None,10,1152,16]
        u_JI = tf.squeeze(u_JI,axis=-2)
        # [None,1152,10,16]
        u_IJ =tf.transpose(u_JI,[0,2,1,3],name="u_IJ")
        # [None,10,16]
        vj = dynamic_routing(u_IJ)

    with tf.variable_scope("COST"):
        # [None,10]
        vj_norm = tf.norm(vj, ord=2, axis=-1,name='digit_caps_norm')
        los_true = tf.square(tf.maximum(0., 0.9 -vj_norm ))
        los_false = tf.square(tf.maximum(0., vj_norm - 0.1))
        loss = y_onehot * los_true + 0.5 * (1 - y_onehot) * los_false
        margin_loss = tf.reduce_mean(tf.reduce_sum(loss, axis=-1))

    with tf.variable_scope("TRAIN"):
        train_op = tf.train.AdamOptimizer().minimize(margin_loss)

    with tf.variable_scope("TEST"):
        predictions = tf.argmax(vj_norm, axis=-1, output_type=tf.int32)
        accuracy = tf.reduce_mean(tf.cast(tf.equal(y, predictions), tf.float32))


mnist = input_data.read_data_sets('data/',one_hot=False)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    train_X,train_y = mnist.train.next_batch(16)
    print(len(train_X))
    sess.run([conv1],feed_dict={X:train_X,y:train_y})