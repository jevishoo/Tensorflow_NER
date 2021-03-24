import tensorflow as tf
from tf_utils.bert_modeling import get_assignment_map_from_checkpoint
from tensorflow.contrib.crf import crf_log_likelihood
from tensorflow.contrib.layers.python.layers import initializers
from tf_utils import rnncell as rnn
from optimization import create_optimizer


class Model:
    def __init__(self, config):
        self.config = config
        # 喂入模型的数据占位符
        self.input_x_word = tf.compat.v1.placeholder(tf.int32, [None, None], name="input_x_word")
        self.input_x_len = tf.compat.v1.placeholder(tf.int32, name='input_x_len')
        self.input_mask = tf.compat.v1.placeholder(tf.int32, [None, None], name='input_mask')
        self.keep_prob = tf.compat.v1.placeholder(tf.float32, name='dropout_keep_prob')
        self.input_relation = tf.compat.v1.placeholder(tf.int32, [None, None], name='input_relation')  # 实体NER的真实标签
        self.is_training = tf.compat.v1.placeholder(tf.bool, None, name='is_training')
        # self.global_step = tf.get_variable('step', [], initializer=0, trainable=False)
        self.global_step = tf.Variable(0, name='step', trainable=False)

        # BERT Embedding
        self.word_embedding = self.init_embedding()

        # 超参数设置
        self.learning_rate = self.config.learning_rate
        self.embed_learning_rate = self.config.embed_learning_rate
        self.relation_num = self.config.relation_num
        self.initializer = initializers.xavier_initializer()
        self.lstm_dim = self.config.lstm_dim
        # self.embed_dense_dim = self.config.embed_dense_dim
        self.dropout = self.config.dropout

        # CRF超参数
        used = tf.sign(tf.abs(self.input_x_word))
        length = tf.reduce_sum(used, reduction_indices=1)
        self.lengths = tf.cast(length, tf.int32)
        self.batch_size = tf.shape(self.input_x_word)[0]
        self.num_steps = tf.shape(self.input_x_word)[-1]
        lstm_inputs = tf.nn.dropout(self.word_embedding, self.dropout)
        lstm_outputs = self.bilstm_layer(lstm_inputs, self.lstm_dim, self.lengths)
        self.logits = self.project_layer(lstm_outputs)

        self.trans = tf.compat.v1.get_variable(
            name="transitions",
            shape=[self.relation_num + 1, self.relation_num + 1],  # 1
            initializer=self.initializer)

        # 计算损失
        self.loss = self.loss_layer(self.logits, self.lengths)

        # bert模型参数初始化的地方
        init_checkpoint = self.config.bert_file
        # 获取模型中所有的训练参数。
        tvars = tf.trainable_variables()
        if self.config.model_type == "XLNET":
            mark = 'transformer'

        else:
            mark = 'bert'
        bert_variables = [x for x in tvars if mark in x.name]  # BERT的参数
        normal_variables = [x for x in tvars if mark not in x.name]  # 下接结构的参数
        print('bert train variable num: {}'.format(len(bert_variables)))
        print('normal train variable num: {}'.format(len(normal_variables)))

        # 加载BERT模型
        (assignment_map, initialized_variable_names) = get_assignment_map_from_checkpoint(tvars, init_checkpoint)
        tf.compat.v1.train.init_from_checkpoint(init_checkpoint, assignment_map)
        print("**** Trainable Variables ****")
        # 打印加载模型的参数
        train_vars = []
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            else:
                train_vars.append(var)
            print("  name = {}, shape = {}{}".format(var.name, var.shape, init_string))

        # memory limited ==> sequence_length=64 train_batch_size=5
        with tf.variable_scope("optimizer"):
            normal_optimizer = tf.compat.v1.train.AdamOptimizer(self.learning_rate)  # 下接结构的学习率
            normal_op = normal_optimizer.minimize(self.loss, global_step=self.global_step, var_list=normal_variables)
            num_train_steps = int(self.config.num_records / self.config.train_batch_size * self.config.train_epoch)
            if bert_variables:  # 对BERT微调
                print('word2vec trainable!!')
                word2vec_op, self.embed_learning_rate = \
                    create_optimizer(
                        self.loss, self.embed_learning_rate,
                        num_train_steps=num_train_steps,
                        num_warmup_steps=int(num_train_steps * self.config.warmup_proportion),
                        use_tpu=False, var_list=bert_variables
                    )

                self.train_op = tf.group(normal_op, word2vec_op)  # 组装BERT与下接结构参数
            else:
                self.train_op = normal_op

        self.learning_rate = tf.train.exponential_decay(self.learning_rate, self.global_step, self.config.decay_step,
                                                        self.config.decay_rate, staircase=True)
        # saver of the model
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=self.config.num_checkpoints)

    def bilstm_layer(self, lstm_inputs, lstm_dim, lengths, name=None):
        """
        :param lstm_inputs: [batch_size, num_steps, emb_size]
        :param lstm_dim:
        :param lengths:
        :param name:
        :return: [batch_size, num_steps, 2*lstm_dim]
        """
        with tf.name_scope("char_BiLSTM" if not name else name):
            lstm_cell = {}
            for direction in ["forward", "backward"]:
                with tf.name_scope(direction):
                    lstm_cell[direction] = rnn.CoupledInputForgetGateLSTMCell(
                        lstm_dim,
                        use_peepholes=True,
                        initializer=self.initializer,
                        state_is_tuple=True)

            outputs, final_states = tf.nn.bidirectional_dynamic_rnn(
                lstm_cell["forward"],
                lstm_cell["backward"],
                lstm_inputs,
                dtype=tf.float32,
                sequence_length=lengths)
        return tf.concat(outputs, axis=2)

    def project_layer(self, lstm_outputs, name=None):
        """
        hidden layer between lstm layer and logits
        :param lstm_outputs: [batch_size, num_steps, emb_size]
        :param name:
        :return: [batch_size, num_steps, num_tags]
        """
        with tf.name_scope("project" if not name else name):
            with tf.name_scope("hidden"):
                w = tf.get_variable("HW", shape=[self.lstm_dim * 2, self.lstm_dim],
                                    dtype=tf.float32, initializer=self.initializer)

                b = tf.get_variable("Hb", shape=[self.lstm_dim], dtype=tf.float32,
                                    initializer=tf.zeros_initializer())
                output = tf.reshape(lstm_outputs, shape=[-1, self.lstm_dim * 2])
                hidden = tf.tanh(tf.nn.xw_plus_b(output, w, b))

            # project to score of ori_tags.txt
            with tf.name_scope("logits"):
                w = tf.get_variable("LW", shape=[self.lstm_dim, self.relation_num],
                                    dtype=tf.float32, initializer=self.initializer)

                b = tf.get_variable("Lb", shape=[self.relation_num], dtype=tf.float32,
                                    initializer=tf.zeros_initializer())

                pred = tf.nn.xw_plus_b(hidden, w, b)

            return tf.reshape(pred, [-1, self.num_steps, self.relation_num], name='pred_logits')

    def loss_layer(self, project_logits, lengths, name=None):
        """
        计算CRF的loss
        :param project_logits: [1, num_steps, num_tags]
        :param lengths:
        :param name:
        :return: scalar loss
        """

        with tf.name_scope("crf_loss" if not name else name):
            small = -1000.0
            # pad logits for crf loss
            start_logits = tf.concat(
                [small * tf.ones(shape=[self.batch_size, 1, self.relation_num]),
                 tf.zeros(shape=[self.batch_size, 1, 1])],
                axis=-1)
            pad_logits = tf.cast(small * tf.ones([self.batch_size, self.num_steps, 1]), tf.float32)
            logits = tf.concat([project_logits, pad_logits], axis=-1)
            logits = tf.concat([start_logits, logits], axis=1)
            targets = tf.concat(
                [tf.cast(self.relation_num * tf.ones([self.batch_size, 1]), tf.int32), self.input_relation], axis=-1)

            log_likelihood, self.trans = crf_log_likelihood(
                inputs=logits,
                tag_indices=targets,
                transition_params=self.trans,
                sequence_lengths=lengths + 1
            )
            return tf.reduce_mean(-log_likelihood, name='loss')

    def init_embedding(self):
        """
        对BERT的Embedding降维
        :return:
        """
        with tf.name_scope('embedding'):
            word_embedding = self.bert_embed()
            print('self.embed_dense_dim:', self.config.embed_dense_dim)
            word_embedding = tf.layers.dense(word_embedding, self.config.embed_dense_dim, activation=tf.nn.relu)
        print("word_embedding.shape:" + str(word_embedding.shape))

        return word_embedding

    def bert_embed(self):
        """
        读取BERT的TF模型
        :return:
        """
        if self.config.model_type == "ALBERT":
            from tf_utils.albert_modeling import BertModel, BertConfig
        elif self.config.model_type == "BERT":
            from tf_utils.bert_modeling import BertModel, BertConfig
        else:
            import tf_utils.xlnet_modeling as xlnet
            xlnet_config = xlnet.XLNetConfig(json_path=self.config.xlnet_config_file)
            run_config = xlnet.create_run_config(self.is_training, True, self.config)

            xlnet_model = xlnet.XLNetModel(
                xlnet_config=xlnet_config,
                run_config=run_config,
                input_ids=self.input_x_word,
                seg_ids=None,
                input_mask=self.input_mask)
            final_hidden_states = xlnet_model.get_sequence_output()
            self.config.embed_dense_dim = 512
            return final_hidden_states

        bert_config_file = self.config.bert_config_file
        bert_config = BertConfig.from_json_file(bert_config_file)
        # batch_size, max_seq_length = get_shape_list(self.input_x_word)
        # bert_mask = tf.pad(self.input_mask, [[0, 0], [2, 0]], constant_values=1)  # tensor左边填充2列
        model = BertModel(
            config=bert_config,
            is_training=self.is_training,  # 微调
            input_ids=self.input_x_word,
            input_mask=self.input_mask,
            token_type_ids=None,
            use_one_hot_embeddings=False
        )

        if self.config.use_origin_bert:
            final_hidden_states = model.get_sequence_output()  # 原生bert
            self.config.embed_dense_dim = 1024
        else:
            layer_logits = []
            for i, layer in enumerate(model.all_encoder_layers):
                layer_logits.append(
                    tf.layers.dense(
                        layer, 1,
                        kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                        name="layer_logit%d" % i
                    )
                )

            layer_logits = tf.concat(layer_logits, axis=2)  # 第三维度拼接
            layer_dist = tf.nn.softmax(layer_logits)
            seq_out = tf.concat([tf.expand_dims(x, axis=2) for x in model.all_encoder_layers], axis=2)
            pooled_output = tf.matmul(tf.expand_dims(layer_dist, axis=2), seq_out)
            pooled_output = tf.squeeze(pooled_output, axis=2)
            char_bert_outputs = pooled_output
            final_hidden_states = char_bert_outputs  # 多层融合bert
            self.config.embed_dense_dim = 512

        return final_hidden_states
