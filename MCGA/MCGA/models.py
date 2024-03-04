import keras
from keras.layers import *
from keras.optimizers import *
from keras.models import Model


from Hypers import *

class Transoformer(Layer):

    def __init__(self, nb_head, size_per_head, **kwargs):
        self.nb_head = nb_head
        self.size_per_head = size_per_head
        self.output_dim = nb_head * size_per_head
        super(Transoformer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.WQ = self.add_weight(name='WQ',
                                  shape=(input_shape[0][-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        self.WK = self.add_weight(name='WK',
                                  shape=(input_shape[1][-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        self.WV = self.add_weight(name='WV',
                                  shape=(input_shape[2][-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        super(Transoformer, self).build(input_shape)

    def Mask(self, inputs, seq_len, mode='mul'):
        if seq_len == None:
            return inputs
        else:
            mask = K.one_hot(seq_len[:, 0], K.shape(inputs)[1])
            mask = 1 - K.cumsum(mask, 1)  # 沿第二维累计求和
            for _ in range(len(inputs.shape) - 2):
                mask = K.expand_dims(mask, 2)
            if mode == 'mul':
                return inputs * mask
            if mode == 'add':
                return inputs - (1 - mask) * 1e12

    def call(self, x):
        if len(x) == 3:
            Q_seq, K_seq, V_seq = x
            Q_len, V_len = None, None
        elif len(x) == 5:
            Q_seq, K_seq, V_seq, Q_len, V_len = x
        Q_seq = K.dot(Q_seq, self.WQ)  # shape=(?,30,400)
        Q_seq = K.reshape(Q_seq, (-1, K.shape(Q_seq)[1], self.nb_head, self.size_per_head))
        Q_seq = K.permute_dimensions(Q_seq, (0, 2, 1, 3))

        K_seq = K.dot(K_seq, self.WK)
        K_seq = K.reshape(K_seq, (-1, K.shape(K_seq)[1], self.nb_head, self.size_per_head))
        K_seq = K.permute_dimensions(K_seq, (0, 2, 1, 3))
        V_seq = K.dot(V_seq, self.WV)
        V_seq = K.reshape(V_seq, (-1, K.shape(V_seq)[1], self.nb_head, self.size_per_head))
        V_seq = K.permute_dimensions(V_seq, (0, 2, 1, 3))

        Q_seq_reshape = K.reshape(Q_seq, (-1, K.shape(Q_seq)[2], K.shape(Q_seq)[3]))
        K_seq_reshape = K.reshape(K_seq, (-1, K.shape(K_seq)[2], K.shape(K_seq)[3]))
        A = K.batch_dot(Q_seq_reshape, K_seq_reshape, axes=[2, 2]) / self.size_per_head ** 0.5
        A = K.reshape(A, (-1, K.shape(Q_seq)[1], K.shape(A)[1], K.shape(A)[2]))

        A = K.permute_dimensions(A, (0, 3, 2, 1))
        A = self.Mask(A, V_len, 'add')
        A = K.permute_dimensions(A, (0, 3, 2, 1))
        A = K.softmax(A)

        A_reshape = K.reshape(A, (-1, K.shape(A)[2], K.shape(A)[3]))
        V_seq_reshape = K.reshape(V_seq, (-1, K.shape(V_seq)[2], K.shape(V_seq)[3]))
        O_seq = K.batch_dot(A_reshape, V_seq_reshape, axes=[2, 1])
        O_seq = K.reshape(O_seq, (-1, K.shape(A)[1], K.shape(O_seq)[1], K.shape(O_seq)[2]))

        # O_seq = K.batch_dot(A, V_seq, axes=[3,2])
        O_seq = K.permute_dimensions(O_seq, (0, 2, 1, 3))
        O_seq = K.reshape(O_seq, (-1, K.shape(O_seq)[1], self.output_dim))
        O_seq = self.Mask(O_seq, Q_len, 'mul')
        return O_seq

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1], self.output_dim)


class Fastformer(Layer):

    def __init__(self, nb_head, size_per_head, **kwargs):
        self.nb_head = nb_head
        self.size_per_head = size_per_head
        self.output_dim = nb_head * size_per_head
        self.now_input_shape = None
        super(Fastformer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.now_input_shape = input_shape
        self.WQ = self.add_weight(name='WQ',
                                  shape=(input_shape[0][-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        self.WK = self.add_weight(name='WK',
                                  shape=(input_shape[1][-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        self.Wa = self.add_weight(name='Wa',
                                  shape=(self.output_dim, self.nb_head),
                                  initializer='glorot_uniform',
                                  trainable=True)
        self.Wb = self.add_weight(name='Wb',
                                  shape=(self.output_dim, self.nb_head),
                                  initializer='glorot_uniform',
                                  trainable=True)
        self.WP = self.add_weight(name='WP',
                                  shape=(self.output_dim, self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)

        super(Fastformer, self).build(input_shape)

    def Mask(self, inputs, seq_len, mode='mul'):
        if seq_len == None:
            return inputs
        else:
            mask = K.one_hot(seq_len[:, 0], K.shape(inputs)[1])
            mask = 1 - K.cumsum(mask, 1)
            for _ in range(len(inputs.shape) - 2):
                mask = K.expand_dims(mask, 2)
            if mode == 'mul':
                return inputs * mask
            if mode == 'add':
                return inputs - (1 - mask) * 1e12

    def call(self, x):
        if len(x) == 3:
            Q_seq, K_seq, V_seq = x
        elif len(x) == 5:
            Q_seq, K_seq, V_seq, Q_len, V_len = x
        Q_seq = K.dot(Q_seq, self.WQ)

        Q_seq_D = K.reshape(Q_seq, (-1, self.now_input_shape[0][1], self.nb_head * self.size_per_head))

        Q_seq_A = K.permute_dimensions(K.dot(Q_seq_D, self.Wa), (0, 2, 1)) / self.size_per_head ** 0.5
        if len(x) == 5:
            Q_seq_A = Q_seq_A - (1 - K.expand_dims(Q_len, axis=1)) * 1e8
        Q_seq_A = K.softmax(Q_seq_A)
        Q_seq = K.reshape(Q_seq, (-1, self.now_input_shape[0][1], self.nb_head, self.size_per_head))
        Q_seq = K.permute_dimensions(Q_seq, (0, 2, 1, 3))

        K_seq = K.dot(K_seq, self.WK)
        K_seq = K.reshape(K_seq, (-1, self.now_input_shape[1][1], self.nb_head, self.size_per_head))
        K_seq = K.permute_dimensions(K_seq, (0, 2, 1, 3))

        Q_seq_AO = Q_seq_A
        Q_seq_A = Lambda(lambda x: K.repeat_elements(K.expand_dims(x, axis=3), self.size_per_head, axis=3))(Q_seq_A)
        QA = K.sum(multiply([Q_seq_A, Q_seq]), axis=2)

        QA = Lambda(lambda x: K.repeat_elements(K.expand_dims(x, axis=2), self.now_input_shape[1][1], axis=2))(QA)

        QAK = multiply([K_seq, QA])
        QAK_D = K.reshape(QAK, (-1, self.now_input_shape[0][1], self.nb_head * self.size_per_head))
        QAK_A = K.permute_dimensions(K.dot(QAK_D, self.Wb), (0, 2, 1)) / self.size_per_head ** 0.5
        if len(x) == 5:
            QAK_A = QAK_A - (1 - K.expand_dims(Q_len, axis=1)) * 1e8
        QAK_A = K.softmax(QAK_A)

        QAK_AO = QAK_A
        QAK_A = Lambda(lambda x: K.repeat_elements(K.expand_dims(x, axis=3), self.size_per_head, axis=3))(QAK_A)
        QK = K.sum(multiply([QAK_A, QAK]), axis=2)

        QKS = Lambda(lambda x: K.repeat_elements(K.expand_dims(x, axis=2), self.now_input_shape[0][1], axis=2))(QK)

        QKQ = multiply([QKS, Q_seq])
        QKQ = K.permute_dimensions(QKQ, (0, 2, 1, 3))
        QKQ = K.reshape(QKQ, (-1, self.now_input_shape[0][1], self.nb_head * self.size_per_head))
        QKQ = K.dot(QKQ, self.WP)
        QKQ = K.reshape(QKQ, (-1, self.now_input_shape[0][1], self.nb_head, self.size_per_head))
        QKQ = K.permute_dimensions(QKQ, (0, 2, 1, 3))
        QKQ = QKQ + Q_seq
        QKQ = K.permute_dimensions(QKQ, (0, 2, 1, 3))
        QKQ = K.reshape(QKQ, (-1, self.now_input_shape[0][1], self.nb_head * self.size_per_head))

        return QKQ

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1], self.output_dim)


def Attention(dim1, dim2):
    vecs_input = Input(shape=(dim1, dim2), dtype='float32')
    user_vecs = Dropout(0.2)(vecs_input)
    user_att = Dense(200, activation='tanh')(user_vecs)
    user_att = keras.layers.Flatten()(Dense(1)(user_att))
    user_att = Activation('softmax')(user_att)
    user_vec = keras.layers.Dot((1, 1))([user_vecs, user_att])
    model = Model(vecs_input, user_vec)
    return model


def get_text(length, word_embedding_matrix):
    sentence_input = Input(shape=(length,), dtype='int32')
    word_embedding_layer = Embedding(word_embedding_matrix.shape[0], 300, weights=[word_embedding_matrix],
                                   trainable=True)
    word_vecs = word_embedding_layer(sentence_input)
    droped_vecs = Dropout(0.2)(word_vecs)
    word_rep = Transoformer(20, 20)([droped_vecs] * 3)
    droped_rep = Dropout(0.2)(word_rep)
    title_vec = Attention(length, 400)(droped_rep)
    sentEncodert = Model(sentence_input, title_vec)
    return sentEncodert


def get_entity(length, entity_dict):
    sentence_input = Input(shape=(length,), dtype='int32')
    word_embedding_layer = Embedding(len(entity_dict) + 1, 300, trainable=True)
    word_vecs = word_embedding_layer(sentence_input)
    droped_vecs = Dropout(0.2)(word_vecs)
    word_rep = Transoformer(5, 40)([droped_vecs] * 3)  # 5
    droped_rep = Dropout(0.2)(word_rep)
    title_vec = Attention(length, 200)(droped_rep)
    sentEncodert = Model(sentence_input, title_vec)
    return sentEncodert


def get_news(title_word_embedding_matrix, content_word_embedding_matrix, entity_dict, category_dict,
                     subcategory_dict):
    news_input = Input(shape=(MAX_TITLE + 2 + MAX_CONTENT + MAX_ENTITY,), dtype='int32')
    title_input = Lambda(lambda x: x[:, :MAX_TITLE])(news_input)
    vert_input = Lambda(lambda x: x[:, MAX_TITLE:MAX_TITLE + 1])(news_input)
    subvert_input = Lambda(lambda x: x[:, MAX_TITLE + 1:MAX_TITLE + 2])(news_input)
    content_input = Lambda(lambda x: x[:, MAX_TITLE + 2:MAX_TITLE + 2 + MAX_CONTENT])(
        news_input)
    entity_input = Lambda(lambda x: x[:, MAX_TITLE + 2 + MAX_CONTENT:])(news_input)
    title_encoder = get_text(MAX_TITLE,
                                     title_word_embedding_matrix)
    content_encoder = get_text(MAX_CONTENT,
                                       content_word_embedding_matrix)
    entity_encoder = get_entity(MAX_ENTITY, entity_dict)
    vert_embedding_layer = Embedding(len(category_dict) + 1, 128, trainable=True)
    subvert_embedding_layer = Embedding(len(subcategory_dict) + 1, 128, trainable=True)
    vert_vec = vert_embedding_layer(vert_input)
    subvert_vec = subvert_embedding_layer(subvert_input)
    vert_vec = Reshape((128,))(vert_vec)
    subvert_vec = Reshape((128,))(subvert_vec)
    vert_vec = Dense(128)(vert_vec)
    subvert_vec = Dense(128)(subvert_vec)
    vert_vec = Dropout(0.2)(vert_vec)
    subvert_vec = Dropout(0.2)(subvert_vec)
    title_vec = title_encoder(title_input)
    content_vec = content_encoder(content_input)
    entity_vec = entity_encoder(entity_input)
    vec = Concatenate(axis=-1)([title_vec, content_vec, vert_vec, subvert_vec, entity_vec])
    vec = Dense(400, activation='relu')(vec)
    sentEncodert = Model(news_input, vec)
    return sentEncodert


def news_level_representation ():
    vecs_input = Input(shape=(MAX_TITLE + 2, num3 * 40))
    title_input = Lambda(lambda x: x[:, :MAX_TITLE, :])(vecs_input)
    vert_input = Lambda(lambda x: x[:, MAX_TITLE:MAX_TITLE + 1, :])(vecs_input)
    subvert_input = Lambda(lambda x: x[:, MAX_TITLE + 1:MAX_TITLE + 2, :])(vecs_input)
    title_vec = Attention(MAX_TITLE, num3 * 40)(title_input)
    vert_vec = Reshape((num3 * 40,))(vert_input)
    subvert_vec = Reshape((num3 * 40,))(subvert_input)
    vec = Concatenate(axis=-1)([title_vec, vert_vec, subvert_vec])
    vec = Dense(40)(vec)
    return Model(vecs_input, vec)


def get_user():
    user_vecs_input = Input(shape=(MAX_CLICK, 400))
    news_vecs_input = Input(shape=(500,))
    user_vecs = Dropout(0.2)(user_vecs_input)
    news_vecs = Reshape((50, 10))(news_vecs_input)
    user_nc_vecs = Concatenate(axis=-1)([user_vecs, news_vecs])
    user_nc_vecs_dense = Dense(400)(user_nc_vecs)
    user_nc_vecs_dense = Dropout(0.2)(user_nc_vecs_dense)
    user_vecs0 = Transoformer(num4, 20)([user_nc_vecs_dense, user_nc_vecs_dense, user_nc_vecs_dense])
    user_vec = Attention(MAX_CLICK, num4 * 20)(user_vecs0)
    user_vec = Dense(370)(user_vec)
    model = Model(inputs=[user_vecs_input, news_vecs_input], outputs=user_vec)
    return model


def get_user_encoder(title_word_embedding_matrix, category_dict, subcategory_dict):
    user_vecs_input = Input(shape=(MAX_CLICK, MAX_TITLE + 2 + MAX_CONTENT + MAX_ENTITY))
    news_vecs_input = Input(shape=(500,))
    title_input = Lambda(lambda x: x[:, :, :MAX_TITLE])(user_vecs_input)
    vert_input = Lambda(lambda x: x[:, :, MAX_TITLE:MAX_TITLE + 1])(user_vecs_input)
    subvert_input = Lambda(lambda x: x[:, :, MAX_TITLE + 1:MAX_TITLE + 2])(user_vecs_input)
    word_embedding_layer = Embedding(title_word_embedding_matrix.shape[0], 300, weights=[title_word_embedding_matrix],
                                     trainable=True)
    vert_embedding_layer = Embedding(len(category_dict) + 1, 300, trainable=True)
    subvert_embedding_layer = Embedding(len(subcategory_dict) + 1, 300, trainable=True)
    news_vecs = Dense(1600)(news_vecs_input)
    news_vecs = Dropout(0.2)(news_vecs)
    title_vecs = word_embedding_layer(title_input)
    vert_vecs = vert_embedding_layer(vert_input)
    subvert_vecs = subvert_embedding_layer(subvert_input)
    user_vecs = Concatenate(axis=-2)([title_vecs, vert_vecs, subvert_vecs])
    user_vecs = Reshape((MAX_CLICK * (MAX_TITLE + 2), 300))(user_vecs)  # 1600,300
    news_vecs = Reshape((1600, 1))(news_vecs)
    user_nc_vecs = Concatenate(axis=-1)([user_vecs, news_vecs])
    user_nc_vecs = Dense(300)(user_nc_vecs)
    user_vecs = Dropout(0.2)(user_vecs)
    user_nc_vecs = Dropout(0.2)(user_nc_vecs)
    user_vecs = Fastformer(num3, 40)([user_nc_vecs, user_nc_vecs, user_nc_vecs])  # 3
    user_vecs = Dropout(0.2)(user_vecs)
    user_vecs = Reshape((MAX_CLICK, MAX_TITLE + 2, num3 * 40))(user_vecs)
    user_vecs = TimeDistributed(news_level_representation())(user_vecs)
    user_vecs = Dropout(0.2)(user_vecs)
    user_vec = Attention(MAX_CLICK, 40)(user_vecs)
    user_vec = Dense(30)(user_vec)
    user_vec = Dropout(0.2)(user_vec)
    return Model(inputs=[user_vecs_input, news_vecs_input], outputs=user_vec)


def create_model_MGCA(title_word_embedding_matrix, content_word_embedding_matrix, entity_dict, category_dict,
                      subcategory_dict):
    title_inputs = Input(shape=(MAX_TITLE + 2 + MAX_CONTENT + MAX_ENTITY,), dtype='int32')
    clicked_title_input = Input(shape=(MAX_CLICK, MAX_TITLE + 2 + MAX_CONTENT + MAX_ENTITY,),
                                dtype='int32')
    contexts_encoder = get_news(title_word_embedding_matrix, content_word_embedding_matrix,
                                        entity_dict, category_dict, subcategory_dict)
    candi_title_vec = contexts_encoder(title_inputs)
    candi_title_vec = Dropout(0.2)(candi_title_vec)
    print('candi_news_words:', candi_title_vec)
    clicked_entity_input = Input(shape=(MAX_SENTS, max_entity_num, 100))
    clicked_one_hop_input = Input(shape=(MAX_SENTS, max_entity_num, max_entity_num, 100))
    entity_inputs = Input(shape=(max_entity_num, 100), dtype='float32')
    one_hop_inputs = Input(shape=(max_entity_num, max_entity_num, 100), dtype='float32')
    clicked_onehop = keras.layers.Reshape((MAX_SENTS, max_entity_num * max_entity_num, 100))(
        clicked_one_hop_input)
    clicked_entity = keras.layers.Concatenate(axis=-2)([clicked_onehop, clicked_entity_input])
    news_onehop = keras.layers.Reshape((max_entity_num * max_entity_num, 100))(one_hop_inputs)
    news_entity = keras.layers.Concatenate(axis=-2)([news_onehop, entity_inputs, ])
    news_entity = keras.layers.Reshape((max_entity_num * (max_entity_num + 1) * 100,))(news_entity)
    news_entity = keras.layers.RepeatVector(MAX_SENTS)(news_entity)
    news_entity = keras.layers.Reshape((MAX_SENTS, max_entity_num * (max_entity_num + 1), 100))(news_entity)
    entity_emb = keras.layers.Concatenate(axis=-2)([clicked_entity, news_entity])
    pair_graph = create_pair_pair(max_entity_num)
    entity_vecs = TimeDistributed(pair_graph)(entity_emb)
    user_entity_vecs = keras.layers.Lambda(lambda x: x[:, :, :100])(entity_vecs)
    news_entity_vecs = keras.layers.Lambda(lambda x: x[:, :, 100:])(entity_vecs)
    news_entity_vecs_unreshape = news_entity_vecs
    user_entity_vecs = keras.layers.Reshape((5000,))(user_entity_vecs)
    user_entity_vecs = Dense(100)(user_entity_vecs)
    user_entity_vecs = Dropout(0.2)(user_entity_vecs)
    news_entity_vecs = Reshape((5000,))(news_entity_vecs)
    news_entity_vecs = Dense(100)(news_entity_vecs)
    news_entity_vecs = Dropout(0.2)(news_entity_vecs)
    news_vecs = keras.layers.Concatenate(axis=-1)([candi_title_vec, news_entity_vecs])

    '''usernews_entity_vecs = keras.layers.Concatenate(axis=-1)([user_entity_vecs, news_vecs])
    usernews_entity_vecs = Dense(100)(usernews_entity_vecs)
    usernews_entity_vecs = Dropout(0.2)(usernews_entity_vecs)
    print('user_entity_vecs',user_entity_vecs)
    print('news_vecs',news_vecs)
    print('usernews_entity_vecs',usernews_entity_vecs)

    user_entity_vecs = Fastformer(3, 40)([usernews_entity_vecs, user_entity_vecs, user_entity_vecs])
    print('user_entity_vecs',user_entity_vecs)


    #user_entity_vecs Tensor("dropout_348/cond/Merge:0", shape=(?, 100), dtype=float32)
    #news_vecs Tensor("concatenate_300/concat:0", shape=(?, 500), dtype=float32)
    #usernews_entity_vecs Tensor("dense_545/BiasAdd:0", shape=(?, 100), dtype=float32)
    #user_entity_vecs Tensor("fastformer_21/Reshape_15:0", shape=(?, 100, 120), dtype=float32)


    user_entity_vecs = Dropout(0.2)(user_entity_vecs)

    print('-----------------user_entity_vecs',user_entity_vecs)
    user_entity_vecs = AttentivePooling(100, 120)(user_entity_vecs)
    print('-----------------user_entity_vecs',user_entity_vecs)
    user_entity_vecs = Dense(100)(user_entity_vecs)
    user_entity_vecs = Dropout(0.2)(user_entity_vecs)
    '''


    user_encoder = get_user_encoder(title_word_embedding_matrix, category_dict, subcategory_dict)
    user_encoder1 = get_user()
    user_vecs = TimeDistributed(contexts_encoder)(clicked_title_input)
    user_vec1 = user_encoder1([user_vecs, news_vecs])
    user_vec2 = user_encoder([clicked_title_input, news_vecs])
    user_vecs_all = keras.layers.Concatenate(axis=-1)(
        [user_vec1, user_vec2, user_entity_vecs])
    news_vecs = keras.layers.Concatenate(axis=-1)([candi_title_vec, news_entity_vecs])
    score = keras.layers.Dot(axes=-1)([user_vecs_all, news_vecs])
    model = Model(
        [title_inputs, entity_inputs, one_hop_inputs, clicked_title_input, clicked_entity_input, clicked_one_hop_input],
        score)
    return model


def create_model_new(title_word_embedding_matrix, content_word_embedding_matrix, entity_dict, category_dict,
                     subcategory_dict):
    print('create_model_new')

    clicked_entity_input = Input(shape=(MAX_SENTS, max_entity_num, 100))
    clicked_one_hop_input = Input(shape=(MAX_SENTS, max_entity_num, max_entity_num, 100))
    clicked_title_input = Input(shape=(MAX_CLICK, MAX_TITLE + 2 + MAX_CONTENT + MAX_ENTITY,),
                                dtype='int32')
    title_inputs = Input(shape=(1 + npratio, MAX_TITLE + 2 + MAX_CONTENT + MAX_ENTITY,), dtype='int32')
    entity_inputs = Input(shape=(1 + npratio, max_entity_num, 100), dtype='float32')
    one_hop_inputs = Input(shape=(1 + npratio, max_entity_num, max_entity_num, 100),
                           dtype='float32')
    all_in_one_model = create_model_MGCA(title_word_embedding_matrix, content_word_embedding_matrix, entity_dict,
                                         category_dict, subcategory_dict)
    doc_score = []
    for i in range(1 + npratio):
        ti = keras.layers.Lambda(lambda x: x[:, i, :, ])(title_inputs)
        ei = keras.layers.Lambda(lambda x: x[:, i, :, :])(entity_inputs)
        eo = keras.layers.Lambda(lambda x: x[:, i, :, :, :])(one_hop_inputs)

        score = all_in_one_model([ti, ei, eo, clicked_title_input, clicked_entity_input, clicked_one_hop_input])
        score = keras.layers.Reshape((1, 1,))(score)
        doc_score.append(score)
    doc_score = keras.layers.Concatenate(axis=-2)(doc_score)
    doc_score = keras.layers.Reshape((1 + npratio,))(doc_score)
    logits = keras.layers.Activation(keras.activations.softmax, name='recommend')(doc_score)
    model = Model([title_inputs, entity_inputs, one_hop_inputs, clicked_title_input,
                   clicked_entity_input, clicked_one_hop_input], logits)
    model.compile(loss=['categorical_crossentropy'],
                  optimizer=Adam(lr=0.0001),
                  metrics=['acc'])
    return model, all_in_one_model


def create_pair_pair(max_entity_num, ):
    num = max_entity_num * (max_entity_num + 1)
    gat = get_entity_graph_encoder(max_entity_num)
    gat_fuse = Dense(100)
    entity_input = Input(shape=(num + num, 100))
    user_entity_input = keras.layers.Lambda(lambda x: x[:, :num, :])(
        entity_input)
    news_entity_input = keras.layers.Lambda(lambda x: x[:, num:, :])(entity_input)
    user_entity_zerohop = keras.layers.Lambda(lambda x: x[:, max_entity_num * max_entity_num:, :])(user_entity_input)
    user_entity_onehop = keras.layers.Lambda(lambda x: x[:, :max_entity_num * max_entity_num, :])(user_entity_input)
    user_entity_onehop = keras.layers.Reshape((max_entity_num, max_entity_num, 100))(user_entity_onehop)
    user_can = TimeDistributed(gat)(user_entity_onehop)
    user_can = keras.layers.Concatenate(axis=-1)([user_can, user_entity_zerohop])
    user_can = gat_fuse(user_can)
    user_can = keras.layers.Reshape((max_entity_num * 100,))(user_can)
    user_can = keras.layers.RepeatVector(max_entity_num)(user_can)
    user_can = keras.layers.Reshape((max_entity_num, max_entity_num, 100))(user_can)
    news_entity_zerohop = keras.layers.Lambda(lambda x: x[:, max_entity_num * max_entity_num:, :])(news_entity_input)
    news_entity_onehop = keras.layers.Lambda(lambda x: x[:, :max_entity_num * max_entity_num, :])(news_entity_input)
    news_entity_onehop = keras.layers.Reshape((max_entity_num, max_entity_num, 100))(news_entity_onehop)
    news_can = TimeDistributed(gat)(news_entity_onehop)
    news_can = keras.layers.Concatenate(axis=-1)([news_can, news_entity_zerohop])
    news_can = gat_fuse(news_can)
    news_can = keras.layers.Reshape((max_entity_num * 100,))(news_can)
    news_can = keras.layers.RepeatVector(max_entity_num)(news_can)
    news_can = keras.layers.Reshape((max_entity_num, max_entity_num, 100))(news_can)
    user_entity_onehop = keras.layers.Concatenate(axis=-2)([user_entity_onehop, news_can])
    news_entity_onehop = keras.layers.Concatenate(axis=-2)([news_entity_onehop, user_can])
    gcat = GraphCoAttNet(max_entity_num)
    user_entity_onehop = TimeDistributed(gcat)(user_entity_onehop)
    news_entity_onehop = TimeDistributed(gcat)(news_entity_onehop)
    user_entity_vecs = keras.layers.Concatenate(axis=-1)([user_entity_zerohop, user_entity_onehop])
    news_entity_vecs = keras.layers.Concatenate(axis=-1)([news_entity_zerohop, news_entity_onehop])
    Merge = Dense(100)
    user_entity_vecs = Merge(user_entity_vecs)
    news_entity_vecs = Merge(news_entity_vecs)
    user_entity_vecs = keras.layers.Concatenate(axis=-2)([user_entity_vecs, news_entity_zerohop])
    news_entity_vecs = keras.layers.Concatenate(axis=-2)([news_entity_vecs, user_entity_zerohop])
    gcat0 = GraphCoAttNet(max_entity_num)
    user_entity_vec = gcat0(user_entity_vecs)
    news_entity_vec = gcat0(news_entity_vecs)
    vec = keras.layers.Concatenate(axis=-1)([user_entity_vec, news_entity_vec])
    model = Model(entity_input, vec)
    return model


def GraphCoAttNet(num):
    entity_input = Input(shape=(num * 2, 100))
    entity_emb = keras.layers.Lambda(lambda x: x[:, :num, :])(entity_input)
    candidate_emb = keras.layers.Lambda(lambda x: x[:, num:, :])(entity_input)
    entity_vecs = Transoformer(5, 20)([entity_emb] * 3)
    entity_co_att = Dense(100)(entity_vecs)
    candidate_co_att = Dense(100)(candidate_emb)
    S = keras.layers.Dot(axes=-1)([entity_co_att, candidate_co_att])
    entity_self_att = Dense(100)(entity_vecs)
    candidate_co_att = Dense(100)(candidate_emb)
    entity_co_att = keras.layers.Dot(axes=[-1, -2])([S, candidate_emb, ])
    entity_att = keras.layers.Add()([entity_self_att, entity_co_att])
    entity_att = keras.layers.Activation('tanh')(entity_att)
    entity_att = keras.layers.Reshape((num,))(Dense(1)(entity_att))
    entity_att = keras.layers.Activation('tanh')(entity_att)
    entity_vec = keras.layers.Dot(axes=[-1, -2])([entity_att, entity_vecs])
    model = Model(entity_input, entity_vec)
    return model


def get_entity_graph_encoder(max_entity_num, ):
    entity_input = Input(shape=(max_entity_num, 100), dtype='float32')
    entity_vecs = Transoformer(2, 50)([entity_input, entity_input, entity_input])
    entity_vecs = Add()([entity_vecs, entity_input])
    entity_vecs = entity_input
    droped_rep = Dropout(0.2)(entity_vecs)
    entity_vec = Attention(max_entity_num, 100)(droped_rep)
    sentEncodert = Model(entity_input, entity_vec)
    return sentEncodert