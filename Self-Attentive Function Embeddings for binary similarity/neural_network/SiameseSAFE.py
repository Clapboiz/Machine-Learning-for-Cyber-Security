import tensorflow as tf

class SiameseSelfAttentive:

    def __init__(self,
                 rnn_state_size,  # Dimension of the RNN State
                 learning_rate,  # Learning rate
                 l2_reg_lambda,
                 batch_size,
                 max_instructions,
                 embedding_matrix,  # Matrix containg the embeddings for each asm instruction
                 trainable_embeddings,
                 # if this value is True, the embeddings of the asm instruction are modified by the training.
                 attention_hops,  # attention hops parameter r of [1]
                 attention_depth,  # attention detph parameter d_a of [1]
                 dense_layer_size,  # parameter e of [1]
                 embedding_size,  # size of the final function embedding, in our test this is twice the rnn_state_size
                 ):
        self.rnn_depth = 1  # if this value is modified then the RNN becames a multilayer network. In our tests we fix it to 1 feel free to be adventurous.
        self.learning_rate = learning_rate
        self.l2_reg_lambda = l2_reg_lambda
        self.rnn_state_size = rnn_state_size
        self.batch_size = batch_size
        self.max_instructions = max_instructions
        self.embedding_matrix = embedding_matrix
        self.trainable_embeddings = trainable_embeddings
        self.attention_hops = attention_hops
        self.attention_depth = attention_depth
        self.dense_layer_size = dense_layer_size
        self.embedding_size = embedding_size
        self.instructions_embeddings_t = tf.Variable(self.embedding_matrix, trainable=self.trainable_embeddings, name="instructions_embeddings")
        self.generate_new_safe()

    def restore_model(self, old_session):
        graph = old_session.graph

        self.x_1 = graph.get_tensor_by_name("x_1:0")
        self.x_2 = graph.get_tensor_by_name("x_2:0")
        self.len_1 = graph.get_tensor_by_name("lengths_1:0")
        self.len_2 = graph.get_tensor_by_name("lengths_2:0")
        self.y = graph.get_tensor_by_name('y_:0')
        self.cos_similarity = graph.get_tensor_by_name("siamese_layer/cosSimilarity:0")
        self.loss = graph.get_tensor_by_name("Loss/loss:0")
        self.train_step = graph.get_operation_by_name("Train_Step/Adam")

        return

    def self_attentive_network(self, input_x, lengths):
        # Embedding lookup
        embedded_functions = tf.nn.embedding_lookup(self.instructions_embeddings_t, input_x)

        # Creating the Bi-GRU layer
        rnn_layer = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(self.rnn_state_size, return_sequences=True))
        H = rnn_layer(embedded_functions, mask=tf.sequence_mask(lengths))

        # Tile the weight matrices for the batch
        ws1_tiled = tf.tile(tf.expand_dims(self.WS1, 0), [tf.shape(H)[0], 1, 1], name="WS1_tiled")
        ws2_tiled = tf.tile(tf.expand_dims(self.WS2, 0), [tf.shape(H)[0], 1, 1], name="WS2_tiled")

        # Compute the attention matrix
        A = tf.nn.softmax(tf.matmul(ws2_tiled, tf.nn.tanh(tf.matmul(ws1_tiled, tf.transpose(H, perm=[0, 2, 1])))), name="Attention_Matrix")
        self.A = A  # Assigning the attention matrix to self.A
        # Embedding matrix M
        M = tf.matmul(A, H, name="Attention_Embedding")

        # Flatten the matrix M
        flattened_M = tf.reshape(M, [tf.shape(M)[0], self.attention_hops * self.rnn_state_size * 2])

        return flattened_M
    
    def generate_new_safe(self):
        self.trainable_variables = tf.compat.v1.trainable_variables()
        # Sử dụng tf.compat.v1.placeholder thay vì tf.placeholder
        self.x_1 = tf.compat.v1.placeholder(tf.int32, [None, self.max_instructions], name="x_1")
        self.lengths_1 = tf.compat.v1.placeholder(tf.int32, [None], name='lengths_1')
        self.x_2 = tf.compat.v1.placeholder(tf.int32, [None, self.max_instructions], name="x_2")
        self.lengths_2 = tf.compat.v1.placeholder(tf.int32, [None], name='lengths_2')
        self.y = tf.compat.v1.placeholder(tf.float32, [None], name='y_')
        # Euclidean norms; p = 2
        self.norms = []

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        with tf.name_scope('parameters_Attention'):
            self.WS1 = tf.Variable(tf.random.truncated_normal([self.attention_depth, 2 * self.rnn_state_size], stddev=0.1), name="WS1")
            self.WS2 = tf.Variable(tf.random.truncated_normal([self.attention_hops, self.attention_depth], stddev=0.1), name="WS2")

            rnn_layers_fw = [tf.keras.layers.GRUCell(size) for size in ([self.rnn_state_size] * self.rnn_depth)]
            rnn_layers_bw = [tf.keras.layers.GRUCell(size) for size in ([self.rnn_state_size] * self.rnn_depth)]

            self.cell_fw = tf.keras.layers.StackedRNNCells(rnn_layers_fw)
            self.cell_bw = tf.keras.layers.StackedRNNCells(rnn_layers_bw)

        with tf.name_scope('Self-Attentive1'):
            self.function_1 = self.self_attentive_network(self.x_1, self.lengths_1)
        with tf.name_scope('Self-Attentive2'):
            self.function_2 = self.self_attentive_network(self.x_2, self.lengths_2)

        dense_layer = tf.keras.layers.Dense(self.dense_layer_size, activation=tf.nn.relu)
        self.dense_1 = dense_layer(self.function_1)
        self.dense_2 = dense_layer(self.function_2)
        # self.dense_1 = tf.nn.relu(tf.layers.dense(self.function_1, self.dense_layer_size))
        # self.dense_2 = tf.nn.relu(tf.layers.dense(self.function_2, self.dense_layer_size))

        with tf.name_scope('Embedding1'):
                self.function_embedding_1 = tf.keras.layers.Dense(self.embedding_size)(self.dense_1)
        with tf.name_scope('Embedding2'):
            self.function_embedding_2 = tf.keras.layers.Dense(self.embedding_size)(self.dense_2)

        # Define Cosine Similarity
        with tf.name_scope('Cosine_Similarity'):
            dot_product = tf.reduce_sum(tf.multiply(self.function_embedding_1, self.function_embedding_2), axis=1)
            norm_1 = tf.sqrt(tf.reduce_sum(tf.square(self.function_embedding_1), axis=1))
            norm_2 = tf.sqrt(tf.reduce_sum(tf.square(self.function_embedding_2), axis=1))
            self.cos_similarity = dot_product / (norm_1 * norm_2)

            # CalculateMean cross-entropy loss
        with tf.name_scope("Loss"):
            A_square = tf.matmul(self.A, tf.transpose(self.A, perm=[0, 2, 1]))

            I = tf.eye(tf.shape(A_square)[1])
            I_tiled = tf.tile(tf.expand_dims(I, 0), [tf.shape(A_square)[0], 1, 1], name="I_tiled")
            self.A_pen = tf.norm(A_square - I_tiled)

            self.loss = tf.reduce_sum(tf.square(self.cos_similarity - self.y), name="loss")
            # self.loss = tf.reduce_sum(tf.squared_difference(self.cos_similarity, self.y), name="loss")
            self.regularized_loss = self.loss + self.l2_reg_lambda * l2_loss + self.A_pen

        # Train step
        with tf.name_scope("Train_Step"):
            optimizer = tf.keras.optimizers.Adam(self.learning_rate)
            # Initialize the gradient tape
            with tf.GradientTape() as tape:
                # Ensure that the operations are being recorded for automatic differentiation
                tape.watch(self.trainable_variables)
                # Call your loss computation here
                self.loss = tf.reduce_sum(tf.square(self.cos_similarity - self.y), name="loss")
                self.regularized_loss = self.loss + self.l2_reg_lambda * l2_loss + self.A_pen

            # Compute the gradients using the tape
            gradients = tape.gradient(self.regularized_loss, self.trainable_variables)
            # Apply the gradients to adjust the weights
            self.train_step = optimizer.apply_gradients(zip(gradients, self.trainable_variables))
