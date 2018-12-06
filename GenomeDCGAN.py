import tensorflow as tf 
import numpy as np 
import matplotlib.pyplot as plt 
import DataUtil as dataUtil
import SequenceUtil as seqUtil
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

seqFileNames = ["Resultados/Criadas/Extraidas/NUniforme.txt", "Resultados/Criadas/Extraidas/Uniforme.txt", "Resultados/Criadas/Extraidas/Aleatoria1.txt", "Resultados/Criadas/Extraidas/Aleatoria2.txt", "Resultados/Criadas/Extraidas/Aleatoria3.txt", "Resultados/Criadas/Extraidas/Aleatoria4.txt", "Resultados/Criadas/Extraidas/Aleatoria5.txt"]
fastaFileNames = ["Resultados/Gerados/NUniforme.fasta", "Resultados/Gerados/Uniforme.fasta", "Resultados/Gerados/Aleatoria1.fasta", "Resultados/Gerados/Aleatoria2.fasta", "Resultados/Gerados/Aleatoria3.fasta", "Resultados/Gerados/Aleatoria4.fasta", "Resultados/Gerados/Aleatoria5_Evolucao_10.fasta"]
indice = 6

#Carregamento do Dataset de sequências
seqFileName = seqFileNames[indice]
fileGenomeSaveName = fastaFileNames[indice]
#Nome das Sequências
genomeName = "Markov"

#Quantidade de amostras a serem processadas, caso não se deseje processar o dataset inteiro
dataAmount = 2000

#Carregamento do dataset
genomeData = dataUtil._dataset_load(seqFileName, False, 10, 2, dataAmount)
#Tamanho das sequências
gen_length = seqUtil._sequence_size(seqFileName)
#Quantidade de sequências - Obtem automatiamente
gen_qtd = seqUtil._sequence_quantity(seqFileName) if dataAmount == 0 else dataAmount
#Tamanho do batch
batch_size = 1
#Quantidade de amostras a serem testadas
generationAmount = 0
#Controle de ruído de geração
n_noise = 1000
#Distribuição do ruído de geração (True = normal, False = uniforme)
randomDistribution = True

generatedGenomeList = []

#Carregamento dos genoas e criação do Dataset
batched_genomeData = genomeData.batch(batch_size)
iterator = batched_genomeData.make_one_shot_iterator()
next_element = iterator.get_next()

#Variaveis para alimentação da rede (Genoma e ruído)
gen_in = tf.placeholder(
    dtype = tf.float32,
    shape = [None, 4, gen_length],
    name = "Gen"
)
noise = tf.placeholder(
    dtype = tf.float32,
    shape = [None, n_noise]
)

#Variáveis para manutenção da rede
keep_prob = tf.placeholder(
    dtype = tf.float32,
    name = 'keep_prob'
)
is_training = tf.placeholder(
    dtype = tf.bool,
    name = 'is_training'
)

#Função de ativação
def _lrelu(x):
    return tf.maximum(x, tf.multiply(x, 0.2))

#Cross Entropy para cálculo de perdas
def _binary_cross_entropy(x, z):
    eps = 1e-12
    return (-(x * tf.log(z + eps) + (1. - x) * tf.log(1. - z + eps)))

#Definição do Discriminador
def _discriminator(gen_in, reuse = None, keep_prob = keep_prob):
    activation = _lrelu
    with tf.variable_scope("discriminator", reuse = reuse):
        ##print("Shape inicial: " + str(tf.shape(gen_in)))

        x = tf.reshape(gen_in, shape = [-1, 4, gen_length, 1])
        ##print("________________________________Discriminator input Shape: " + str(tf.shape(x)))
        #Convolução (8x128x2) - Dropout
        x = tf.layers.conv2d(inputs = x, kernel_size = 2, filters = 128, strides = 2, padding = 'same', activation = activation)
        x = tf.layers.dropout(inputs = x, rate = keep_prob)
        
        #Convolução (8x128x1) - Dropout
        x = tf.layers.conv2d(inputs = x, kernel_size = 2, filters = 128, strides = 1, padding = 'same', activation = activation)
        x = tf.layers.dropout(inputs = x, rate = keep_prob)
        
        #Convolução (8x128x1) - Dropout
        x = tf.layers.conv2d(inputs = x, kernel_size = 2, filters = 128, strides = 1, padding = 'same', activation = activation)
        x = tf.layers.dropout(inputs = x, rate = keep_prob)

        #Camada densa 512 neurônios
        x = tf.layers.dense(inputs = x, units = 512, activation = activation)
        #Camada densa de classificação
        x = tf.layers.dense(inputs = x, units = 1, activation = tf.nn.sigmoid)
        print("________________________________Discriminator output Shape: " + str(tf.shape(x)))

        return x

#Definição do Gerador
def _generator(noise_in, keep_prob = keep_prob, is_training = is_training):
    activation = _lrelu
    momentum = 0.1
    with tf.variable_scope("generator", reuse = None):
        x = noise_in

        x = tf.layers.dense(inputs = x, units = gen_length, activation = activation)
        x = tf.layers.dropout(inputs = x, rate = keep_prob)
        x = tf.layers.batch_normalization(inputs = x, training = is_training, momentum = momentum)

        x = tf.reshape(x, shape = [-1, 4, gen_length, 1])
        x = tf.layers.conv2d_transpose(inputs = x, kernel_size = 4, filters = 64, strides = 1, padding = 'same', activation = activation)
        x = tf.layers.dropout(inputs = x, rate = keep_prob)

        x = tf.layers.batch_normalization(inputs = x, training = is_training, momentum = momentum)
        x = tf.layers.conv2d_transpose(inputs = x, kernel_size = 8, filters = 128, strides = 1, padding = 'same',activation = activation)
        x = tf.layers.dropout(inputs = x, rate = keep_prob)

        x = tf.layers.batch_normalization(inputs = x, training = is_training, momentum = momentum)
        x = tf.layers.conv2d_transpose(inputs = x, kernel_size = 8, filters = 128, strides = 1, padding = 'same', activation = activation)
        x = tf.layers.dropout(inputs = x, rate = keep_prob)

        x = tf.layers.batch_normalization(inputs = x, training = is_training,momentum = momentum)
        x = tf.layers.conv2d_transpose(inputs = x, kernel_size = 8, filters = 1, strides = 1, padding = 'same', activation = tf.nn.sigmoid)
        x = tf.reshape(x, shape = [-1, 4, gen_length])

        print("________________________________Generator output Shape: " + str(tf.shape(x)))

        return x

g = _generator(noise, keep_prob, is_training)
d_real = _discriminator(gen_in)
d_fake = _discriminator(g, reuse = True)

vars_g = [var for var in tf.trainable_variables() if var.name.startswith("generator")]
vars_d = [var for var in tf.trainable_variables() if var.name.startswith("discriminator")]


d_reg = tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(1e-6), vars_d)
g_reg = tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(1e-6), vars_g)

loss_d_real = _binary_cross_entropy(tf.ones_like(d_real), d_real)
loss_d_fake = _binary_cross_entropy(tf.zeros_like(d_fake), d_fake)
loss_g = tf.reduce_mean(_binary_cross_entropy(tf.ones_like(d_fake), d_fake))
loss_d = tf.reduce_mean(0.5 * (loss_d_real + loss_d_fake))

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    optimizer_d = tf.train.RMSPropOptimizer(learning_rate=0.00015).minimize(loss_d + d_reg, var_list=vars_d)
    optimizer_g = tf.train.RMSPropOptimizer(learning_rate=0.00015).minimize(loss_g + g_reg, var_list=vars_g)

#Criação da Seção do Tensorflow
sess = tf.Session()
sess.run(tf.global_variables_initializer())

#Treinamento da Rede
for i in range(0, int(gen_qtd/batch_size)):
    train_d = True
    train_g = True
    keep_prob_train = 0.6

    #Criação do Ruído
    n = np.random.normal(0, 1, [4, n_noise]).astype(np.float32) if randomDistribution else np.random.uniform(0.0, 1, [5, 11200]).astype(np.float32)
    
    batch = np.array(sess.run(next_element))

    d_real_ls, d_fake_ls, g_ls, d_ls = sess.run(
        [loss_d_real, loss_d_fake, loss_g, loss_d], 
        feed_dict={gen_in: batch, noise: n, keep_prob: keep_prob_train, is_training:True}
    )

    d_real_ls = np.mean(d_real_ls)
    d_fake_ls = np.mean(d_fake_ls)
    g_ls = g_ls
    d_ls = d_ls

    if g_ls * 1.5 < d_ls:
        train_g = False
        pass
    if d_ls * 2 < g_ls:
        train_d = False
        pass

    if train_d:
        sess.run(optimizer_d, feed_dict={noise: n, gen_in: batch, keep_prob: keep_prob_train, is_training:True})
           
    if train_g:
        sess.run(optimizer_g, feed_dict={noise: n, keep_prob: keep_prob_train, is_training:True})
    
    if not i % 100:
        print (i*batch_size, d_ls, g_ls, d_real_ls, d_fake_ls)
        if not train_g:
            print("not training generator")
        if not train_d:
            print("not training discriminator")
        gen_genome = sess.run(g, feed_dict = {noise: n, keep_prob: 1.0, is_training:False})
        generatedGenomeList.append(np.squeeze(gen_genome))

    #Geração das Sequências
    if i >= gen_qtd - generationAmount:
        print(i)
        gen_genome = sess.run(g, feed_dict = {noise: n, keep_prob: 1.0, is_training:False})
        generatedGenomeList.append(np.squeeze(gen_genome))

seqUtil._sequence_list_saving(list(map(dataUtil._sequence_matrice_remapping, generatedGenomeList)), fileGenomeSaveName, genomeName, "w")