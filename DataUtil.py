import tensorflow as tf 
import numpy as np 
import random as random
import SequenceUtil as seqUtil

#divide uma dada sequência pelo tamanho fornecido
def _split_sequence(string, size, parsed):
    str_list = []
    n = len(string)
    for i in range(0, n, size):
        if(string[i:i + size] != "\n"):
            str_list.append(string[i:i + size])
    
    return list(map(_sequence_mapping, str_list)) if parsed else str_list

def _file_save(filename, string, mode = 'w'):
    with open(filename, mode) as openfile:
        #str_seq = open_file.readlines()
        openfile.write(string)
##1 if random.random() >= 0.5 else 
def _sequence_mapping(nucleotide):
    if nucleotide in ("A", "a"):
        return 0.25
    elif nucleotide in ("C", "c"):
        return 0.5
    elif nucleotide in ("G", "g"):
        return 0.75 ##3
    elif nucleotide in ("T", "t"):
        return 1 ##4
    elif nucleotide in ("N", "n"):
        return random.choice([0.25, 0.5, 0.75, 1]) ##0
    elif nucleotide in ("W", "w"):
        return random.choice([0.25, 1]) ##5
    elif nucleotide in ("R", "r"):
        return random.choice([0.25, 0.75]) ##6
    elif nucleotide in ("Y", "y"):
        return random.choice([0.5, 1]) ##7
    elif nucleotide in ("S", "s"):
        return random.choice([0.5, 0.75]) ##8
    elif nucleotide in ("D", "d"):
        return random.choice([0.25, 0.75, 1]) ##9
    elif nucleotide in ("K", "k"):
        return random.choice([0.75, 1]) ##10
    elif nucleotide in ("B", "b"):
        return random.choice([0.5, 0.75, 1]) ##11
    elif nucleotide in ("M", "m"):
        return random.choice([0.25, 0.5]) ##12

def _sequence_remapping(code):
    if (0 <= code <= 0.25):
        return "A"
    elif (0.25 <= code <= 0.5):
        return "C"
    elif (0.5 <= code <= 0.75):
        return "G"
    elif (0.75 <= code <= 1):
        return "T"


def _sequence_matrice_mapping(code):
    if code in ("A", "a"):
        return np.array([[1.0],[0.0],[0.0],[0.0]])
    elif code in ("C", "c"):
        return np.array([[0.0],[1.0],[0.0],[0.0]])
    elif code in ("G", "g"):
        return np.array([[0.0],[0.0],[1.0],[0.0]])
    elif code in ("T", "t"):
        return np.array([[0.0],[0.0],[0.0],[1.0]])
    else:
        return np.array([[0.0],[0.0],[0.0],[0.0]])

def _sequence_matrice_remapping(sequence):
    genome = []
    
    for i in range(len(sequence[0])):
        a = (sequence[0][i], "A")
        c = (sequence[1][i], "C")
        g = (sequence[2][i], "G")
        t = (sequence[3][i], "T")

        if (a == max(a, c, g, t)):
            genome.append("A")
        elif (c == max(a, c, g, t)):
            genome.append("C")
        elif (g == max(a, c, g, t)):
            genome.append("G")
        elif (t == max(a, c, g, t)):
            genome.append("T")
    return ''.join(genome)

#Converte o genoma em uma sequência onde cada nucletideo 
#é representado por um inteiro
def _genome_parser(genome_data):
    genome = str(genome_data).split("")
    #features = {"genome": tf.VarLenFeature((),tf.string)}
    parsed_genome = []
    for i in genome:
        if i in ("A"):
            parsed_genome.append(1)
        if i in ("C"):
            parsed_genome.append(2)
        if i in ("G"):
            parsed_genome.append(3)
        if i in ("T"):
            parsed_genome.append(4)
    return parsed_genome

#Carregamento do genoma em uma lista, onde cada sequência 
#genômica é quebrada e armazenada em um item da lista
def _genome_Load(fileName, parsed, datasetSize = 0):
    # genome_seq = seqUtil._sequence_extractor(fileName)
    with open(fileName, "r") as open_file:
        genome_seq = open_file.readlines()
    
    if(datasetSize > 0):
        genome_seq = genome_seq[:datasetSize]

    genome_data = []
    for i in genome_seq:
        if(parsed == 0):
            genome_data.append(_split_sequence(i, 1, False))
        elif(parsed == 1):
            genome_data.append(_split_sequence(i, 1, True))
        elif(parsed == 2):
            partial_parsed_genome = list(map(_sequence_matrice_mapping, _split_sequence(i, 1, False)))
            holder = partial_parsed_genome[0]
            for i  in range(1, len(partial_parsed_genome)):
                holder = np.concatenate((holder, partial_parsed_genome[i]), axis = 1)
            genome_data.append(holder)
    
    return np.asarray(genome_data)


def _dataset_load(fileName, test, testNumber, parsed, datasetSize = 0):
    tensor = tf.convert_to_tensor(_genome_Load(fileName, parsed, datasetSize), dtype = tf.float32)
    dataset = tf.data.Dataset.from_tensor_slices(tensor)

    if(test == True):
        iterator = dataset.make_one_shot_iterator()
        next_element = iterator.get_next()

        with tf.Session() as sess:
            for i in range(testNumber):
                print("Genome " + str(i) + ": ")
                print(sess.run(next_element)) 
    print("Dataset: \nTipo: %s \nFormato: %s \nClasses: %s" % 
    (dataset.output_types, dataset.output_shapes, dataset.output_classes) )

    return dataset


