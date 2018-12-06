import Bio.SeqIO as biopy 
import time as time

#Extrai a sequência e a armazena em um arquivo txt, onde cada sequência esta contida em uma linha
def _sequence_extractor_saving(fileName, fastaName):
    with open(fileName, "w") as out_file:
        for record in biopy.parse(fastaName, "fasta"):
            out_file.write(str(record.seq) + "\n")

def _sequence_extractor(fastaName):
    sequenceList = []
    for record in biopy.parse(fastaName, "fasta"):
        sequenceList.append(str(record.seq))
    return sequenceList

def _sequence_saving(sequence, fileName, sequenceName, option = "w"):
    with open(fileName, option) as out_file:
        out_file.write(">" + sequenceName + " " + time.strftime("%d/%m/%Y") + " " + time.strftime("%H:%M:%S") + "\n" + ''.join(sequence) + "\n")

def _sequence_list_saving(genome_list, fileName, sequenceName, option = "w"):
    fastaText = ""
    for i, genome in enumerate(genome_list):
        fastaText += ">" + sequenceName + "_" + str(i) + " " + time.strftime("%d/%m/%Y") + " " + time.strftime("%H:%M:%S") + "\n" + ''.join(genome) + "\n" + "\n"      

    with open(fileName, option) as out_file:
        out_file.write(fastaText)

#Split em uma sequnência usando o tamanho como parâmetro
def _split_sequence(string, size):
    str_list = []
    n = len(string)
    for i in range(0, n, size):
        str_list.append(string[i:i + size])
    return str_list

#Não usar. Função mal feita, lenta e ocupa quantidaes absurdas de memória
def _parsed_sequence_extractor(fileName, fastaName):
    with open(fileName, "w") as out_file:
        data = ""
        i = 0
        for record in biopy.parse(fastaName, "fasta"):
            print("Sequencia: " + str(i) + " - " + str(record.id))
            sequence = _split_sequence(str(record.seq), 1)
            for i in range(0, len(sequence)):
                data = data + str(sequence[i]) + ", " if (i < len(sequence) - 1) else data + str(sequence[i])
                # data = data + str(i) + ", "
            out_file.write(str(data) + "\n")

#Extrai uma sequência e normaliza seu tamanho
def _sequence_extractor_completition(fileName, fastaName, nucleotide = "N", max_size = 0):
    if(max_size == 0):
        for record in biopy.parse(fastaName, "fasta"):
            max_size = max_size if len(record.seq) < max_size else len(record.seq)

    with open(fileName, "w") as out_file:
        for record in biopy.parse(fastaName, "fasta"):
            out_file.write(str(record.seq) + "N" * (max_size - len(record.seq)) + "\n")

#Retorna o tamanho da senquencia genômica normalizada
def _sequence_size(fileName):
    with open(fileName, "r") as gen_file:
        return len(gen_file.readline()) - 1

#Retorna a quantidade de sequências em um arquivo txt
def _sequence_quantity(fileName):
    lines = 0
    with open(fileName, "r") as gen_file:
        for line in gen_file:
            lines = lines + 1
    return lines
