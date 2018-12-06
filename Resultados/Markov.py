import numpy as np
from numpy import linalg as la
import tensorflow as tf
import configparser
import time as time
from Bio.Seq import Seq
import Bio.SeqIO as biopy 
import HeatMap as heatMap

def _sequence_saving(genome_list, fileName, sequenceName, option = "w"):
    fastaText = ""
    for i, genome in enumerate(genome_list):
        fastaText += ">" + sequenceName + "_" + str(i) + " " + time.strftime("%d/%m/%Y") + " " + time.strftime("%H:%M:%S") + "\n" + ''.join(genome) + "\n" + "\n"      

    with open(fileName, option) as out_file:
        out_file.write(fastaText)

def _markov_Calculator_Saving(genomeDict, fileName, matriceFileName, imageFileName, option = "w"):
    probList = np.array(list(genomeDict.values()))
    probMat = (probList.reshape((4, 4))/100)
    resMat = np.loadtxt(matriceFileName)
    diffMat = np.absolute(resMat - probMat)

    txtText = "> {} - BPTotal: X \n".format(fileName[30:-4])
    for key, val in genomeDict.items():
        txtText += "{} = {} \n".format(key, val)
    
    txtText += "\n \n ###################### Resultados ###################### \n \n"
    txtText += "\n \n Matriz Resultante: \n {} ".format(resMat)
    txtText += "\n \n Matriz Estimada: \n {} ".format(probMat)
    txtText += "\n \n Matriz de Diferenca: \n {} ".format(diffMat)
    txtText += "\n \n Norma Euclidiana: {} \n \n".format(np.linalg.norm(diffMat, 'fro'))

    with open(fileName, option) as out_file:
        out_file.write(txtText)

    heatMap._plotarHeatmap((diffMat*100), imageFileName)

def _sequence_extractor(fastaName):
    sequenceList = []
    for record in biopy.parse(fastaName, "fasta"):
        sequenceList.append(str(record.seq))
    return sequenceList

def _genome_Markov_Generator(initializationFile, size, quantity, fastaFileName, matriceFileName, genomeName = "Markov"):

    config = configparser.ConfigParser()
    config.read(initializationFile)

    #Read state chance probabilities from a ini file for easy configuration
    za = float(config['MarkovProbs']['0a'])
    zc = float(config['MarkovProbs']['0c'])
    zg = float(config['MarkovProbs']['0g'])
    zt = float(config['MarkovProbs']['0t'])
    aa = float(config['MarkovProbs']['aa'])
    ac = float(config['MarkovProbs']['ac'])
    ag = float(config['MarkovProbs']['ag'])
    at = float(config['MarkovProbs']['at'])
    ca = float(config['MarkovProbs']['ca'])
    cc = float(config['MarkovProbs']['cc'])
    cg = float(config['MarkovProbs']['cg'])
    ct = float(config['MarkovProbs']['ct'])
    ga = float(config['MarkovProbs']['ga'])
    gc = float(config['MarkovProbs']['gc'])
    gg = float(config['MarkovProbs']['gg'])
    gt = float(config['MarkovProbs']['gt'])
    ta = float(config['MarkovProbs']['ta'])
    tc = float(config['MarkovProbs']['tc'])
    tg = float(config['MarkovProbs']['tg'])
    tt = float(config['MarkovProbs']['tt'])

    probMatrice = np.array([aa, ac, ag, at, ca, cc, cg, ct, ga, gc, gg, gt, ta, tc, tg, tt])
    probMatrice = probMatrice.reshape((4, 4))
    np.savetxt(matriceFileName, probMatrice)

    genome = ''
    nucleotide = '0'
    nucleotideList = ['A', 'C', 'G', 'T']
    genome_list = []

    percentage = 0

    for i in range(quantity):
        genome = ''
        
        for j in range(size):
            if(nucleotide == '0'):
                nucleotide = np.random.choice(nucleotideList, 1, p = [za, zc, zg, zt])
            elif(nucleotide == 'A'):
                nucleotide = np.random.choice(nucleotideList, 1, p = [aa, ac, ag, at])
            elif(nucleotide == 'C'):
                nucleotide = np.random.choice(nucleotideList, 1, p = [ca, cc, cg, ct])
            elif(nucleotide == 'G'):
                nucleotide = np.random.choice(nucleotideList, 1, p = [ga, gc, gg, gt])
            elif(nucleotide == 'T'):
                nucleotide = np.random.choice(nucleotideList, 1, p = [ta, tc, tg, tt])

            genome += nucleotide[0]

        genome_list.append(genome)

        newPercentage = (((i+1)/quantity)*100)//1
        if(percentage != newPercentage):
            percentage = newPercentage
            print("Generation Progress: %i%s" % (percentage, '%'))

    #return genome_list

    _sequence_saving(genome_list, fastaFileName, "markov")

def _genome_Markov_Calculator(genome_list):
    bpDict = {}

    genomeBP = ["AA", "AC", "AG", "AT", "CA", "CC", "CG", "CT", "GA", "GC", "GG", "GT", "TA", "TC", "TG", "TT"]

    for genome in genome_list:
        for bp in genomeBP:
            bpDict[bp] = Seq(genome).count_overlap(bp) + bpDict[bp] if bp in bpDict else Seq(genome).count_overlap(bp)
    
    A = bpDict["AA"] + bpDict["AC"] + bpDict["AG"] + bpDict["AT"] 
    C = bpDict["CA"] + bpDict["CC"] + bpDict["CG"] + bpDict["CT"] 
    G = bpDict["GA"] + bpDict["GC"] + bpDict["GG"] + bpDict["GT"]
    T = bpDict["TA"] + bpDict["TC"] + bpDict["TG"] + bpDict["TT"]

    total = 0
    for bp in bpDict:
        total += bpDict[bp]
        print("%s = %i" % (bp, bpDict[bp]))

    bpDict["AA"] = (bpDict["AA"]/A) * 100
    bpDict["AC"] = (bpDict["AC"]/A) * 100
    bpDict["AG"] = (bpDict["AG"]/A) * 100
    bpDict["AT"] = (bpDict["AT"]/A) * 100
    bpDict["CA"] = (bpDict["CA"]/C) * 100
    bpDict["CC"] = (bpDict["CC"]/C) * 100
    bpDict["CG"] = (bpDict["CG"]/C) * 100
    bpDict["CT"] = (bpDict["CT"]/C) * 100
    bpDict["GA"] = (bpDict["GA"]/G) * 100
    bpDict["GC"] = (bpDict["GC"]/G) * 100
    bpDict["GG"] = (bpDict["GG"]/G) * 100
    bpDict["GT"] = (bpDict["GT"]/G) * 100
    bpDict["TA"] = (bpDict["TA"]/T) * 100
    bpDict["TC"] = (bpDict["TC"]/T) * 100
    bpDict["TG"] = (bpDict["TG"]/T) * 100
    bpDict["TT"] = (bpDict["TT"]/T) * 100

    print("Total de BP: %i" %(total))

    return bpDict

#Variaveis Gerais
names = ["NUniforme", "Uniforme", "Aleatoria1", "Aleatoria2", "Aleatoria3", "Aleatoria4", "Aleatoria5"]
names = ["Aleatoria3"]

#Variaveis Geração
iniReadingPath = "Cadeias de Markov/{}.ini"
matricesSavePath = "Cadeias de Markov Resultantes/Matriz_{}.txt" 
fastaSavePath = "Criadas/{}.fasta"

#Variaveis Estimação
fastaReadingPath = "Gerados/{}.fasta"
estimatedMarkovSavePath = "Cadeias de Markov Resultantes/{}.txt" 
matriceReadinPath = "Cadeias de Markov Resultantes/Matriz_{}.txt"
imageSavePath = "Graficos/{}"

def _geracao_Markov(names, size, quantity):
    for name in names:
        _genome_Markov_Generator(iniReadingPath.format(name), size, quantity, fastaSavePath.format(name), matricesSavePath.format(name))

def _estimacao_Markov(names):
    for name in names:
        genome_seq = _sequence_extractor(fastaReadingPath.format(name))
        _markov_Calculator_Saving(_genome_Markov_Calculator(genome_seq), estimatedMarkovSavePath.format(name), matriceReadinPath.format(name), imageSavePath.format(name))

#_geracao_Markov(names, 1000, 30000)
_estimacao_Markov(names)