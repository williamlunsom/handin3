import itertools

def read_fasta_file(filename):
    """
    Reads the given FASTA file f and returns a dictionary of sequences.

    Lines starting with ';' in the FASTA file are ignored.
    """
    sequences_lines = {}
    current_sequence_lines = None
    with open(filename) as fp:
        for line in fp:
            line = line.strip()
            if line.startswith(';') or not line:
                continue
            if line.startswith('>'):
                sequence_name = line.lstrip('>')
                current_sequence_lines = []
                sequences_lines[sequence_name] = current_sequence_lines
            else:
                if current_sequence_lines is not None:
                    current_sequence_lines.append(line)
    sequences = {}
    for name, lines in sequences_lines.items():
        sequences[name] = ''.join(lines)
    return sequences










class codon_dicts:
    def __init__(self):
        self.forward_start_dict = {'atg': 1,'atc': 2,'ata': 3,'att': 4,'gtg': 5,'gtt': 6,'ttg': 7}
        self.forward_stop_dict = {'tag': 72,'taa': 73,'tga': 74}
        self.bacward_start_dict = {'cat': 142,'aat': 143,'cac': 144,'caa': 145,'tat': 146,'cag': 147,'gat': 148}
        self.bacward_stop_dict = {'cta': 75,'tta': 76,'tca': 77}

        letters = ['a','c','g','t']

        combinations = []
        for combo in itertools.product(letters, repeat=3):
            combinations.append(''.join(combo))

        forward_indices = range(8,len(combinations)+8)
        backward_indices = range(78,len(combinations)+78)
        self.forward_combo_dict = dict(zip(combinations, forward_indices))
        self.backward_combo_dict = dict(zip(combinations, backward_indices))



#x = 'TGAGTATCACTTAGGTCTATGTCTAGTCGTCTTTCGTAATGTTTGGTCTTGTCACCAGTTATCCTATGGCGCTCCGAGTCTGGTTCTCGAAATAAGCATCCCCGCCCAAGTCATGCACCCGTTTGTGTTCTTCGCCGACTTGAGCGACTTAATGAGGATGCCACTCGTCACCATCTTGAACATGCCACCAACGAGGTTGCCGCCGTCCATTATAACTACAACCTAGACAATTTTCGCTTTAGGTCCATTCACTAGGCCGAAATCCGCTGGAGTAAGCACAAAGCTCGTATAGGCAAAACCGACTCCATGAGTCTGCCTCCCGACCATTCCCATCAAAATACGCTATCAATACTAAAAAAATGACGGTTCAGCCTCACCCGGATGCTCGAGACAGCACACGGACATGATAGCGAACGTGACCAGTGTAGTGGCCCAGGGGAACCGCCGCGCCATTTTGTTCATGGCCCCGCTGCCGAATATTTCGATCCCAGCTAGAGTAATGACCTGTAGCTTAAACCCACTTTTGGCCCAAACTAGAGCAACAATCGGAATGGCTGAAGTGAATGCCGGCATGCCCTCAGCTCTAAGCGCCTCGATCGCAGTAATGACCGTCTTAACATTAGCTCTCAACGCTATGCAGTGGCTTTGGTGTCGCTTACTACCAGTTCCGAACGTCTCGGGGGTCTTGATGCAGCGCACCACGATGCCAAGCCACGCTGAATCGGGCAGCCAGCAGGATCGTTACAGTCGAGCCCACGGCAATGCGAGCCGTCACGTTGCCGAATATGCACTGCGGGACTACGGACGCAGGGCCGCCAACCATCTGGTTGACGATAGCCAAACACGGTCCAGAGGTGCCCCATCTCGGTTATTTGGATCGTAATTTTTGTGAAGAACACTGCAAACGCAAGTGGCTTTCCAGACTTTACGACTATGTGCCATCATTTAAGGCTACGACCCGGCTTTTAAGACCCCCACCACTAAATAGAGGTACATCTGA'
#x = 
#z_long = '3333321021021021021021021021021021021021021021021021021021021021021021033333333334564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564563210210210210210210210210210210210210210210210210210210210210210210210210210210210210210210210210210210210210210210210210210210210321021021021021021021021033334564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564563333333456456456456456456456456456456456456456456456456456456456456456456456456456456456456456456456456456456456456456456332102102102102102102102102102102102102102102102102102102102102102102102102102102102102102102102103210210210210210210210210210210210210210210210210210210210210210'
#z      = 'NNNNNCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCNNNNNNNNNNRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRNCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCNCCCCCCCCCCCCCCCCCCCCCCCCNNNNRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRNNNNNNNRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRNNCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCNCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC'

g1 = read_fasta_file('./data-handin3/genome1.fa')
true_ann1 = read_fasta_file('./data-handin3/true-ann1.fa')

end_point = 50000

ann = true_ann1['true-ann1'][:end_point]
x = g1['genome1'][:end_point]

def translate_annotation_to_indices_codon(ann, x):
    codon_dicts = codon_dicts()
    i = 0
    x = x.lower()
    res = []
    while i < len(ann)-2:
        #check for non-coding:
        if ann[i] == "N":
            res.append(0)
            i += 1
        elif ann[i] == "C":
            #check for start of forward coding sequence:
            if (ann[i-1] == "N") or (ann[i-1] == "R"):
                res.append(codon_dicts.forward_start_dict[x[i:i+3]])
                i += 3
            #check for stop of forward coding sequence:
            elif (ann[i+1] == "N") or (ann[i+1] == "R"):
                res.append(codon_dicts.forward_stop_dict[x[i:i+3]])
                i += 3
            else:
                res.append(codon_dicts.forward_combo_dict[x[i:i+3]])
                i += 3
        else:
            #check for stop of backward coding sequence
            if (ann[i-1] == "N") or (ann[i-1] == "C"):
                res.append(codon_dicts.bacward_stop_dict[x[i:i+3]])
                i += 3
            #check for start of backward coding sequence
            elif (ann[i+1] == "N") or (ann[i+1] == "C"):
                res.append(codon_dicts.bacward_start_dict[x[i:i+3]])
                i += 3
            else:
                res.append(codon_dicts.backward_combo_dict[x[i:i+3]])
                i += 3











# i = 0
# res = ""
# while i < len(z_long):
#     if z_long[i] == '3':
#         res += "N"
#         i += 1
#     elif z_long[i] == '2':
#         res += "CCC"
#         i += 3
#     else:
#         res += "RRR"
#         i += 3
# print(res)
