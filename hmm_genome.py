import math
import numpy as np
import time
import itertools


def log(x):
    if x == 0:
        return float('-inf')
    return math.log(x)


def make_table(m, n):
    """Make a table with `m` rows and `n` columns filled with zeros."""
    #return [[0] * n for _ in range(m)]
    return np.zeros([m,n])


def translate_observations_to_indices(obs):
    mapping = {'a': 0, 'c': 1, 'g': 2, 't': 3}
    return [mapping[symbol.lower()] for symbol in obs]


def translate_indices_to_observations(indices):
    mapping = ['a', 'c', 'g', 't']
    return ''.join(mapping[idx] for idx in indices)


def translate_path_to_indices(path):
#    return list(map(lambda x: int(x), path))
    return path


#def translate_indices_to_path(indices):
#    mapping = ['C', 'C', 'C', 'N', 'R', 'R', 'R']
#    return ''.join([mapping[i] for i in indices])

# def translate_indices_to_path_codon_full(indices):
#     res = ""
#     for i in indices:
#         if i == 0:
#             res += "N"
#         elif (0 < i < 75):
#             res += "C"
#         else:
#             res += "R"
#     return res

def translate_indices_to_path_codon(indices):
    res = ""
    for i in indices:
        if i == 0:
            res += "N"
        elif (0 < i < 34):
            res += "C"
        else:
            res += "R"
    return res


def count_transitions_and_emissions(K, D, x, z):
    """
    Returns a KxK matrix and a KxD matrix containing counts cf. above
    """
    transition_count = make_table(K, K)
    emission_count = make_table(K, D)
    observations_indices = translate_observations_to_indices(x)
    path_idices = translate_path_to_indices(z)
    for (index,observation) in enumerate(observations_indices):
        from_path_index = path_idices[index-1]
        to_path_index = path_idices[index]
        to_emission_index = observations_indices[index]
        if index == 0:
            emission_count[to_path_index,to_emission_index] += 1
            continue
        
        transition_count[from_path_index,to_path_index] += 1
        emission_count[from_path_index,to_emission_index] += 1
    return transition_count, emission_count


def training_by_counting(K, D, x, z, transition_count, emission_count):
    """
    Returns a HMM trained on x and z cf. training-by-counting.
    """
    observations_indices = translate_observations_to_indices(x)
    path_idices = translate_path_to_indices(z)
    init_probs = np.array([1 if path_index == path_idices[0] else 0 for path_index in range(K)])
    trans_probs = np.array([[0 if sum(transition_count[i]) == 0 else transition_count[i][j]/sum(transition_count[i]) for j in range(K)] for i in range(K)])
    emission_probs = np.array([[0 if sum(emission_count[i]) == 0 else emission_count[i][j]/sum(emission_count[i]) for j in range(D)] for i in range(K)])
    hmm_model = hmm(init_probs, trans_probs, emission_probs)
    return hmm_model


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


def compute_w_log(model, x):
    x = translate_observations_to_indices(x)
    K = len(model.init_probs)
    N = len(x)
    
    #w = make_table(k, n)
    w = np.ones((K, N))*float('-inf')
    
    # Base case: fill out w[i][0] for i = 0..k-1
    for k in range(0, K):
        w[k, 0] = log(model.init_probs[k]) + log(model.emission_probs[k, x[0]])
    
    # Inductive case: fill out w[i][j] for i = 0..k, j = 0..n-1
    t = time.time()
    i = 0
    for n in range(1, N):
        i += 1

        for k in range(0, K):
            vec = np.log(model.emission_probs[k, x[n]]) + w[:, n-1] + np.log(model.trans_probs[:, k])
            max_vec = np.maximum(w[k,n], vec)
            w[k,n] = max_vec[-1]
                
        if i > 50:
            estimate = ((time.time()-t)/51)*(N-n)
            rest = estimate % 60**2
            hours = (estimate - rest)/(60**2)
            minutes =rest/60
            print("\t{:0.2f} percent complete. Estimated {:0.0f} hours and {:0.1f} minutes remaining".format((n)/float(N)*100, hours, minutes), end="\r")
            i = 0
            t = time.time()
            
    return w


def opt_path_prob_log(w):
    return np.max(w[:, -1])


def backtrack_log(w, model, x):
    x = translate_observations_to_indices(x)
    K = len(model.init_probs)

    N = w.shape[1]
    z = np.empty((N), dtype=int)
    z[N-1] = np.argmax(w[:, -1])
    for n in range(N-2, -1, -1):
        possibilities = [log(model.emission_probs[z[n+1], x[n+1]]) + w[k, n] + log(model.trans_probs[k, z[n+1]]) for k in range(0, K)]
        z[n] = np.argmax(possibilities)
   
    return z


def compute_annotation(model, x):
    print('\tComputing w_log...')
    w = compute_w_log(model, x)
    print('\tBacktracking...')
    z = backtrack_log(w, model, x)
    return translate_indices_to_path_codon(z)


def compute_accuracy(true_ann, pred_ann):
    if len(true_ann) != len(pred_ann):
        return 0.0
    return sum(1 if true_ann[i] == pred_ann[i] else 0 
               for i in range(len(true_ann))) / len(true_ann)


#def translate_annotation_to_indices(annotation):
#    result = ""
#    i = 0
#    while i < len(annotation):
#        if annotation[i] == 'N':
#            result += '3'
#            i += 1
#        elif annotation[i] == 'C':
#            result += '210'
#            i += 3
#        else:
#            result += '456'
#            i += 3
#    return result


# def translate_annotation_to_indices_codon_full(ann, x):
#     codon_dict = codon_dicts_full()
#     i = 0
#     x = x.lower()
#     res = []
#     while i < len(ann)-1:
#         try:
#             #check for non-coding:
#             if ann[i] == "N":
#                 res.append(0)
#                 i += 1
#             elif ann[i] == "C":
#                 #check for start of forward coding sequence:
#                 if (ann[i-1] == "N") or (ann[i-1] == "R"):
#                     res.extend([codon_dict.forward_start_dict[x[i:i+3]]]*3)
#                     i += 3
#                 #check for stop of forward coding sequence:
#                 elif (ann[i+1] == "N") or (ann[i+1] == "R"):
#                     res.extend([codon_dict.forward_stop_dict[x[i:i+3]]]*3)
#                     i += 3
#                 else:
#                     #res.extend([8]*3)
#                     res.extend([codon_dict.forward_combo_dict[x[i:i+3]]]*3)
#                     i += 3
#             else:
#                 #check for stop of backward coding sequence
#                 if (ann[i-1] == "N") or (ann[i-1] == "C"):
#                     res.extend([codon_dict.bacward_stop_dict[x[i:i+3]]]*3)
#                     i += 3
#                 #check for start of backward coding sequence
#                 elif (ann[i+1] == "N") or (ann[i+1] == "C"):
#                     res.extend([codon_dict.bacward_start_dict[x[i:i+3]]]*3)
#                     i += 3
#                 else:
#                     #res.extend([78]*3)
#                     res.extend([codon_dict.backward_combo_dict[x[i:i+3]]]*3)
#                     i += 3
#         except KeyError:
#             break
#     #match length by copying last index
#     length_diff = len(ann) - len(res)
#     res.extend([res[i-1]]*length_diff)
#     assert(len(res) == len(ann))

#     return res

def translate_annotation_to_indices_codon(ann, x):
    codon_dict = codon_dicts()
    i = 0
    x = x.lower()
    res = []
    while i < len(ann)-1:
        try:
            #check for non-coding:
            if ann[i] == "N":
                res.append(0)
                i += 1
            elif ann[i] == "C":
                #check for start of forward coding sequence:
                if (ann[i-1] == "N") or (ann[i-1] == "R"):
                    res.extend([codon_dict.forward_start_dict[x[i:i+3]]])
                    i += 3
                #check for stop of forward coding sequence:
                elif (ann[i+1] == "N") or (ann[i+1] == "R"):
                    res.extend([codon_dict.forward_stop_dict[x[i:i+3]]])
                    i += 3
                else:
                    res.extend(codon_dict.forward_coding_sequence)
                    i += 3
            else:
                #check for stop of backward coding sequence
                if (ann[i-1] == "N") or (ann[i-1] == "C"):
                    res.extend([codon_dict.bacward_stop_dict[x[i:i+3]]])
                    i += 3
                #check for start of backward coding sequence
                elif (ann[i+1] == "N") or (ann[i+1] == "C"):
                    res.extend([codon_dict.bacward_start_dict[x[i:i+3]]])
                    i += 3
                else:
                    res.extend(codon_dict.backward_coding_sequence)
                    i += 3
        except KeyError:
            break
    #match length by copying last index (if necessary)
    length_diff = len(ann) - len(res)
    res.extend([res[-1]]*length_diff)
    assert(len(res) == len(ann))

    return res


class codon_dicts:
    def __init__(self):
        self.forward_start_dict = {'atg': [1,2,3],'atc': [4,5,6],'ata': [7,8,9],'att': [10,11,12],'gtg': [13,14,15],'gtt': [16,17,18],'ttg': [19,20,21]}
        self.forward_coding_sequence = [22,23,24]
        self.forward_stop_dict = {'tag': [25,26,27],'taa': [28,29,30],'tga': [31,32,33]}
        self.bacward_stop_dict = {'cta': [34,35,36],'tta': [37,38,39],'tca': [40,41,42]}
        self.backward_coding_sequence = [43,44,45]
        self.bacward_start_dict = {'cat': [46,47,48],'aat': [49,50,51],'cac': [52,53,54],'caa': [55,56,57],'tat': [58,59,60],'cag': [61,62,63],'gat': [64,65,66]}
        

# class codon_dicts_full:
#     def __init__(self):
#         self.forward_start_dict = {'atg': 1,'atc': 2,'ata': 3,'att': 4,'gtg': 5,'gtt': 6,'ttg': 7}
#         self.forward_stop_dict = {'tag': 72,'taa': 73,'tga': 74}
#         self.bacward_start_dict = {'cat': 142,'aat': 143,'cac': 144,'caa': 145,'tat': 146,'cag': 147,'gat': 148}
#         self.bacward_stop_dict = {'cta': 75,'tta': 76,'tca': 77}

#         letters = ['a','c','g','t']

#         combinations = []
#         for combo in itertools.product(letters, repeat=3):
#             combinations.append(''.join(combo))

#         forward_indices = range(8,len(combinations)+8)
#         backward_indices = range(78,len(combinations)+78)
#         self.forward_combo_dict = dict(zip(combinations, forward_indices))
#         self.backward_combo_dict = dict(zip(combinations, backward_indices))


class hmm:
    def __init__(self, init_probs, trans_probs, emission_probs):
        self.init_probs = init_probs
        self.trans_probs = trans_probs
        self.emission_probs = emission_probs


if __name__=="__main__":
    """ Main script """
    print('files...')
    g1 = read_fasta_file('./data-handin3/genome1.fa')
    g2 = read_fasta_file('./data-handin3/genome2.fa')
    true_ann1 = read_fasta_file('./data-handin3/true-ann1.fa')
    true_ann2 = read_fasta_file('./data-handin3/true-ann2.fa')

    end_point = -1
    end_point = 10000

    true_ann1_cut = true_ann1['true-ann1'][:end_point]
    true_ann2_cut = true_ann2['true-ann2'][:end_point]  
    x1 = g1['genome1'][:end_point]
    x2 = g2['genome2'][:end_point]

    print('Translating annotation to indices...')
    z1 = translate_annotation_to_indices_codon(true_ann1_cut, x1)
    z2 = translate_annotation_to_indices_codon(true_ann2_cut, x2)

    #K = 147
    K = 67
    D = 4

    print('Training...')
    transition_count1, emission_count1 = count_transitions_and_emissions(K, D, x1, z1)
    transition_count2, emission_count2 = count_transitions_and_emissions(K, D, x2, z2)
    transition_count = transition_count1 + transition_count2
    emission_count = emission_count1 + emission_count2
    hmm_state_genome1 = training_by_counting(K, D, x1, z1, transition_count, emission_count)

    print("Transition matrix:")
    print(hmm_state_genome1.trans_probs)
    print("Emission matrix: (first 10 rows)")
    print(hmm_state_genome1.emission_probs[:10,:])

    print('Computing annotations...')
    print('Genome1:')
    genome1_ann = compute_annotation(hmm_state_genome1, x1)
    print('genome2:')
    genome2_ann = compute_annotation(hmm_state_genome1, x2)

    print('Computing accuracy...')
    print("accuracy, genome1: {0}".format(compute_accuracy(true_ann1_cut, genome1_ann)))
    print("accuracy, genome2: {0}".format(compute_accuracy(true_ann2_cut, genome2_ann)))
    print('Done!')