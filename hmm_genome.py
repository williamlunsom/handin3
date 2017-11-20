import math
import numpy as np
import time
import itertools
import os

def log(x):
    if x == 0:
        return float('-inf')
    return math.log(x)


def make_table(m, n):
    """Make a table with `m` rows and `n` columns filled with zeros."""
    return np.zeros([m,n])


def translate_observations_to_indices(obs):
    mapping = {'a': 0, 'c': 1, 'g': 2, 't': 3}
    return [mapping[symbol.lower()] for symbol in obs]


def translate_indices_to_observations(indices):
    mapping = ['a', 'c', 'g', 't']
    return ''.join(mapping[idx] for idx in indices)


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
    path_idices = z
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


def training_by_counting(K, D, transition_count, emission_count):
    """
    Returns a HMM trained on x and z cf. training-by-counting.
    """
    init_probs = np.append(1, np.zeros(K-1))
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
    
    w = np.ones((K, N))*float('-inf')
    
    # Base case: fill out w[i][0] for i = 0..k-1
    for k in range(0, K):
        w[k, 0] = log(model.init_probs[k]) + log(model.emission_probs[k, x[0]])
    
    # Inductive case: fill out w[i][j] for i = 0..k, j = 0..n-1
    t = time.time()
    i = 0

    for n in range(1, N):
        i += 1
        temp_matrix = np.expand_dims(np.log(model.emission_probs[:,x[n]]), axis=0) + np.expand_dims(w[:,n-1], axis=1) + np.log(model.trans_probs)
        w[:,n] = np.max(np.stack((np.max(temp_matrix,0),w[:,n])), axis=0)
        
        if i > 1000:
            estimate = ((time.time()-t)/1001)*(N-n)
            rest = estimate % 60**2
            hours = (estimate - rest)/(60**2)
            minutes =rest/60
            print("\t{:0.2f} percent complete. Estimated {:0.0f} hours and {:0.1f} minutes remaining".format((n)/float(N)*100, hours, minutes), end="\r")
            i = 0
            t = time.time()
    print("\n")            
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


class hmm:
    def __init__(self, init_probs, trans_probs, emission_probs):
        self.init_probs = init_probs
        self.trans_probs = trans_probs
        self.emission_probs = emission_probs


def load_fasta_data(data_folder, start_file, num_files, prefix, end_index=-1):
    """
    Load all genomes and annotations.
    """
    print("Loading genomes and annotations...")
    data = []
    for i in range(start_file, start_file + num_files):
        filename = data_folder + prefix + str(i) + '.fa'
        file_data = read_fasta_file(filename)
        data_slice = file_data[prefix + str(i)][:end_index]

        data.append(data_slice)
    print("\n")

    return data


def count_multiple_transitions_and_emissions(genomes, true_anns, K, D):
    transition_counts = []
    emission_counts = []
    L = len(genomes)
    print("Counting emissions and transitions:")
    for i in range(L):
        print("\tFile {0} of {1}".format(i+1,L), end="\r")
        z = translate_annotation_to_indices_codon(true_anns[i], genomes[i])
        transition_count, emission_count = count_transitions_and_emissions(K, D, genomes[i], z)
        transition_counts.append(transition_count)
        emission_counts.append(emission_count)
    transition_counts = np.array(transition_counts)
    emission_counts = np.array(emission_counts)
    print("\n")

    return transition_counts, emission_counts


def train_cross_validation_models(transition_counts, emission_counts, K, D):
    L = len(transition_counts)
    hmm_models = []
    print("Building models for cross validation:")
    for i in range(L):
        print("\tModel {0} of {1}".format(i+1,L), end="\r")
        exlude_idx = np.arange(L) != i
        trans_counts = np.sum(transition_counts[exlude_idx],0)
        emiss_counts = np.sum(emission_counts[exlude_idx],0)
        model = training_by_counting(K, D, trans_counts, emiss_counts)
        hmm_models.append(model)
    print("\n")

    return hmm_models


def compute_cross_validation_annotations(hmm_models, genomes):
    annotations = []
    for i in range(len(genomes)):
        annotations.append(compute_annotation(hmm_models[i], genomes[i]))

    return annotations


def compute_multiple_annotations(hmm_model, genomes):
    annotations = []
    for i in range(len(genomes)):
        annotations.append(compute_annotation(hmm_model, genomes[i]))
    return annotations


def write_fasta_files(annotations, prefix, folder, start_file = 1):
    if not os.path.exists(folder):
        os.makedirs(folder)
    for i in range(len(annotations)):
        filename = "./" + folder + "/" + prefix + str(i+1) + '.fa'
        with open(filename, 'w') as f:
            f.write(">" + prefix + str(i+start_file) + "\n")
            j = 0
            while j < len(annotations[i]):
                f.write(annotations[i][j:j+60] + "\n")
                j += 60


if __name__=="__main__":
    """ Main script """
    # Model parameters:
    K = 67
    D = 4

    data_folder = './data-handin3/'
    start_file = 1
    num_files = 5
    end_index = -1

    """ Cross validation on first 5 genomes """
    print("Starting cross validation.")
    # Load data
    genomes = load_fasta_data(data_folder, start_file, num_files, "genome", end_index)
    true_annotations = load_fasta_data(data_folder, start_file, num_files, "true-ann", end_index)
    # Count transitions and emmissions
    transition_counts, emission_counts = count_multiple_transitions_and_emissions(genomes, true_annotations, K, D)
    # Build and train models
    cross_models = train_cross_validation_models(transition_counts, emission_counts, K, D)
    # Predict annotations
    predicted_annotations = compute_cross_validation_annotations(cross_models, genomes)
    # Write predictions to file
    write_fasta_files(predicted_annotations, "cross_ann", "cross-val-annotations")


    """ Prediction on the 5 unannotated genomes """
    print("Predicting annotations on unknown genomes.")
    # Train model on first 5 genomes
    total_trans_counts = np.sum(transition_counts, 0)
    total_emiss_counts = np.sum(emission_counts, 0)
    best_model = training_by_counting(K, D, total_trans_counts, total_emiss_counts)
    start_file = 6
    unknown_genomes = load_fasta_data(data_folder, start_file, num_files, "genome", end_index)
    # Make predictions on the new genomes
    predicted_unknown_annotations = compute_multiple_annotations(best_model, unknown_genomes)
    write_fasta_files(predicted_unknown_annotations, "pred-ann", "predicted-annotations", start_file)
