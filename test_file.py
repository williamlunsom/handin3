from hmm_genome import *

real_ann = [0,0,0,1,2,3,22,23,24,25,26,27,0,0,0,34,35,36,43,44,45,46,47,48,4,5,6,22,23,24,28,29,30,37,38,39,43,44,45,49,50,51,0,0,0]

genome = read_fasta_file("test_genome.fa")
genome = genome["test_genome"]
true_ann = read_fasta_file("test_ann.fa")
true_ann = true_ann["test_ann"]


K = 67
D = 4

z = translate_annotation_to_indices_codon(true_ann, genome)
transition_counts, emission_counts = count_transitions_and_emissions(K, D, genome, z)

print(z == real_ann)
print(emission_counts)