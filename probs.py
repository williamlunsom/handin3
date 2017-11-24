import numpy as np

em = np.load("emission_probs.npy")
tr = np.load("trans_probs.npy")

#Start C
C_start_from_N_probs = tr[0,[1,4,7,10,13,16,19]]
C_start_from_N_total_prob = np.sum(C_start_from_N_probs)
print("C_start_from_N_total_prob {}\n".format(C_start_from_N_total_prob))

C_start_from_R_probs = tr[[48, 51, 54, 57, 60, 63, 66],:]
C_start_from_R_probs = C_start_from_R_probs[:,[1,4,7,10,13,16,19]]
C_start_from_R_total_probs = np.sum(C_start_from_R_probs, axis=0)
C_start_from_R_total_prob = np.sum(C_start_from_R_total_probs)
print("C_start_from_R_total_prob {}\n".format(C_start_from_R_total_prob))

C_start_relative_probs = (C_start_from_N_probs*C_start_from_N_total_prob + C_start_from_R_total_probs*C_start_from_R_total_prob)
C_start_relative_probs = C_start_relative_probs/np.sum(C_start_relative_probs)
print("C_start_relative_probs {}\n".format(C_start_relative_probs))

#Stop C
C_stop_probs = tr[24, [25, 28, 31]]
C_stop_total_prob = np.sum(C_stop_probs)
C_stop_relative_probs = C_stop_probs/C_stop_total_prob
print("C_stop_total_prob {}\n".format(C_stop_total_prob))
print("C_stop_relative_probs {}\n".format(C_stop_relative_probs))

#Start R
R_start_probs = tr[45,[46, 49, 52, 55, 58, 61, 64]]
R_start_total_prob = np.sum(R_start_probs)
R_start_relative_probs = R_start_probs/R_start_total_prob
print("R_start_from_R_total_prob {}\n".format(R_start_total_prob))
print("R_start_relative_probs {}\n".format(R_start_relative_probs))

# Stop R
R_stop_from_N_probs = tr[0,[34, 37, 40]]
R_stop_from_N_total_prob = np.sum(R_stop_from_N_probs)
print("R_stop_from_N_total_prob {}\n".format(R_stop_from_N_total_prob))

R_stop_from_C_probs = tr[[27, 30, 33],:]
R_stop_from_C_probs = R_stop_from_C_probs[:, [34, 37, 40]]
R_stop_from_C_total_probs = np.sum(R_stop_from_C_probs, axis=0)
R_stop_from_C_total_prob = np.sum(R_stop_from_C_total_probs)
print("R_stop_from_C_total_prob {}\n".format(R_stop_from_C_total_prob))

R_stop_relative_probs = R_stop_from_C_total_probs*R_stop_from_C_total_prob + R_stop_from_N_probs*R_stop_from_N_total_prob
R_stop_relative_probs = R_stop_relative_probs/np.sum(R_stop_relative_probs)
print("R_stop_relative_probs {}\n".format(R_stop_relative_probs))

#N
N_to_N_prob = tr[0,0]
print("N_to_N_prob {}\n".format(N_to_N_prob))

R_start_to_N_total_prob = 1.0 - C_start_from_R_total_prob
print("R_start_to_N_total_prob {}\n".format(R_start_to_N_total_prob))

C_stop_to_N_total_prob = 1.0 - R_stop_from_C_total_prob
print("C_stop_to_N_total_prob {}\n".format(C_stop_to_N_total_prob))

#Repeat
R_to_R_prob = tr[45,43]
C_to_C_prob = tr[24,22]
print("R_to_R_prob {}\n".format(R_to_R_prob))
print("C_to_C_prob {}\n".format(C_to_C_prob))


#Emissions
N_em = em[0,:]
print("N_em {}\n".format(N_em))

em_22 = em[22,:]
em_23 = em[23,:]
em_24 = em[24,:]
print("em_22 {}\n".format(em_22))
print("em_23 {}\n".format(em_23))
print("em_24 {}\n".format(em_24))

em_43 = em[43,:]
em_44 = em[44,:]
em_45 = em[45,:]
print("em_43 {}\n".format(em_43))
print("em_44 {}\n".format(em_44))
print("em_45 {}\n".format(em_45))
