import awkward as ak
import numpy as np
import uproot as uproot
import pandas as pd

# This only works on CERN's machines
datafile = uproot.open('/eos/cms/store/user/bmaier/hgcal/hue_hackathon/samples_v1/test/diphoton/diphoton_merged_test.root')

events = datafile['lCToTSTsAssoc/lCTo3simTS_tree']

# totEvents = len(events['event'].array())
# print(totEvents)

nEvents = 1

id = np.array(ak.flatten(events['id'].array(entry_stop=nEvents)))
eta = np.array(ak.flatten(events['eta'].array(entry_stop=nEvents)))
phi = np.array(ak.flatten(events['phi'].array(entry_stop=nEvents)))
pos_x = np.array(ak.flatten(events['pos_x'].array(entry_stop=nEvents)))
pos_y = np.array(ak.flatten(events['pos_y'].array(entry_stop=nEvents)))
pos_z = np.array(ak.flatten(events['pos_z'].array(entry_stop=nEvents)))
layer = np.array(ak.flatten(events['layer'].array(entry_stop=nEvents)))
simTst_idx = np.array(ak.flatten(events['simTst_idx'].array(entry_stop=nEvents)))

clue3DHigh = events['isSeedCLUE3DHigh'].array(entry_stop=nEvents)
clue3DLow = events['isSeedCLUE3DLow'].array(entry_stop=nEvents)
clue3Dhigh_flat = ak.flatten(clue3DHigh)
clue3Dlow_flat = ak.flatten(clue3DLow)
isSeed = [clue3Dhigh_flat[i] or clue3Dlow_flat[i] for i in range(len(clue3Dhigh_flat))]
isSeed_idx = np.array(isSeed)

trueTrack1 = []
trueTrack2 = []
trueTrack3 = []

for i in range(len(simTst_idx)):
    trueTrack1.append(simTst_idx[i,0])
    trueTrack2.append(simTst_idx[i,1])
    trueTrack3.append(simTst_idx[i,2])

trueTrack1 = np.array(trueTrack1)
trueTrack2 = np.array(trueTrack2)
trueTrack3 = np.array(trueTrack3)

variables = ["lcID", "lcEta", "lcPhi", "lcX", "lcY", "lcZ", "lcLayer", "lcToTrack1", "lcToTrack2", "lcToTrack3", 'isSeed']

dataframe = np.vstack((id,eta,phi,pos_x,pos_y,pos_z,layer,trueTrack1,trueTrack2,trueTrack3, isSeed_idx)).T
df = pd.DataFrame(dataframe,columns=variables)

df.to_csv('diphoton.csv')