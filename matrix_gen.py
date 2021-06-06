import re
import os
import numpy as np
import copy
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdRGroupDecomposition
from rdkit.Chem import rdmolops
from rdkit.Chem import rdFMCS
from collections import defaultdict
from itertools import product

class Comp_matrix():

    def __init__(self, scaffold, mols, maxmol=10000):
        self.scaffold = scaffold
        self.mols = mols
        self.maxmol = maxmol
        self.rgp = rdRGroupDecomposition.RGroupDecompositionParameters()
        self.rgp.removeHydrogensPostMatch = True
        self.rgp.alignment =True
        self.removeAllHydrogenRGroups=True
        self.rg = rdRGroupDecomposition.RGroupDecomposition(self.scaffold, self.rgp)
    
    def process(self):
        for mol in self.mols:
            if mol.HasSubstructMatch(self.scaffold):
                self.rg.Add(mol)
        self.rg.Process()
        self.dataset = self.rg.GetRGroupsAsColumns()
        self.core = copy.deepcopy(self.dataset['Core'][0])
        print('Process Done!')
    
    def generate(self):
        res = enumeratemol(self.core, self.rg, maxmol=self.maxmol)
        return res


def makebond(target, chain):
    newmol = Chem.RWMol(rdmolops.CombineMols(target, chain))   
    atoms = newmol.GetAtoms()
    mapper = defaultdict(list)
    for idx, atm in enumerate(atoms):
        atom_map_num = atm.GetAtomMapNum()
        mapper[atom_map_num].append(idx)
    for idx, a_list in mapper.items():
        if len(a_list) == 2:
            atm1, atm2 = a_list
            rm_atoms = [newmol.GetAtomWithIdx(atm1),newmol.GetAtomWithIdx(atm2)]
            nbr1 = [x.GetOtherAtom(newmol.GetAtomWithIdx(atm1)) for x in newmol.GetAtomWithIdx(atm1).GetBonds()][0]
            nbr1.SetAtomMapNum(idx)
            nbr2 = [x.GetOtherAtom(newmol.GetAtomWithIdx(atm2)) for x in newmol.GetAtomWithIdx(atm2).GetBonds()][0]
            nbr2.SetAtomMapNum(idx)
    newmol.AddBond(nbr1.GetIdx(), nbr2.GetIdx(), order=Chem.rdchem.BondType.SINGLE)
    nbr1.SetAtomMapNum(0)
    nbr2.SetAtomMapNum(0)
    newmol.RemoveAtom(rm_atoms[0].GetIdx())
    newmol.RemoveAtom(rm_atoms[1].GetIdx())
    newmol = newmol.GetMol()
    return newmol

def enumeratemol(core,rg, maxmol=10000):
    smilist = []
    dataset = rg.GetRGroupsAsColumns()
    labels = list(dataset.keys())
    pat = re.compile("R\d+")
    labels = [label for label in labels if pat.match(label)]
    rgs = np.asarray([dataset[label] for label in labels])
    
    i, j = rgs.shape
    combs = [k for k in product(range(j), repeat=i)]
    res = []
    for i in combs:
        mol = core
        for idx,j in enumerate(i):
            mol = makebond(mol, rgs[idx][j])
        mol.SetProp('COMBINATION_IDX', '_'.join([str(combidx) for combidx in i]))
        mol.SetProp('CORE', Chem.MolToSmiles(core))

        AllChem.Compute2DCoords(mol)
        mol = Chem.RemoveHs(mol)
        smi = Chem.MolToSmiles(mol)
        if smi not in smilist:
            smilist.append(smi)
            res.append(mol)
        else:
            pass
        if len(res) > maxmol:
            break
    return res
