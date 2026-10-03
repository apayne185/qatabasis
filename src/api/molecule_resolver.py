from __future__ import annotations
import re
import urllib.request
import urllib.parse
from dataclasses import dataclass, field
from typing import Optional
import os
import json
from src.api.problems import ChemistryProblem, MOLECULE_REGISTRY


ELECTRON_COUNTS = {
    'H':1, 'He':2, 'Li':3, 'Be':4, 'B':5, 'C':6, 'N':7, 'O':8,
    'F':9, 'Ne':10, 'Na':11, 'Mg':12,'Al':13, 'Si':14, 'P':15,
    'S':16, 'Cl':17,'Ar':18, 'K':19, 'Ca':20, 'Sc':21, 'Ti':22,
    'V':23, 'Cr':24, 'Mn':25, 'Fe':26,'Co':27,'Ni':28, 'Cu':29,   
    'Zn':30,'Ga':31,'Ge':32, 'As':33, 'Se':34,'Br':35, 'Kr':36,
}
CORE_ELECTRONS = {
    'H':0, 'He':0, 'Li':2,'Be':2, 'B':2, 'C':2, 'N':2, 'O':2,
    'F':2, 'Ne':2, 'Na':2,'Mg':2, 'Al':2,'Si':2, 'P':2, 'S':2,     
    'Cl':2, 'Ar':2, 'K':10,'Ca':10, 'Sc':10, 'Ti':10, 'V':10,
    'Cr':10, 'Mn':10,'Fe':10, 'Co':10,'Ni':10, 'Cu':10, 'Zn':10,
}




@dataclass
class ResolutionResult:
    name:str
    geometry:str                # PySCF atom string
    source:str                   #registry, pubchem, smiles, raw
    total_electrons: int
    active_electrons: int
    active_orbitals:Optional[int] 
    estimated_qubits: int
    freeze_core: bool
    warnings:list[str] = field(default_factory=list)
    metadata:dict= field(default_factory=dict) 


    def to_chemistry_problem(self, force_tier= None):
        problem = ChemistryProblem(
            atom_coordinates=self.geometry,
            reps=self._recommended_reps(),
            name=self.name,
            force_tier=force_tier,
        ) 

        fci = None
        if "registry_entry" in self.metadata:   
            fci = self.metadata["registry_entry"].get("fci_energy")
        
        problem.fci_energy = fci 

        problem.resolution_metadata = {
            "source": self.source,
            "total_electrons":self.total_electrons,
            "active_electrons": self.active_electrons,
            "freeze_core":self.freeze_core,
            "estimated_qubits": self.estimated_qubits,
            "warnings":self.warnings,    
            "fci_energy":fci,   
        }    
        return problem
    

    def _recommended_reps(self) -> int:
        q = self.estimated_qubits
        if q <= 4: return 1
        if q <= 12: return 2
        if q <= 16: return 2
        return 3
    


class MoleculeResolver:
    def __init__(self, max_qubits: int= 20, basis: str = "sto-3g", allow_network: bool = True, cache_dir: str | None = None):
        self.max_qubits= max_qubits
        self.basis= basis     
        self.allow_network = allow_network
        self.cache_dir= cache_dir  

        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)  


    def resolve(self, molecule_input: str, freeze_core: bool = True) -> ResolutionResult:
        molecule_input= molecule_input.strip()

        result = (
            self._try_local_registry(molecule_input)
            or self._try_raw_geometry(molecule_input)
            or self._try_smiles(molecule_input)
            or self._try_pubchem(molecule_input) )


        if result is None:
            raise ResolutionError(f"Could not resolve '{molecule_input}'.")
        result = self._apply_active_space(result, freeze_core)

        if result.estimated_qubits > self.max_qubits:  # enforce qubit cap
            raise MoleculeTooBigError(f"'{result.name}' requires {result.estimated_qubits} qubits after active space reduction (limit:{self.max_qubits})  ")
        return result    
    



    def resolve_batch(self, molecules: list[str],freeze_core: bool = True) -> dict[str, ResolutionResult]:
        results = {} 

        for mol in molecules:
            try:
                results[mol] = self.resolve(mol, freeze_core=freeze_core)   
            except (MoleculeTooBigError,ResolutionError) as e:  
                results[mol] = None
                print(f"[Resolver] Skipping '{mol}': {e} ")    

        return results




    def _try_local_registry(self, name: str) -> Optional[ResolutionResult]:
        key = name.upper() if name.upper() in MOLECULE_REGISTRY else name
        if key not in MOLECULE_REGISTRY:
            return None    
        
        entry= MOLECULE_REGISTRY[key]
        geometry = entry["geometry"]
        atoms = self._parse_atom_string(geometry)
        total_e = self._count_electrons(atoms)
        est_q= self._estimate_qubits_sto3g(atoms)   

        print(f"[Resolver] '{name}' found in local registry ")   

        return ResolutionResult(
            name=key, geometry=geometry, source="registry",
            total_electrons=total_e, active_electrons=total_e,
            active_orbitals=None, estimated_qubits=est_q,
            freeze_core=False,
            metadata={"registry_entry": entry},)   
    


    def _try_pubchem(self, name: str) -> Optional[ResolutionResult]:
        if not self.allow_network:
            return None
        cache_key = re.sub(r'[^a-zA-Z0-9]', '_', name.lower())
        cache_file= (os.path.join(self.cache_dir, f"{cache_key}.json")  if self.cache_dir else None)   
        
        if cache_file and os.path.exists(cache_file):
            with open(cache_file) as f:
                cached = json.load(f)   

            print(f"[Resolver] '{name}' loaded from PubChem cache.")
            return self._pubchem_dict_to_result(name,cached)

        try:
            # Printed before the network call, not after -- on a
            # network-restricted host (blocked egress, DNS blackhole) this
            # can stall up to the request timeout (~10-25s, see
            # _fetch_pubchem_geometry) before failing. Without this,
            # that stall looks like the process silently hanging rather
            # than a network attempt in progress.
            print(f"[Resolver] '{name}' not in local registry/SMILES -- "
                  f"trying PubChem network lookup...")
            geometry, metadata = self._fetch_pubchem_geometry(name)
            if geometry is None:
                return None
            
            if cache_file:
                with open(cache_file,'w') as f:
                    json.dump({"geometry": geometry, "metadata": metadata}, f)   

            return self._pubchem_dict_to_result(name, {"geometry": geometry, "metadata": metadata})

        except Exception as e:
            print(f"[Resolver] PubChem lookup failed for '{name}':{e} ")
            return None    
        


    def _try_smiles(self, smiles: str) -> Optional[ResolutionResult]:
        if ' ' in smiles or ';' in smiles:
            return None
        if not re.match(r'^[A-Za-z0-9@+\-\[\]()=#%./\\]*$', smiles):
            return None   
        

        try:
            from rdkit import Chem
            from rdkit.Chem import AllChem, Descriptors  

        except ImportError:
            print("[Resolver] RDKit not available, SMILES resolution disabled.")
            return None

        mol= Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        mol= Chem.AddHs(mol)
        result_code = AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())  

        if result_code!= 0:
            AllChem.EmbedMolecule(mol, AllChem.ETKDG())
        AllChem.MMFFOptimizeMolecule(mol)

        conf= mol.GetConformer()
        atoms= []
        geom_parts = []
        for atom in mol.GetAtoms():
            symbol = atom.GetSymbol()
            pos = conf.GetAtomPosition(atom.GetIdx())
            geom_parts.append(f"{symbol} {pos.x:.6f} {pos.y:.6f} {pos.z:.6f}")
            atoms.append(symbol)      

        geometry = "; ".join(geom_parts)
        total_e = self._count_electrons(atoms)    
        est_q = self._estimate_qubits_sto3g(atoms)  
        mol_wt = Descriptors.MolWt(Chem.RemoveHs(mol))   
        
        print(f"[Resolver] '{smiles}' resolved via RDKit MMFF geometry")  

        return ResolutionResult(
            name=smiles, geometry=geometry, source="smiles",
            total_electrons=total_e, active_electrons=total_e,
            active_orbitals=None, estimated_qubits=est_q,
            freeze_core=False,
            metadata={"smiles": smiles,"mol_weight": mol_wt, "num_atoms": len(atoms)},)   
    


    def _try_raw_geometry(self, raw: str) -> Optional[ResolutionResult]:
        if not re.search(r'[A-Z][a-z]?\s+[-\d.]', raw):
            return None    
        
        atoms = self._parse_atom_string(raw)
        total_e = self._count_electrons(atoms)
        est_q= self._estimate_qubits_sto3g(atoms)
        print(f"[Resolver] Using raw geometry string directly. ")  

        return ResolutionResult(
            name="custom", geometry=raw, source="raw",
            total_electrons=total_e, active_electrons=total_e,
            active_orbitals=None, estimated_qubits=est_q,
            freeze_core=False,
        )




    def _apply_active_space(self, result: ResolutionResult, freeze_core: bool) -> ResolutionResult:
        if not freeze_core:
            return result

        atoms= self._parse_atom_string(result.geometry)
        core_e = sum(CORE_ELECTRONS.get(a, 0) for a in atoms)
        active_e= result.total_electrons - core_e
        est_q_after=self._estimate_qubits_sto3g(atoms, freeze_core=True)

        if core_e > 0:
            result.warnings.append(f"Frozen {core_e} core electrons. Active space: {active_e} electrons, {est_q_after} qubits ")

        result.active_electrons = active_e
        result.estimated_qubits = est_q_after
        result.freeze_core= freeze_core
        return result   
    




    def _fetch_pubchem_geometry(self, name: str):
        encoded = urllib.parse.quote(name)
        cid_url = (
            f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/"
            f"name/{encoded}/cids/JSON")   

        req = urllib.request.Request(cid_url, headers={"User-Agent": "VQE-middleware/1.0"}) 

        with urllib.request.urlopen(req, timeout=10) as resp:
            cid_data = json.loads(resp.read())

        cids = cid_data.get("IdentifierList", {}).get("CID", [])
        if not cids:
            return None,None   
        
        cid = cids[0]

        sdf_url = (
            f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/"
            f"cid/{cid}/SDF?record_type=3d")   

        req2 = urllib.request.Request(sdf_url, headers={"User-Agent":"VQE-middleware/1.0"}) 

        with urllib.request.urlopen(req2, timeout=15) as resp:
            sdf_text = resp.read().decode("utf-8")

        geometry = self._parse_sdf_geometry(sdf_text)
        metadata={"pubchem_cid": cid, "name": name}
        print(f"[Resolver] '{name}' fetched from PubChem (CID={cid}).") 

        return geometry, metadata   
    



    def _parse_sdf_geometry(self,sdf_text: str) -> Optional[str]:
        lines = sdf_text.strip().splitlines()
        try:
            counts_line = lines[3]
            n_atoms = int(counts_line[:3].strip()) 

        except (IndexError, ValueError):
            return None

        atom_block_start= 4
        geom_parts= []        

        for i in range(n_atoms):
            line = lines[atom_block_start + i]
            parts = line.split()
            if len(parts) < 4:
                continue 

            x,y,z= parts[0], parts[1], parts[2]
            symbol= parts[3]  

            if not re.match(r'^[A-Z][a-z]?$',symbol):
                continue 

            geom_parts.append(f"{symbol} {x} {y} {z}")


        return "; ".join(geom_parts)  if geom_parts else None
    




    def _pubchem_dict_to_result(self, name: str, cached: dict) -> ResolutionResult:
        geometry = cached["geometry"]
        metadata = cached.get("metadata",{})    
        atoms= self._parse_atom_string(geometry)
        total_e= self._count_electrons(atoms) 
        est_q= self._estimate_qubits_sto3g(atoms)   

        return ResolutionResult(
            name=name, geometry=geometry,source="pubchem",
            total_electrons=total_e, active_electrons=total_e,
            active_orbitals=None, estimated_qubits=est_q,
            freeze_core=False, metadata=metadata,)




    @staticmethod
    def _parse_atom_string(geometry:str) -> list[str]:
        symbols=[]
        for part in geometry.split(";"):
            part = part.strip()
            if not part:
                continue    

            symbol = part.split()[0]
            symbols.append(symbol)  

        return symbols
    


    @staticmethod
    def _count_electrons(atoms: list[str]) -> int:
        return sum(ELECTRON_COUNTS.get(a, 0) for a in atoms)
    

    @staticmethod
    def _estimate_qubits_sto3g(atoms: list[str],freeze_core: bool = False) -> int:
        basis_fns = 0
        sto3g_total= {'H':1,'He':1,'Li':5,'Be':5,'B':5,'C':5,'N':5,
                         'O':5,'F':5,'Ne':5,'Na':9,'Mg':9,'Al':9,'Si':9,
                         'P':9,'S':9,'Cl':9,'Ar':9}   
        sto3g_valence = {'H':1,'He':1,'Li':4,'Be':4,'B':4,'C':4,'N':4,
                         'O':4,'F':4,'Ne':4,'Na':8,'Mg':8,'Al':8,'Si':8,
                         'P':8,'S':8,'Cl':8,'Ar':8}      
        
        table= sto3g_valence if freeze_core else sto3g_total 

        for atom in atoms:
            basis_fns += table.get(atom, 9) 

        return 2 * basis_fns




class ResolutionError(Exception):
    pass

class MoleculeTooBigError(Exception):
    pass