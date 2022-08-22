import numpy as np 
import pandas as pd 
import csv 
import global_vars
import pickle 
from protlearn.features import aaindex1
from protlearn.features import entropy
from protlearn.features import atc
from protlearn.features import entropy
from lightgbm import LGBMRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, r2_score

#--------------------------------------------------------------------
# GENERATING PROTTRANS TRANSFORMER EMBEDDINGS 
transformer = pd.read_csv("project_contents/app/transformer_embeddings.csv").drop(columns=["Unnamed: 0", "Unnamed: 0.1", "protein"])

def generate_embeddings(df, fasta): 
    """ """
    
    df1 = df.drop(columns="index")
    
    if fasta == "": 
        row = pd.DataFrame(columns=df1.columns)
        row.loc[0] = 0
        return row

    split_fasta = fasta.strip(">").split("SV=")[0]

    for idx, string in enumerate(df["index"]):
        
        if split_fasta in string: 
            row = pd.DataFrame(columns=df1.columns)
            row.loc[0] = df1.loc[idx]
            return row

    
def generate_transformer_features(fasta_lst): 
    """"""

    embed1 = generate_embeddings(transformer, fasta_lst[0])
    embed2 = generate_embeddings(transformer, fasta_lst[1])
    embed3 = generate_embeddings(transformer, fasta_lst[2])

    transformer_feats = pd.concat([embed1, embed2, embed3], axis=1)
    return transformer_feats 

#--------------------------------------------------------------------
# GENERATING PROTLEARN PHYSIOCHEMICAL FEATURES 

def generate_chemical_features(fasta_lst, columns): 
    """ """
    
    # initialising dataframe 
    df1 = pd.DataFrame()

    for j in columns: 
        df1[j+"_entropy"] = 0
    
    df2 = df1.append([0])
    for j in columns: 
        df2[j+'_C'] = 0
        df2[j+'_H'] = 0
        df2[j+'_N'] = 0
        df2[j+'_O'] = 0
        df2[j+'_S'] = 0
        df2[j+'_total_bound'] = 0
        df2[j+'_single_bound'] = 0
        df2[j+'_double_bound'] = 0
    
    # assigning values 
    for i in range(len(fasta_lst)): 
        seq = fasta_lst[i].split(" ")[-1]

        for j in columns:
            if seq != '':
                ent = entropy(seq)
                atoms, bonds = atc(seq)
                df2[j+'_entropy'] = str(ent)
                df2[j+'_C'] = str(atoms[0][0])
                df2[j+'_H'] = str(atoms[0][1])
                df2[j+'_N'] = str(atoms[0][2])
                df2[j+'_O'] = str(atoms[0][3])
                df2[j+'_S'] = str(atoms[0][4])
                df2[j+'_total_bound'] = str(bonds[0][0])
                df2[j+'_single_bound'] = str(bonds[0][1])
                df2[j+'_double_bound'] = str(bonds[0][2])

    return df2.drop(columns=0)


def generate_aa_features(indx, family, seq): 
    """ """
    
    df1 = pd.DataFrame(columns=[indx])
    
    if seq != '':
        aaind, inds = aaindex1(seq, standardize='zscore')
        
        for j in range(553):
            df1.loc[0,family+'_'+inds[j]] = aaind[0][j]
    
        df1.drop(df1.iloc[:, 1:552], axis=1,inplace=True)
        df1.drop(['ANDN920101','KARS160122'],axis=1,inplace=True)

        return df1

    for j in range(553):
        df1.loc[0,family+'_'+indx[j]] = 0
    
    df1.drop(df1.iloc[:, 1:552], axis=1,inplace=True)

    return df1 


def encode_str(id_dict, val): 
    """
    function encodes a string as an integer using an ID dict 
    params: 
        id_dict: dictionary containing strings and corresponding
                integer IDs
        val: a string or integer value 
    returns: an integer ID value 
    """
    
    if val == "": 
        return 0

    encoded_id = id_dict[val]
    return encoded_id 

    
def generate_protlearn_features(fasta_lst, columns, indx, families, food_group, prot_fam_id, food_group_id): 
    """ """
    
    chemical_feats = generate_chemical_features(fasta_lst, columns)
    aa_feats_1st = generate_aa_features(indx, columns[0], fasta_lst[0].split(" ")[-1])
    aa_feats_2nd = generate_aa_features(indx, columns[1], fasta_lst[1].split(" ")[-1])
    aa_feats_3rd = generate_aa_features(indx, columns[2], fasta_lst[2].split(" ")[-1])
    
    all_features = pd.concat([chemical_feats, aa_feats_1st, aa_feats_2nd, aa_feats_3rd], axis=1)
    all_features['1st family'] = encode_str(prot_fam_id, families[0])
    all_features['2nd family'] = encode_str(prot_fam_id, families[1])
    all_features['3rd family'] = encode_str(prot_fam_id, families[2])
    all_features['Food group'] = encode_str(food_group_id, food_group)
    
    return all_features 

#--------------------------------------------------------------------
# GENERATING TEST DATA 

def generate_test_data(nutrition_data, protlearn_features, transformer_features, shap_features): 
    """ """
    
    prot_nutrient = pd.concat([nutrition_data, protlearn_features], axis=1)
    prot_nutrient = prot_nutrient.filter(shap_features, axis=1)
    print(transformer_features.shape)
    print(prot_nutrient.shape)
    
    prot_nutrient_trans = pd.concat([prot_nutrient, transformer_features], axis=1)
    
    for i in range(1, 12): 
        prot_nutrient_trans.loc[i] = prot_nutrient_trans.loc[0]
    
    prot_nutrient_trans["Indispensable Amino Acid"] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    
    return prot_nutrient_trans

#--------------------------------------------------------------------
# RUNNING MODEL AND GENERATING PREDICTIONS 

def predict_digestibility(nutrient_data, fasta_lst, families, food_group): 
    """ """
    transformer_features = generate_transformer_features(fasta_lst)
    protlearn_features = generate_protlearn_features(fasta_lst, global_vars.columns, 
                        global_vars.inds, families, food_group, global_vars.PROTEIN_FAMILY_ID, 
                        global_vars.FOOD_GROUP_ID)
    demo_data = generate_test_data(nutrient_data, protlearn_features, 
                transformer_features, global_vars.shap_features)
    X_test = demo_data.drop(columns=["Food Item"])
    print(X_test, demo_data.columns)
    
    pickled_model = pickle.load(open('lgbm_model.pkl', 'rb'))
    digest_arr = pickled_model.predict(X_test)

    return digest_arr

def calculate_diaas(nutrition_data, digest_arr):
    """ """
    ref1 =  {"TRP":17,"THR":44,"ILE":55,"LEU":96, "LYS":69,"SAA":33, "AAA":94, "VAL":55, "HIS":21}
    ref2 = {"TRP":8.5,"THR":31,"ILE":32,"LEU":66, "LYS":57,"SAA":27, "AAA":52, "VAL":43, "HIS":20}
    ref3 = {"TRP":6.6,"THR":25,"ILE":30,"LEU":61, "LYS":48,"SAA":23, "AAA":41, "VAL":40, "HIS":16}

    prot = nutrition_data["Protein"]*10
    aa_composition = np.array(nutrition_data[["TRP","THR","ILE","LEU","LYS","MET","CYS","PHE","TYR","VAL","ARG","HIS"]])
    diaas = {"TRP":0,"THR":0,"ILE":0,"LEU":0, "LYS":0,"SAA":0, "AAA":0, "VAL":0, "HIS":0}
    temp = []

    for i in range(len(digest_arr)):
        digest = aa_composition[0][i]*digest_arr[i]
        diaas_val = (digest/prot)*1000
        temp.append(diaas_val)

    diaas["TRP"] = temp[0]
    diaas["THR"] = temp[1]
    diaas["ILE"] = temp[2]
    diaas["LEU"] = temp[3]
    diaas["LYS"] = temp[4]
    diaas["SAA"] = temp[5] + temp[6]
    diaas["AAA"] = temp[7] + temp[8]
    diaas["VAL"] = temp[9]
    diaas["HIS"] = temp[11]

    diaas1 = []
    diaas2 = []
    diaas3 = []

    for key in diaas: 
        diaas1.append(float(diaas[key]/ref1[key]))
        diaas2.append(float(diaas[key]/ref2[key]))
        diaas3.append(float(diaas[key]/ref3[key]))

        
    return min(diaas1), min(diaas2), min(diaas3)
