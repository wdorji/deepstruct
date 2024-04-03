# DeepStruct

# Download Dataset

The main dataset for training the model was the [STRING physical interaction for humans](https://string-db.org/cgi/download?sessionId=bhwu3EQrde6c&species_text=Homo+sapiens) where the file is named as <b>9606.protein.physical.links.v12.0.txt.gx</b>.

One benchmark dataset was the [Pan’s human dataset](https://github.com/JhaKanchan15/PPI_GNN/blob/main/Human_features/npy_file_new(human_dataset).npy) where you can extract the negative and positve edges by running the <b>deepstruct/preprocessing_scripts/create_pan_interactome.py</b> script. Another benchmark is the [STRING physical interaction for mouse](https://string-db.org/cgi/download?sessionId=bhwu3EQrde6c&species_text=Mus+musculus) where the file is named as <b>10090.protein.physical.links.v12.0.txt.gx</b>. To create the benchmark of proteins with multiple conformations, run the <b>deepstruct/preprocessing_scripts/create_multi_feats_interactions.py</b> script.

# Preprocessing and creating datasets

1. Make a positve PPI file ready in the format of:
    <div style="background-color: #f9f9f9; border: 1px solid #ccc; padding: 10px;">
    line1:    protein1  [Space]  protein2 <br>
    line2:    9606.ENSP00000000233 [Space]  9606.ENSP00000264718 <br>
    line3:    9606.ENSP00000000233 [Space]  9606.ENSP00000346046 <br>
    line4:    ...
    </div>

2. Get all graphs ids from the positve PPI file using <b>deepstruct/preprocessing_scripts/get_PPI_IDS.py</b> which you can run by using the command:<br>
    <div style="background-color: #f9f9f9; border: 1px solid #ccc; padding: 10px;">

    <b>python3 get_PPI_IDS.py [-h] <br>[-pos_int_data path to postive interaction dataset] <br>[-prefix prefix to save ouput txt file of PPI network IDS in data folder]
                     <br>[-STRING A boolean argument to specify if the PPI is from STRING database, if so only interactions with greater than 70% confidence is used] </b>
    </div>
    

3. Then feed this file to the [uniprot mapping tool](https://www.uniprot.org/id-mapping) to map its respective PPI network type to UniProtKB. For mapping:

    1) Select TSV as the download format
    2) Select No for compressed
    3) In the *Customize column* remove all pre-existing options 
    4) Drill down UniProt Data -> Sequences and select Sequence 
    5) Drill down External Resources -> 3D structure and select PDB 

    The downloaded file should look like this:

    <div style="background-color: #f9f9f9; border: 1px solid #ccc; padding: 10px;">
    line1:    From [Tab] Entry [Tab] Sequence [Tab] PDB <br>
    line2:    network_id1[Tab]uniprot_id_1[Tab]AAFG..[Tab]4NE1;1J6ZA;.. <br>
    line3:     network_id2[Tab]uniprot_id_1[Tab]AAFG..[Tab]4NE1;1J6ZA;.. <br>
    line4:    ...
    </div>

 
    <b>"From"</b> denotes the protein network ID such as the string ID <br>
    <b>"Entry"</b> denotes the uniprot ID <br>
    <b>"Sequences"</b> denotes the protein sequence <br>
    <b>"PDB"</b> denotes the list of all PDB files where the protein is found separeted by <b>;</b> 

4. Download and hydrate relevant pdb files mapped via the uniprot tool using <b>deepstruct/preprocessing_scripts/download_pdbs.py</b> which you can run by using the command:<br>

    <div style="background-color: #f9f9f9; border: 1px solid #ccc; padding: 10px;">

    <b>python3 get_PPI_IDS.py [-h]<br> [-ID_mapping path to uniprot mapping file]<br> [-hydrated_pdbs_dname path to hydrated pdbs directory]<br> [-pdbs_dname path to pdbs directory]<br>
    [-pos_int_data path to postive interaction dataset]<br> [-prefix prefix to save ouput txt file of PPI network IDS in data folder]<br>
                     [-STRING A boolean argument to specify if the PPI is from STRING database, if so only interactions with greater than 70% confidence is used] </b>
    </div>

5. Create positive, negative interaction datasets and protein descriptors using <b>deepstruct/preprocessing_scripts/create_feats_interactions.py</b> which you can run by using the command:<br>

    <div style="background-color: #f9f9f9; border: 1px solid #ccc; padding: 10px;">

    <b>python3 create_feats_interactions.py [-h] <br>[-ID_mapping path to uniprot mapping file] <br>[-dict_data path to pdb to chains information dictionary]<br> [-original A boolean argument to specify to not use proteins used in the multiconformation dataset]<br> [-eval_fold A boolean argument to specify to create 5 fold cross validation for performance evaluation]<br> [-hydrated_pdbs_dname path to hydrated pdbs directory]<br> [-pdbs_dname path to pdbs directory]<br>
    [-pos_int_data path to postive interaction dataset] <br>[-prefix prefix to save all output files to]<br>
    [-pdb_chains_dname path to pdb of single chains directory]<br>
                     [-STRING A boolean argument to specify if the PPI is from STRING database, if so only interactions with greater than 70% confidence is used] </b>
    </div>

# Train and test model

1. Train your model using <b>deepstruct/model/model.py</b> which you can run by using the command:<br>

    <div style="background-color: #f9f9f9; border: 1px solid #ccc; padding: 10px;">

    <b>python3 model.py [-h]<br> [-ID_mapping path to uniprot mapping file]<br> [-hydrated_pdbs_dname path to hydrated pdbs directory]<br> [-pdbs_dname path to pdbs directory]<br>
    [-pos_int_data path to postive interaction dataset]<br> [-prefix prefix to save ouput txt file of PPI network IDS in data folder]<br>
                     [-STRING A boolean argument to specify if the PPI is from STRING database, if so only interactions with greater than 70% confidence is used] </b>
    </div>