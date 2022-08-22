import streamlit as st 
from PIL import Image
from backend import * 
import global_vars
import matplotlib.pyplot as plt 
import seaborn as sns 

#--------------------------------------------------------------------

# INITIAL PAGE CONFIG 

# images to display
logo = Image.open("project_contents/app/logo.png")
structure = Image.open("project_contents/app/protstructure.png")

# page configuration settings 
st.set_page_config(page_title="Digestibility Prediction Tool", 
                    page_icon=logo, 
                    layout="centered", initial_sidebar_state="auto")


left_col, right_col = st.columns(2)
with left_col:
    #st.image(structure, width=200)
    st.image(
            "https://bestanimations.com/media/protein/60289276protein-animation-22.gif", # I prefer to load the GIFs using GIPHY
            width=200, # The actual size of most gifs on GIPHY are really small, and using the column-width parameter would make it weirdly big. So I would suggest adjusting the width manually!
        )
with right_col:
    st.markdown("#### Modified Causal Quantitative Structure–Activity Relationship (MC-QSAR)")
    st.markdown("**A tool for predicting protein characteristics, such as digestibility**")


st.markdown("---")
st.markdown(
        """
        #### Summary
        Modified Causal QSAR is a tool for predicting protein characteristics, such as flavor, texture and protein digestibility.
        For the digestibility scenario, the database must include nutritional information of the food/protein and 3 protein 
        structures and families. In addition to that, the food scientist must select the type of food they are analyzing. 
        The protein biochemical characteristics are obtained through [Prolearn](https://protlearn.readthedocs.io/en/latest/introduction.html) and the embeddings through [ProtTrans](https://ieeexplore.ieee.org/document/9477085). The variables
        related to cause and effect relationships are defined through [DECI](https://www.microsoft.com/en-us/research/publication/deep-end-to-end-causal-inference/).
        Details of our work will be provided later on a paper.  
        """
    )

    

# displaying image and title


st.markdown("#### Food Protein Digestibility Predictor")


#--------------------------------------------------------------------

# SIZE BAR

#--------------------------------------------------------------------

# INPUT FIELDS  

with st.container(): 
    
     
    csv_upload = st.file_uploader("Upload CSV of nutritional information and \
                            amino acid composition of target protein")
    almond = pd.read_csv("project_contents/app/streamlit_demo_almondmilk.csv",index_col=0)
    with st.expander("See CSV example"):
        st.table(almond)
    

    #Method 1
    

    col1, col2 = st.columns(2)

    # fasta sequences of protein families of the food
    with col1: 
        fasta1 = st.text_input("Enter FASTA sequence of first protein family")
        fasta2 = st.text_input("Enter FASTA sequence of second protein family")
        fasta3 = st.text_input("Enter FASTA sequence of third protein family")
    
    with st.expander("See FASTA Example"):
        st.write("""
         Sample of FASTA sequence:

                >sp|Q43607|PRU1_PRUDU Prunin 1 Pru du 6 OS=Prunus dulcis OX=3755 PE=1 SV=1 MAKAFVFSLCLLLVFNGCLAARQSQLSPQNQCQLNQLQAREPDNRIQAEAGQIETWNFNQGDFQCAGVAASRITIQRNGLHLPSYSNAPQLIYIVQGRGVLGAVFSGCPETFEESQQSSQQGRQQEQEQERQQQQQGEQGRQQGQQEQQQERQGRQQGRQQQEEGRQQEQQQGQQGRPQQQQQFRQLDRHQKTRRIREGDVVAIPAGVAYWSYNDGDQELVAVNLFHVSSDHNQLDQNPRKFYLAGNPENEFNQQGQSQPRQQGEQGRPGQHQQPFGRPRQQEQQGNGNNVFSGFNTQLLAQALNVNEETARNLQGQNDNRNQIIQVRGNLDFVQPPRGRQEREHEERQQEQLQQERQQQGEQLMANGLEETFCSLRLKENIGNPERADIFSPRAGRISTLNSHNLPILRFLRLSAERGFFYRNGIYSPHWNVNAHSVVYVIRGNARVQVVNENGDAILDQEVQQGQLFIVPQNHGVIQQAGNQGFEYFAFKTEENAFINTLAGRTSFLRALPDEVLANAYQISREQARQLKYNRQETIALSSSQQRRAVV
     """)
    
    
    # corresponding protein families of the fasta sequences 

    prot_families = np.array(["","GLOBULIN","ALBUMIN","OVALBUMIN","OVOTRANSFERRIN","OVOMUCOID", 
                               "CASEIN","GLYCININ","CONGLYCININ","GLUTELIN","GLIADINS","ZEIN", 
                               "PROLAMIN","MYOSIN","MYOGLOBIN","PATATIN","LECTIN","LEGUMIN","OTHER"])
    
    food_groups = np.array(["Cereal & cereal products", "Roots & tubers", "Legumes & oilseeds", "Oil byproducts", 
                "Fish & fish products", "Animal products", "Milk products", "Fruits & vegetable products", 
                "Others", "Plant based ", "Mixed food (animal + cereal product)", "Mixed food (plant based)",
                "Mixed food (cereal + legume)", "Mixed food (cereal + animal product)"])

    with col2: 
        fam1 = st.selectbox("Protein Family 1:", prot_families)
        fam2 = st.selectbox("Protein Family 2:", prot_families)
        fam3 = st.selectbox("Protein Family 3:", prot_families)
    
    food_group = st.selectbox("Food group:", food_groups)

#--------------------------------------------------------------------

# BACKEND INTEGRATION AND OUTPUT 


with st.container(): 

    st.header("Digestibility Report")
    predict = st.button("Generate predictions")

    if predict: 

        fasta_lst = [fasta1, fasta2, fasta3]
        families = [fam1, fam2, fam3]
        nutrient_data = pd.read_csv(csv_upload)

        digest_arr = predict_digestibility(nutrient_data, fasta_lst, 
                                            families, food_group)
        c = ['lightskyblue', 'violet', 'pink', 'palegreen', 'khaki', 
            'coral', 'silver', 'aquamarine', 'forestgreen', 'royalblue', 
            'lightcoral', 'tan']
        
        fig = plt.figure(figsize =(10, 7))
        splot = sns.barplot(x=global_vars.aa, y=np.round(digest_arr, 2))
        plt.bar(global_vars.aa, digest_arr, color=c)
        plt.bar_label(splot.containers[0])
        plt.ylim(0, 1.5)
        st.pyplot(fig)

        d1, d2, d3 = calculate_diaas(nutrient_data, digest_arr)
        with st.sidebar:
    
            st.metric(label="DIAAS for Infants (0-6 months)", value=str(np.round(d1, 2)*100))
            st.metric(label="DIAAS for Children (6 months - 3 years)", value=str(np.round(d2, 2)*100))
            st.metric(label="DIAAS for Adults", value=str(np.round(d3, 2)*100))



#--------------------------------------------------------------------
