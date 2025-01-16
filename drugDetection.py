import streamlit as st
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Draw
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import io
import base64
from PIL import Image

class MoleculeFeatureExtractor:
    """Extract molecular features using RDKit"""
    @staticmethod
    def calculate_descriptors(mol):
        if mol is None:
            return None
        
        descriptors = {
            'MolWeight': Descriptors.ExactMolWt(mol),
            'LogP': Descriptors.MolLogP(mol),
            'NumRotatableBonds': Descriptors.NumRotatableBonds(mol),
            'NumHAcceptors': Descriptors.NumHAcceptors(mol),
            'NumHDonors': Descriptors.NumHDonors(mol),
            'TPSA': Descriptors.TPSA(mol),
            'NumRings': Descriptors.RingCount(mol)
        }
        
        # Calculate Morgan fingerprints
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
        fp_bits = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(fp, fp_bits)
        
        # Combine descriptors and fingerprints
        features = np.array(list(descriptors.values()))
        return features

class DrugScreeningModel:
    """Machine learning models for toxicity and activity prediction"""
    def __init__(self):
        self.toxicity_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.activity_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        
    def train_models(self, X_train, y_tox_train, y_act_train):
        # Scale features
        X_scaled = self.scaler.fit_transform(X_train)
        
        # Train models
        self.toxicity_model.fit(X_scaled, y_tox_train)
        self.activity_model.fit(X_scaled, y_act_train)
    
    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        tox_prob = self.toxicity_model.predict_proba(X_scaled)[:, 1]
        activity = self.activity_model.predict(X_scaled)
        return tox_prob, activity

class DrugScreeningApp:
    def __init__(self):
        st.set_page_config(page_title="Drug Screening Application", layout="wide")
        self.model = DrugScreeningModel()
        self.feature_extractor = MoleculeFeatureExtractor()
    
    def run(self):
        st.title("Drug Screening Application")
        
        # Sidebar for navigation
        page = st.sidebar.selectbox("Choose a page", ["Single Molecule", "Batch Processing"])
        
        if page == "Single Molecule":
            self.single_molecule_page()
        else:
            self.batch_processing_page()
    
    def single_molecule_page(self):
        st.header("Single Molecule Analysis")
        
        # Input methods
        input_method = st.radio("Choose input method", ["SMILES", "Draw Molecule (Coming Soon)"])
        
        if input_method == "SMILES":
            smiles = st.text_input("Enter SMILES string", "CC(=O)OC1=CC=CC=C1C(=O)O")
            
            if st.button("Analyze"):
                self.process_single_molecule(smiles)
    
    def batch_processing_page(self):
        st.header("Batch Processing")
        
        uploaded_file = st.file_uploader("Upload CSV file with SMILES column", type="csv")
        
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            if 'SMILES' not in df.columns:
                st.error("CSV must contain a 'SMILES' column")
                return
                
            if st.button("Process Batch"):
                self.process_batch(df)
    
    def process_single_molecule(self, smiles):
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                st.error("Invalid SMILES string")
                return
            
            # Create two columns for results
            col1, col2 = st.columns(2)
            
            with col1:
                # Display 2D structure
                img = Draw.MolToImage(mol)
                st.image(img, caption="2D Structure")
                
                # Display basic properties
                st.subheader("Basic Properties")
                props = {
                    "Molecular Weight": Descriptors.ExactMolWt(mol),
                    "LogP": Descriptors.MolLogP(mol),
                    "H-Bond Donors": Descriptors.NumHDonors(mol),
                    "H-Bond Acceptors": Descriptors.NumHAcceptors(mol),
                    "Rotatable Bonds": Descriptors.NumRotatableBonds(mol)
                }
                
                for name, value in props.items():
                    st.write(f"{name}: {value:.2f}")
            
            with col2:
                # Calculate predictions
                features = self.feature_extractor.calculate_descriptors(mol)
                tox_prob, activity = self.model.predict(features.reshape(1, -1))
                
                st.subheader("Predictions")
                
                # Create gauge charts for toxicity and activity
                self.plot_gauge("Toxicity Risk", tox_prob[0])
                self.plot_gauge("Predicted Activity", activity[0])
                
                # Calculate and display composite score
                composite_score = 0.5 * (1 - tox_prob[0]) + 0.5 * activity[0]
                st.metric("Composite Score", f"{composite_score:.2f}")
                
                # Add interpretation
                st.subheader("Interpretation")
                self.interpret_results(tox_prob[0], activity[0], composite_score)
        
        except Exception as e:
            st.error(f"Error processing molecule: {str(e)}")
    
    def process_batch(self, df):
        results = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for idx, row in df.iterrows():
            try:
                mol = Chem.MolFromSmiles(row['SMILES'])
                if mol is not None:
                    features = self.feature_extractor.calculate_descriptors(mol)
                    tox_prob, activity = self.model.predict(features.reshape(1, -1))
                    composite_score = 0.5 * (1 - tox_prob[0]) + 0.5 * activity[0]
                    
                    results.append({
                        'SMILES': row['SMILES'],
                        'Toxicity_Probability': tox_prob[0],
                        'Predicted_Activity': activity[0],
                        'Composite_Score': composite_score
                    })
                
                progress = (idx + 1) / len(df)
                progress_bar.progress(progress)
                status_text.text(f"Processing molecule {idx + 1}/{len(df)}")
            
            except Exception as e:
                st.error(f"Error processing molecule {idx + 1}: {str(e)}")
        
        if results:
            results_df = pd.DataFrame(results)
            st.dataframe(results_df)
            
            # Download button for results
            csv = results_df.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="screening_results.csv">Download Results CSV</a>'
            st.markdown(href, unsafe_allow_html=True)
    
    @staticmethod
    def plot_gauge(title, value):
        # Create a simple gauge chart using st.progress
        st.write(title)
        st.progress(value)
        st.write(f"{value:.2%}")
    
    @staticmethod
    def interpret_results(toxicity, activity, composite):
        # Add interpretation text based on the results
        if composite > 0.7:
            st.success("This compound shows promising characteristics with a good balance of safety and efficacy.")
        elif composite > 0.4:
            st.warning("This compound shows moderate potential but may need optimization.")
        else:
            st.error("This compound may not be suitable for further development.")
        
        # Add detailed interpretation
        st.write("Detailed Analysis:")
        st.write(f"- Toxicity Risk: {'Low' if toxicity < 0.3 else 'Moderate' if toxicity < 0.7 else 'High'}")
        st.write(f"- Activity Potential: {'High' if activity > 0.7 else 'Moderate' if activity > 0.3 else 'Low'}")

if __name__ == "__main__":
    app = DrugScreeningApp()
    app.run()