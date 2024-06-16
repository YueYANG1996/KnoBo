# Install environment for KnoBo
conda create --name knobo python=3.10 -y
conda activate knobo
pip install -r requirements.txt

# download features
wget https://knowledge-bottlenecks.s3.amazonaws.com/features.zip
unzip features.zip -d data

# download MIMIC data
wget https://knowledge-bottlenecks.s3.amazonaws.com/MIMIC-CXR.zip
unzip MIMIC-CXR.zip -d data/datasets

# download ISIC data
wget https://knowledge-bottlenecks.s3.amazonaws.com/ISIC.zip
unzip ISIC.zip -d data/datasets

# download grounding functions
wget https://knowledge-bottlenecks.s3.amazonaws.com/grounding_functions.zip
unzip grounding_functions.zip -d data

# download model weights
wget https://knowledge-bottlenecks.s3.amazonaws.com/model_weights.zip
unzip model_weights.zip -d data

rm features.zip MIMIC-CXR.zip ISIC.zip grounding_functions.zip model_weights.zip