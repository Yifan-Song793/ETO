# 1. Install Python dependencies
pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt

cd eval_agent
pip install -r requirements.txt
cd ..

pip install scienceworld==1.1.3

cd envs/webshop
pip install -e .
python -m spacy download en_core_web_lg
conda install -y -c conda-forge openjdk=11

# 2. Download data for WebShop environment
gdown https://drive.google.com/uc?id=1G_0ccLWn5kZE5rpeyAdh_YuoNzvBUjT9
gdown https://drive.google.com/uc?id=11zOUDkJSgGhYin9NxQtG8PVpDsika86y
unzip data.zip
mkdir search_index
unzip indexes.zip -d search_index/

# 3. Download data for ALFWorld environment
cd ../..
cd eval_agent/data/alfworld
gdown https://drive.google.com/uc?id=1y7Vqeo0_xm9d3I07vZaP6qbPFtyuJ6kI
unzip alfworld_data.zip

# 4. Download expert trajectories for SFT behavioral cloning
cd ../../..
gdown https://drive.google.com/uc?id=1YbhbL8RhQGDWFv5y6k1qgwRqSyFFsao8
unzip ETO_data.zip