# ML4641-38

[Website](https://github.gatech.edu/pages/ychen3555/CS4641_midpoint/) <br>
[Report](https://docs.google.com/document/d/e/2PACX-1vTB7w6BZvP2KsmktiehBi8kyLjqEYaXBjcmXkMPJ3VKWX9yeukYJmgcOwkJCQmKS1z8zdyKzSjLOLyv/pub)


### Explaination of code:
The `/data` directory contains the following.
- `/data/raw_data/links.csv, /data/raw_data/movies.csv, /data/raw_data/ratings.csv, /data/raw_data/tags.csv` from the raw dataset.
- `/data/data.csv,` containing ***postprocessed*** data
- `/data/preprocessing/preprocessing.py` used to generate `/data/data.csv` from raw data
- `/data/preprocessing/visualize_data.py` used to generate visuals from raw data
- `/data/visuals` directory, containing images generated from raw data

The `/matrix_factorization` directory implements collaborative filtering
- `/matrix_factorization/model.py` contains the matrix factorization class.
 - `/matrix_factorization/train.py` contains training and evaluation scripts
 - `/matrix_factorization/tune_hyperparameters.py` performs a hyperparameter sweep and visualization
 - `/matrix_factorization/utils` contains some helpful functions, such as plotting visuals
 - `/matrix_factorization/visuals` contains images generated from scripts.

The `/knn` directory implements our KNN rating prediction with cosine similarity
 - `knn/knn.py` contains the full training, testing, and visualization scripts
- `K-means/visuals` contains visuals

 The `/neural_network` directory implements NN
 - `neural_network/wandb` directory contains training weights and other wandb info logs. See [here](https://api.wandb.ai/links/alantian2018/ofqmjy3h) for more comprehensive logs.
 -  `neural_network/MLP.py` and `neural_network/autoencoder.py` files are the class models for our autoencoder and MLP.
 - `neural_network/data_utils.py` contains the custom dataset class.
 - `neural_network/inference.py` contains files to load model and test the model. It also generates visuals
 - `neural_network/visualizations` are visualizations. 