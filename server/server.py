from flask import Flask, jsonify, render_template
import open_clip
import pandas as pd
import time
import torch

model, preprocess, tokenizer = None, None, None
final_df = None
image_embeddings = None

app = Flask(__name__)

def init_model():
    global model, preprocess, tokenizer
    print('Initializing model...')
    start_time = time.time()
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    print('Initialized model.\tTime Taken: ', time.time() - start_time)


def init_final_df():
    global final_df
    print('Reading products data...')
    start_time = time.time()
    FILENAMES_DF_PATH = 'assets-224-filename-kaggle.csv'
    filename_df = pd.read_csv('../' + FILENAMES_DF_PATH)
    filename_df.head()
    def get_file_path_from_url(url):
        return '-'.join(url.split('/')[5:])

    products_df = pd.read_csv('../products.csv')
    products_df['filename'] = products_df['searchImage'].apply(get_file_path_from_url)
    products_df = products_df.drop(columns=['Unnamed: 0'])


    final_df = pd.merge(filename_df, products_df, left_on='0', right_on='filename', how='left').drop_duplicates(subset='filename', keep='first').reset_index()
    final_df['index'] = final_df.index
    print('Read products data\tTime taken: ', time.time() - start_time)

def init_image_embeddings():
    global image_embeddings
    print('Reading image embeddings...')
    start_time = time.time()
    image_embeddings = torch.load('../image_embeddings_kaggle.pt', map_location=torch.device('cpu'))
    image_embeddings /= image_embeddings.norm(dim=-1, keepdim=True)
    print('Read image embeddings.\nTime Taken: ', time.time() - start_time)


def init_ml():
    init_final_df()
    
    init_model()
    init_image_embeddings()

@app.route('/')
def home():
    return render_template('landingpage.html')

def getTopK(base_embedding, K=25):
    global image_embeddings
    probs = torch.nn.functional.cosine_similarity(image_embeddings, base_embedding.view(1, 512))
    topk_indices = probs.topk(K).indices
    return topk_indices, probs[topk_indices]

@app.route('/query/<query>')
def get_query(query):
    global tokenizer, model, final_df
    if not query:
        return "HTTP 400 Bad Request: Query cannot be empty", 400
    
    query_encoding = tokenizer(query)
    query_embedding = model.encode_text(query_encoding)
    query_embedding /= query_embedding.norm(dim=-1, keepdim=True)

    topk_indices, topk_scores = getTopK(query_embedding)
    products = final_df.iloc[topk_indices.tolist()].to_dict('records')
    return render_template('query.html', query=query, products=products, topk_scores=topk_scores.tolist())



@app.route('/product/<int:index>')
def get_product(index):
    global image_embeddings, final_df

    topk_indices, topk_scores = getTopK(image_embeddings[index])
    products = final_df.iloc[topk_indices.tolist()].to_dict('records')
    # Implement your logic here
    return render_template('product.html', current_product=final_df.iloc[index].to_dict(), products=products, topk_scores=topk_scores.tolist())

if __name__ == '__main__':
    init_ml()
    # Assert model
    assert model is not None
    assert preprocess is not None
    assert tokenizer is not None 
    # Assert df
    assert final_df is not None
    # Assert embeddings
    assert image_embeddings is not None
    app.run(debug=True)
