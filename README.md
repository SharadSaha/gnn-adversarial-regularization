
# Adversarial attacks and graph regularization

Demonstration of adversarial attack on trained deep learning models using 
deceptive data,and use of graph regularization to build models that 
are robust to such attacks.

---






##  Notebooks
* [fgsm-adversarial-attack](https://colab.research.google.com/drive/1rKE1V_7jXRgRr1r25vvTtNHRtjrOw6Ot?usp=sharing)
* [gnn-graph-regularization](https://colab.research.google.com/drive/1-kYI-APEC_X8r-goM-bIj4nbkeM-Mx-F?usp=sharing)
---

## Fast gradient sign method
The fast gradient sign method works by using the gradients of the 
neural network to create an adversarial example. For an input 
image, the method uses the gradients of the loss with respect to 
the input image to create a new image that maximises the loss. 
This new image is called the adversarial image. 
This can be summarised using the following expression:
```bash
  adv_x = x + ϵ*sign(∇J(θ,x,y))
```

### Addition of adversarial pattern

![App Screenshot](https://github.com/SharadSaha/gnn-adversarial-regularization/blob/main/src/images/grid1.png?raw=true)

### Misclassification on addition of adversarial pattern

![App Screenshot](https://github.com/SharadSaha/gnn-adversarial-regularization/blob/main/src/images/grid2.png?raw=true)


****
****

## Graph regularization

### Dataset
![App Screenshot](https://github.com/SharadSaha/gnn-adversarial-regularization/blob/main/src/images/grid3.png?raw=true)

Access the dataset [here](https://drive.google.com/drive/folders/1iF8R25augxNtgyGYo4p9Ddu0pArzGcD7?usp=sharing).

This dataset contains four types of shapes namely, category 1, category 2, category 3 and category 4.




**Necessary steps to build graph regularized CNN model:**
* dataset loading and image preprocessing.

* get image embeddings/ feature vectors with the help of pretrained Inception v3 model.

* Generate the similarity graph using APIs from neural-structured-learning framework.

* Generate augmented training data from the synthesized graph and sample features. The resulting training data will contain neighbor features in addition to the original node features.

* Create the base convolutional neural network model usng Keras APIs.

* Wrap the base model with the GraphRegularization wrapper class, which is provided by the NSL framework, to create a new graph Keras model. 

* Training and evaluation of both base model and graph regularized model.

* Comparison of performances of both models.


### Comparison of performances of both models

![App Screenshot](https://github.com/SharadSaha/gnn-adversarial-regularization/blob/main/src/images/history.png?raw=true)
* **Key observation:** The performance of graph regularized model is better on validation set as Compared to base CNN model, as it is more robust to deceptive samples.
---

## Tools and technologies

**Deep learning framework:** tensorflow, nsl

**Other tools and frameworks:** scikit-learn, streamlit, dvc, jupyter notebook


--- 

## Hosting

The web application has been hosted [here](https://share.streamlit.io/sharadsaha/gnn-adversarial-regularization/main/src/app.py).


## Run Locally

Clone the project

```bash
  git clone <link-to-this-repo>
```

Go to the project directory

```bash
  cd gnn-adversarial-regularization
```

Install dependencies

```bash
  pip install -r requirements.txt

```

Start the server

```bash
  streamlit run src/app.py
```
