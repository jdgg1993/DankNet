# Training Model

### Setup

1. Before we begin, make sure you've downloaded the [`dataset`](https://www.kaggle.com/sayangoswami/reddit-memes-dataset/downloads/reddit-memes-dataset.zip/3) and you have python 3.6, pip and pipenv installed on your machine.
2. Clone this repo and inside `1. Training` run the command `pipenv install` from a command window. This will setup an environment for you with everything we need to get started.
3. Once everything is done setting up, activate your environment by using `pipenv shell`.
4. We need to now separate our dataset into `dank` and `not dank`. Inside `1. Training`, create a new folder and name it `data`. 
5. Inside `data`, create two new folders. Name one `dank` and the other `not_dank`.
6. Copy the images from the dataset downloaded previously to the folder `dank`.
7. From `1. Training`, run `python split.py`. This will create a 20% split in our dataset and populate the `not_dank` folder with images.
    - NOTE: This is probably the wrong approach to take. The `not_dank` folder should be populated with memes widely considered to be `not dank`. However, for demonstration purposes, lets just go with a random split. After all, a meme's "dankness" is subjective anyway. What's to say one meme is more dank over another? It's not like an image of a car. 
8. Double check both folders inside data contain images. If so, you're ready to start. 