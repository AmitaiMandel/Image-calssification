# Waste-calssification by using neural networks
This project was created for educational purposes using a Kaggle [dataset]( https://www.kaggle.com/rayhanzamzamy/non-and-biodegradable-waste-dataset).

## Exploratory data analysis
The dataset contains 250K images representing both of the main classes of this problem: biodegradable and non-biodegradable.
- Biodegradable (B): food, fruits, vegetables that can be naturally decomposed by microorganisms and, most of them, could be converted into compost.
- Non-biodegradable (N): the material that cannot be naturally decomposed: such as plastics, metals, inorganic elements, etc. Most of these materials can be recycled or reused for new purposes.

The original dataset contains 150K of images that were augmented using flips and rotations to avoid the imbalance of the target.

## Preprocessing

Before training the different models, a preprocessing technique called rescaling was apllied to normalize the images in the dataset. The RGB channel values are in the [0, 255] range, and this is not ideal for a neural network:

```python

normalization_layer = Rescaling(1./255)

normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(normalized_ds.as_numpy_iterator())
first_image = image_batch[0].astype(int)

print(f"Notice that now pixels are between {np.min(first_image)} and {np.max(first_image)}")

```

## Neural networks
The first neural networks that where used for this problem, included - CNN, VGG16, EfficientNet, Inception V3.
This was the roc curves for those networks:

![roc](https://github.com/AmitaiMandel/Image-calssification/blob/main/ROC_Curve.png)


As shown, EfficientNet got the best result out of these 4 methodes. 


## CLIP

Finally, we decided to try a relatively new neural network introduced by OpenAI called [CLIP](https://openai.com/blog/clip/).
CLIP neural network was pre trained on pairs of text and images found across the internet.

The assumption that was made, that if the images contain enough information in order to extract related text, there is a good chance, we will find some kind of unique similarity for the two different classes of the images. And in another words, we wish to explore whether or not the embedded vectors of the images hold enough information, in order identify similarity within the two classes and difference between the 2 classes.

There were 4 steps for this method:
1. Create a training data set that would be constructed from the embedded vectors of the original training data of the images, using CLIP.
2. Create a test data set that would be constructed from the embedded vectors of the original test data of the images, using CLIP.
3. Train Logistic regression model on the new training data.
4. Predict and evaluate according to the new test set.

```python

# train_all_pre_proclist and test_all_pre_proclist contain the file paths of the images.
# train_all_labels and test_all_labels contain the actual labels.


#  Create a training data set that would be constructed from the embedded vectors of the original training data of the images
all_features = []
BATCH_SIZE = 900

ln = len(train_all_pre_proclist)

for i in tqdm(range(0,ln,BATCH_SIZE)):
    images = [
        preprocess(
            Image.open(file)
        ) for file in train_all_pre_proclist[i:i+BATCH_SIZE]
    ]
    image_input = torch.tensor(np.stack(images)).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image_input)
        all_features.append(image_features)

train_features = torch.cat(all_features).cpu().numpy()
        

#  Create a test data set that would be constructed from the embedded vectors of the original test data of the images
all_features = []
BATCH_SIZE = 900

ln = len(test_all_pre_proclist)

for i in tqdm(range(0,ln,BATCH_SIZE)):
    images = [
        preprocess(
            Image.open(file)
        ) for file in test_all_pre_proclist[i:i+BATCH_SIZE]
    ]
    image_input = torch.tensor(np.stack(images)).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image_input)
        all_features.append(image_features)
        
test_features = torch.cat(all_features).cpu().numpy()   


#  Train Logistic regression model on the new training data.
from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression(random_state=0, C=0.316, max_iter=1000)
classifier.fit(train_features, train_all_labels)

#  Predict and evaluate according to the new test set.
predictions = classifier.predict(test_features)
        
```

This neural network received the results:

![cm](https://github.com/AmitaiMandel/Image-calssification/blob/main/CLIP%20cm.png)
![Class report](https://github.com/AmitaiMandel/Image-calssification/blob/main/classification_report.png)



