# Waste-calssification by using neural networks
This project was created for educational purposes using a Kaggle [dataset]( https://www.kaggle.com/rayhanzamzamy/non-and-biodegradable-waste-dataset).

# Exploratory data analysis
The dataset contains 250K images representing both of the main classes of this problem: biodegradable and non-biodegradable.
- Biodegradable (B): food, fruits, vegetables that can be naturally decomposed by microorganisms and, most of them, could be converted into compost.
- Non-biodegradable (N): the material that cannot be naturally decomposed: such as plastics, metals, inorganic elements, etc. Most of these materials can be recycled or reused for new purposes.

The original dataset contains 150K of images that were augmented using flips and rotations to avoid the imbalance of the target.

## Preprocessing

Before training the different models, a preprocessing technique called rescaling was apllied to normalize the images in the dataset. The RGB channel values are in the [0, 255] range, and this is not ideal for a neural network:

'''python
normalization_layer = Rescaling(1./255)

normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(normalized_ds.as_numpy_iterator())
first_image = image_batch[0].astype(int)
# Notice that now pixel values are in `[0, 1]`.
print(f"Notice that now pixels are between {np.min(first_image)} and {np.max(first_image)}")
'''

# Neural networks
The first neural networks that where used for this problem, included - CNN, VGG16, EfficientNet, Inception V3.
This was the roc curves for those networks:


# CLIP

Finally 
