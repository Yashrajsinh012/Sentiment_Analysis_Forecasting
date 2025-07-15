Sentiment Classification Using Semi-Supervised Learning
This project presents a semi-supervised learning pipeline for sentiment classification, focusing on minimizing manual annotation while maintaining strong model performance. The method leverages a small labeled dataset and a large corpus of unlabeled reviews, using active learning and metric learning to iteratively improve the model.

Pipeline Overview
The end-to-end training process follows a semi-supervised learning strategy, consisting of the following steps:

1. Labeled Data Initialization
A labeled dataset is collected from Kaggle and preprocessed.

This dataset is used to train the initial version of the model.

2. Zero-Shot Inference on Unlabeled Data
The trained model is used to perform zero-shot predictions on an unlabeled dataset (Amazon cellphone reviews).

3. Uncertainty-Based Sampling
Predictions are analyzed to identify samples where the model is most uncertain.

These uncertain samples are prioritized for human annotation, enabling efficient use of annotation resources.

4. Manual Annotation
Human annotators label the uncertain samples.

These new labels are merged with the existing labeled dataset.

5. Iterative Fine-Tuning
The model is fine-tuned using the updated dataset.

Initially, only the classifier is fine-tuned.

Later, both the encoder and classifier are fine-tuned.

This process is repeated iteratively until the model reaches satisfactory performance.

This pipeline integrates active learning with iterative fine-tuning to reduce annotation costs while gradually improving model accuracy.

Model Architecture
The sentiment classification model is built with the following components:

1. Encoder Head
The encoder transforms raw text inputs into meaningful sentence-level embeddings. It consists of:

BERT (Frozen):
Utilizes pre-trained BERT to generate contextual token embeddings. The BERT weights remain frozen during training to reduce computational load and prevent overfitting.

2-Layer BiLSTM:
Processes BERT embeddings to capture sequential dependencies in both forward and backward directions.

Attention Layer:
Learns to assign weights to tokens, highlighting the most relevant parts of the sentence.

Linear Projection:
Projects the attended BiLSTM outputs into a fixed-size sentence embedding.

2. Triplet Loss Head
A metric learning head designed to improve the quality of the learned sentence embeddings.

Triplet Loss (Batch Hard Strategy):
Encourages the model to pull embeddings of similar sentiment closer and push dissimilar ones apart in the embedding space. This enhances the model's ability to generalize across variations in sentence structure and phrasing.

3. Classifier Head
The classifier uses the sentence embedding to predict sentiment labels.

Fully Connected Layers:

Three linear layers with ReLU activations.

Includes dropout and batch normalization for regularization and stability.

Output Layer:

Final layer outputs logits for three sentiment classes:

Positive

Negative

Neutral

