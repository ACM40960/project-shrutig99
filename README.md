# Visual Storytelling: AI-Powered Image Caption Generation

## Introduction

Step into a world where pictures transform into stories, and machines learn to speak the language of images. In today's digital age, the convergence of artificial intelligence and computer vision has propelled the development of novel applications that bridge the gap between visual data and human language. Among these, image captioning stands out as a compelling endeavor, wherein machines are tasked with automatically generating descriptive textual explanations for images. This transformative technology holds the potential to revolutionize various domains, including accessibility for visually impaired individuals, content organization, and immersive human-computer interactions, thereby motivating extensive research and innovation in the field of AI-driven image captioning.

![Image Captioning](https://github.com/ACM40960/project-shrutig99/assets/118550614/eb8994d6-08d2-4f01-868f-5ce5bdac936e)


## Motivation

Capturing the spirit of innovation, our motivation stems from the untapped potential within the realm of image captioning. We are driven by the aspiration to amplify the interpretive capabilities of AI, allowing it to not only recognize but deeply understand visual scenes. The potential of AI-driven image captioning across diverse sectors, enabling machines to describe images through text and bridging gaps for the visually impaired. It enhances content management, user experience in e-commerce, education, and entertainment, advancing our understanding of images and language, driving innovative approaches for accurate, coherent, and contextually relevant captions. As human-machine comprehension of visuals merges, the urge to refine AI-powered image captioning intensifies.


## Dataset

**Sample of Flickr8k Dataset:**

![Sample of Flickr8k Dataset](https://github.com/ACM40960/project-shrutig99/assets/118550614/31e6fbb0-ea7a-430f-8e48-6be54ef14c2e)


The Flickr8k dataset is a widely used benchmark dataset in the field of computer vision and natural language processing, specifically for the task of image captioning. It was created to facilitate research and development of algorithms that generate descriptive captions for images using artificial intelligence techniques. The dataset is designed to promote the development of models that can effectively bridge the gap between visual content and textual descriptions.

Key details about the Flickr8k dataset:

- **Size and Content:** The Flickr dataset comprises 8,000 high-quality images, each accompanied by five human-generated captions, resulting in around 40,000 captions. The images cover diverse scenes and activities, making it suitable for training and evaluating image captioning models.
- **Annotations:** Multiple captions per image provide varied perspectives and linguistic nuances for training image captioning models.

The complete dataset can be downloaded from these two links: [Flickr8k_Dataset](https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip) and [Flickr8k_Text](https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip)


## Installation

The entire project was implemented in Jupyter Notebook (Python). Jupyter Notebook is an open-source web application that allows you to create and share documents that contain live code, equations, visualizations, and narrative text. To use Jupyter Notebook in your system,

We used Anaconda for installation of Python as it provides a comprehensive Python distribution with Jupyter Notebook, simplifying package management and environment setup for data science and programming projects. Anaconda can be downloaded by following this [link](https://docs.anaconda.com/free/anaconda/install/index.html). Choose the appropriate installer for your operating system (Windows, macOS, or Linux).

The libraries used in this project are:

```python
!pip install numpy
!pip install pandas
!pip install plotly
!pip install matplotlib
!pip install opencv2
!pip install numpy
!pip install tensorflow
!pip install nltk
!pip install tqdm
!pip install keras
```

Please note that some libraries like TensorFlow, Keras, and NLTK are quite large and might take some time to download and install. Also, make sure you are using a virtual environment if desired, to keep your project's dependencies isolated.

After installing these libraries, you should be able to use the code you provided without any issues. If you encounter any specific errors or issues, feel free to ask for further assistance.


## Data Preprocessing

Before feeding the Flickr8k data into the model, we first split the dataset and then process it in two steps.

**Data Splitting:** The dataset is divided into training (6000 images), validation (1000 images), and test sets (1000 images). The training set is used to train the model, the validation set is used for tuning hyperparameters, and the test set is used to evaluate the final model's performance.

After splitting the dataset, we process it in the following two steps:

**1. Image Preprocessing:**

  - **Resize:** Images are resized to a fixed size (224x224x3) and stored as a tensor to ensure uniformity in input size for the neural network.
  
  - **Normalization:** Pixel values are normalized to range, [0, 1] to help the neural network converge efficiently.

**2. Caption Preprocessing:**

To create a model that predicts the next token of a sentence from previous tokens, we turn the caption associated with any image into a list of tokenized words, before casting it to a tensor that is used to train the network.

  - **Tokenization:** Captions are split into a list of individual words (tokens) to form the vocabulary with special `start` and `end` tokens marking the beginning and end of the sentence.

  > [`start`, 'a', 'man', 'holding', 'a', 'slice', 'of', 'pizza', '.', `end`]
 
This list of tokens is then turned into a list of integers, where every distinct word in the vocabulary has an associated integer value:

  > [0, 3, 98, 754, 3, 396, 207, 139, 3, 753, 18, 1]

Here, we have initialized the special `start` and `end` tokens to integer values 0 and 1 respectively. There is another special token, corresponding to unknown words `"unk"`. All tokens that don't appear anywhere in the vocabulary dictionary are considered unknown words. In the pre-processing step, any unknown tokens are mapped to the integer 2.

This list of integers is converted to a tensor and fed into the model.

  - **Vocabulary Building:** A vocabulary is created by collecting unique tokens from all captions. This vocabulary is used to map words to integers for the model input and output.

A dictionary is created by looping over the captions in the training dataset. If a token appears no less than `vocab_threshold` times in the training set, then it is added as a key to the dictionary and assigned a corresponding unique integer. In general, smaller values for `vocab_threshold` yield a larger number of tokens in the vocabulary.

**3. Caption Sequencing:**

  - **Padding:** Captions are padded with special tokens (like `start` and `end`) to make them of equal length, ensuring they can be processed in batches.
  
  - **Integer Mapping:** Words in the captions are mapped to their corresponding integer values using the vocabulary.

## ML Model

The complete ML Model is shown below.

![Model Algorithm](https://github.com/ACM40960/project-shrutig99/assets/118550614/101b4f4b-8f39-45b2-8821-21bbbbbcb018)

The core of the architecture is a combination of a Convolutional Neural Network (CNN) and a Recurrent Neural Network (RNN). The CNN processes the input image and extracts its visual features, which are then used as input to the RNN. The RNN, typically implemented as a Long Short-Term Memory (LSTM) network, takes these visual features and generates the caption sequentially, word by word.
The training procedure involves exposing the model to pairs of images and associated captions. The model learns to establish connections between image features and textual content. The CNN and RNN parameters are optimized to minimize the divergence between generated captions and ground truth captions in the training dataset.

Hypermeters in concern:

- `batch_size` - the batch size of each training batch. It is the number of image-caption pairs used to amend the model weights in each training step.
- `vocab_threshold` - the minimum word count threshold. Note that a larger threshold will result in a smaller vocabulary, whereas a smaller threshold will include rarer words and result in a larger vocabulary.
- `vocab_from_file` - a Boolean that decides whether to load the vocabulary from file.
- `embed_size` - the dimensionality of the image and word embeddings.
- `hidden_size` - the number of features in the hidden state of the RNN decoder.
- `num_epochs` - the number of epochs to train the model. 

### Encoder

![CNN Encoder](https://github.com/ACM40960/project-shrutig99/assets/118550614/1d7ee28e-01f7-4380-9402-771c4af2be4e)

The CNN-Encoder is a ResNet (Residual Network). ResNet serves as a powerful encoder for extracting high-level visual features from input images. ResNet-50 is a deep convolutional neural network architecture characterized by its residual blocks, which enable the training of significantly deeper networks without falling victim to the vanishing and exploding gradient problems. This is possible as ResNet only stores the key characteristics from the layers called as Identity Function rather. As an encoder in the image caption generator, ResNet-50 takes raw images and progressively transforms them through a series of convolutional layers, pooling operations, and residual connections. These layers capture hierarchical visual patterns, enhancing the model's ability to understand and represent complex image content. The main idea relies on the use of skip connections which allows to take the activations from one layer and suddenly feed it to another layer, even much deeper in the neural network and using that, we can build ResNets which enables to train very deep networks. 

The encoder accepts an image of size (224 x 224 x 3) and uses the pre-trained ResNet-50 architecture (with the final fully-connected layer removed) to extract features from a batch of pre-processed images. The output is then flattened to a vector, before being passed through a Linear convolutional layer to transform the feature vector to have the same size as the word embedding (dimensionality reduction).

This flattened output (vector of embedded image features) is fed into the decoder which here is an LSTM Model.

### Decoder

![RNN Decoder](https://github.com/ACM40960/project-shrutig99/assets/118550614/8232847e-b34f-4024-8074-e902717b257a)

LSTM (Long Short-Term Memory) is a widely adopted decoder in AI image caption generator models due to its sequential processing capability and memory retention. As the decoder, LSTM takes the high-level visual features extracted from the input image by the encoder and generates captions word by word, maintaining contextual coherence. Its ability to capture long-range dependencies and handle variable-length sequences makes it ideal for language generation tasks. By considering both the previously generated words and the encoded image features, LSTM ensures that the generated captions are contextually meaningful and closely aligned with the visual content, resulting in accurate and fluent image descriptions.

## Results


## Conclusion 

Here, we have an end-to-end neural network which not only detects and classifies images but also generates the most relevant captions in a natural language. A Convolutional Neural Network (CNN) model is successfully able to encode an image into a compact representation, further a Recurrent Neural Network (RNN) is able to generate a corresponding sentence. The model is trained to maximize the likelihood of the sentence given the image. BLEU metrics is incorporated to evaluate the quality (relevance) of generated sentences. By mimicking the human perceptual process, the model effectively learns to focus on relevant regions of an image while generating descriptive captions. This attention mechanism not only improves the quality and coherence of the generated captions but also demonstrates the model's ability to grasp the intricate details of the images. The incorporation of attention mechanisms signifies a step forward in enhancing the contextual understanding and storytelling capability of AI-powered image captioning systems.


## Future Scope

There is a large scope of advancement to this project, to name a few:

**1. Enhanced Caption Quality and Relevance:** Future research can explore more sophisticated attention mechanisms and strategies to further improve caption quality and relevance. This could involve incorporating contextual information, object relationships, and scene understanding to generate more accurate and contextually meaningful captions.

**2. Multimodal Captioning:** This model primarily focus on image captioning, but the concepts presented can be extended to other modalities such as videos and audio. Future work can involve creating multimodal captioning models that generate descriptive captions for a combination of visual and auditory inputs, enabling more comprehensive and immersive descriptions. The output of this model can be fed into a text-to-speech model to help visually impaired. 

**3. Domain Adaptation and Transfer Learning:** As the model designed is trained on Flickr8k, future research can investigate techniques for domain adaptation and transfer learning. This would allow the models to generalize better across different datasets and domains, leading to improved performance on diverse sets of images.

**4. Fine-Grained Image Understanding:** The attention mechanisms provides insights into which parts of an image are important for generating captions. Future research can leverage this information to enable models to perform more fine-grained image understanding tasks, such as object detection, image segmentation, and visual question answering.

**5. Controllable and Creative Captioning:** Building upon the foundation of attention mechanisms, researchers can explore methods for controlling the style, tone, and attributes of generated captions. This could lead to models that can produce captions with specific emotional tones, writing styles, or creative elements, catering to a wide range of application scenarios.

**6. Real-Time Captioning and Accessibility:** The speed of caption generation models can be improved to enable real-time captioning for live events, videos, and other dynamic content. This can have a significant impact on accessibility for individuals with hearing impairments or those who prefer captions for various reasons.

**7. Multilingual Captioning:** Expanding the scope of this model to different languages and regions can greatly enhance their practical utility. Future research can focus on developing caption generation models that are capable of describing images in multiple languages, including those with limited training data. The output can be even converted to Braille language.


## Poster

[Poster Link](https://github.com/ACM40960/project-shrutig99/blob/main/Shruti%20Ghorpade_Poster.pdf)


## Literature Review

[Literature Review Link](https://github.com/ACM40960/project-shrutig99/blob/main/Literature_review.pdf)


## Acknowledgements

I am profoundly grateful to Dr. Sarp Akcay for his exceptional guidance and steadfast encouragement throughout the module at University College Dublin, which proved instrumental in shaping the course of this endeavor. My appreciation also extends to University College Dublin for its provision of vital resources that were fundamental in the fruition of this project. Lastly, I extend my gratitude to an array of online resources and various publications whose invaluable insights greatly facilitated the learning process, ensuring accessibility and ease for all involved.


## References

1. D. Donahue, J. Jeff, L. Anne Hendrikcs, S. Guadarrama, M. Rohrbach, S. Venugopalan, K. Saenko, and T. Darrell, "Long-term recurrent convolutional networks for visual recognition and description," arXiv:1411.4389v2, Nov. 2014.
2. D. Elliott and F. Keller, "Image description using visual dependency representations," in Proceedings of the Conference on Empirical Methods in Natural Language Processing (EMNLP), 2013.
3. H. Fang, S. Gupta, F. Iandola, R. Srivastava, L. Deng, P. Dollar, J. Gao, X. He, M. Mitchell, J. Platt, et al., "From captions to visual concepts and back," arXiv:1411.4952, Nov. 2014.
4. M. Hodosh, P. Young, and J. Hockenmaier, "Framing image description as a ranking task: Data, models and evaluation metrics," Journal of Artificial Intelligence Research, pp. 853-899, 2013.
5. Karpathy and F. Fei-Fei Li, "Deep visual-semantic alignments for generating image descriptions," arXiv:1412.2306, Dec. 2014.
6. R. Kiros, R. Salahutdinov, and R. Zemel, "Multimodal neural language models," in International Conference on Machine Learning (ICML), pp. 595-603, 2014.
7. R. Kiros, R. Salakhutdinov, and R. Zemel, "Unifying visual-semantic embeddings with multimodal neural language models," arXiv:1411.2539, Nov. 2014.
8. Krizhevsky, I. Sutskever, and G. Hinton, "ImageNet classification with deep convolutional neural networks," in Advances in Neural Information Processing Systems (NIPS), 2012.
9. G. Kulkarni, V. Premraj, V. Ordonez, S. Dhar, S. Li, Y. Choi, A. C. Berg, and T. L. Berg, "Babytalk: Understanding and generating simple image descriptions," IEEE Transactions on Pattern Analysis and Machine Intelligence (PAMI), vol. 35, no. 12, pp. 2891-2903, 2013.


## Authors

- [Shruti Avinash Ghorpade](https://github.com/shrutig99) - shruti.ghorpade@ucdconnect.ie
- [Aditya Pratap Singh]() - aditya.singh@ucdconnect.ie
  
Feel free to reach out to us for any queries or collaborations

