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

**Note:** Unzip the contents of the Flickr8k_Text folder directly in a folder containing the Flickr8k_Dataset folder.

Alternatively, you can simply download the [Flickr8k Dataset folder](https://github.com/ACM40960/project-shrutig99/tree/main/Flickr8k%20Dataset)

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

A neural network model is defined that takes two inputs: a numeric feature vector (2048 dimensions) and a sequence of text data (with varying lengths). The model consists of a dense layer for the numeric input, an embedding layer followed by an `LSTM` layer for the text input. The outputs of both branches are added and further processed through dense layers, finally producing a `softmax-based` classification output. The model is compiled with categorical cross-entropy loss, `Adam optimizer`, and accuracy metric for training.
`Adam Optimizer` is preferred as Adam (Adaptive Momentum) algorithm performed the best results since we can keep track of the previous gradient and not end up in a local minima.

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

The Decoder's job is to **look at the encoded image and generate a caption word by word.**

Since it's generating a sequence, it would need to be a Recurrent Neural Network (RNN). We will use an LSTM.

The LSTM in this model processes textual input through embedding and captures context, while image features are handled by dense layers. The LSTM's output combines text and image information through addition, followed by dense layers that refine the representation. The final dense layer with softmax generates the next caption word, blending sequential and visual cues for coherent image captioning.

LSTM (Long Short-Term Memory) is a widely adopted decoder in AI image caption generator models due to its sequential processing capability and memory retention. As the decoder, LSTM takes the high-level visual features extracted from the input image by the encoder and generates captions word by word, maintaining contextual coherence. Its ability to capture long-range dependencies and handle variable-length sequences makes it ideal for language generation tasks. By considering both the previously generated words and the encoded image features, LSTM ensures that the generated captions are contextually meaningful and closely aligned with the visual content, resulting in accurate and fluent image descriptions.

### Attention

The Attention network computes weighted encoded images of where the decoder should pay attention.

Intuitively, how would you estimate the importance of a certain part of an image? You would need to be aware of the sequence you have generated so far, so you can look at the image and decide what needs describing next. For example, after you mention a man, it is logical to declare that he is holding a pizza.

This is exactly what the Attention mechanism does – it considers the sequence generated thus far, and attends to the part of the image that needs describing next.


![Attention Network](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning/blob/master/img/att.png?raw=true)

We will use `soft` attention, where the weights of the pixels add up to 1. If there are `p` pixels in our encoded image, then at each timestep `t` :

<p align="center">
    <img src="https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning/blob/master/img/weights.png?raw=true"> 
</p>

This entire process is interpreted as **computing the probability that a pixel is the place to look to generate the next word.**

### Model Overview

![Model Overview](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning/blob/master/img/model.png?raw=true)

- Once the Encoder generates the encoded image, we transform the encoding to create the initial hidden state `h` (and cell state `C`) for the LSTM Decoder.
- At each decode step,
  -  the encoded image and the previous hidden state is used to generate weights for each pixel in the Attention network.
  -  the previously generated word and the weighted average of the encoding are fed to the LSTM Decoder to generate the next word.

### Prediction Algorithm: Beam Search

We use a linear layer to transform the Decoder's output into a score for each word in the vocabulary.

The straightforward – and greedy – option would be to choose the word with the highest score and use it to predict the next word. But this is not optimal because the rest of the sequence hinges on that first word you choose. If that choice isn't the best, everything that follows is sub-optimal. And it's not just the first word – each word in the sequence has consequences for the ones that succeed it.

It might very well happen that if you'd chosen the third best word at that first step, and the second best word at the second step, and so on... that would be the best sequence you could generate.

It would be best if we could somehow not decide until we've finished decoding completely, and choose the sequence that has the highest overall score from a basket of candidate sequences.

Beam Search does exactly that, it has the algorithm:

**1. Initialization:** Start with a single initial caption, with the `start` token.

**2. Step-by-Step Generation:** At each decoding step, expand the top-k candidates from the previous step by predicting the next words using the language model. Calculate the probabilities of multiple candidate sequences by considering the next possible words and their associated probabilities.

**3. Candidate Selection:** Among the generated candidate sequences, retain the top-k sequences with the highest probabilities. These sequences become the new candidates for the next decoding step.

**4. Repeat:** Repeat steps 2 and 3 for a fixed number of decoding steps or until an `end` token is generated for all sequences.

**5. Final Selection:** Once the decoding process is complete, choose the candidate sequence with the highest overall probability as the final generated caption.

The key benefit of beam search lies in its ability to explore multiple potential caption sequences in parallel, ensuring that the generator produces captions that are not only contextually accurate but also exhibit diversity and creativity. Beam search strikes a balance between exploring alternative caption paths and maintaining computational efficiency, ultimately resulting in higher-quality image captions.

![Beam Search](https://github.com/ACM40960/project-shrutig99/assets/118550614/951f38be-2d24-477b-b892-f8a585a4e153)

As you can see, some sequences (striked out) may fail early, as they don't make it to the top k at the next step. Once k sequences (underlined) generate the <end> token, we choose the one with the highest score.

## Results

### Metrics 

To evaluate the model's performance on the validation set, we use the automated [BiLingual Evaluation Understudy (BLEU)](https://aclanthology.org/P02-1040.pdf) evaluation metric. This evaluates a generated caption against reference caption(s).

The authors of the [Show, Attend and Tell](https://arxiv.org/abs/1502.03044) paper observe that correlation between the loss and the BLEU score breaks down after a point, so they recommend to stop training early on when the BLEU score begins to degrade, even if the loss continues to decrease.

We used the BLEU tool available in the [NLTK module](https://www.nltk.org/_modules/nltk/translate/bleu_score.html).

Note that there is considerable criticism of the BLEU score because it doesn't always correlate well with human judgment.

Scale of BLEU Score:

- **0:** A BLEU score of 0 suggests no overlap between the generated and reference text. The generated text is entirely different from the reference, indicating poor quality and low relevance.
- **0 to 0.25:** A BLEU score in this range indicates minimal to marginal overlap with the reference text. The generated text partially captures some elements from the reference, but the quality and relevance are still significantly lacking.
- **0.25 to 0.5:** A BLEU score within this range signifies moderate overlap with the reference text. The generated text manages to capture certain key aspects of the reference, but there's room for improvement in terms of fluency and precision.
- **0.5 to 0.75:** A BLEU score in this range indicates substantial overlap with the reference text. The generated text is considerably aligned with the reference, showcasing good quality and relevance.
- **0.75 to 1:** A BLEU score within this range suggests high alignment with the reference text. The generated text is of excellent quality and closely resembles the reference, demonstrating a strong grasp of context and language.

Some predictions from our model:

![image](https://github.com/ACM40960/project-shrutig99/assets/118550614/9b1adbb2-4acc-4a71-81dd-de250aca3ee0)

![image](https://github.com/ACM40960/project-shrutig99/assets/118550614/955d9330-2dcf-4374-aa32-b4efe37bf952)

![image](https://github.com/ACM40960/project-shrutig99/assets/118550614/70959dec-17f0-4ce5-a878-cea1b1f751db)

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

## License

This project is under the [MIT License](https://github.com/ACM40960/project-shrutig99/blob/main/LICENSE)

## Authors

- [Shruti Avinash Ghorpade](https://github.com/shrutig99) - shruti.ghorpade@ucdconnect.ie
- [Aditya Pratap Singh](https://github.com/ACM40960/project-adityapratap) - aditya.p.singh@ucdconnect.in
  
Feel free to reach out to us for any queries or collaborations! We would love to receive any inputs or feedback.

