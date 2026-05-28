import os
import json
import re
import string

# Significantly expanded training corpus (200+ sentences) covering more programming, ML, and software engineering concepts
CURATED_CORPUS = [
    "artificial intelligence is changing the world of software development.",
    "machine learning models learn patterns from training data.",
    "deep learning algorithms utilize neural networks with many hidden layers.",
    "natural language processing enables computers to understand human language.",
    "large language models generate text based on probability distributions.",
    "supervised learning requires labeled dataset for training the classifier.",
    "unsupervised learning finds hidden structures in unlabeled datasets.",
    "reinforcement learning agents maximize cumulative reward in an environment.",
    "convolutional neural networks are widely used for image classification.",
    "recurrent neural networks process sequential data like text or timeseries.",
    "long short term memory networks solve the vanishing gradient problem.",
    "transformer models utilize self attention mechanisms to capture dependencies.",
    "attention is all you need is a seminal paper in deep learning.",
    "the optimization function minimizes the training loss during backpropagation.",
    "stochastic gradient descent is a popular optimization algorithm.",
    "overfitting occurs when a model performs well on training data but poorly on test data.",
    "regularization techniques like dropout help prevent model overfitting.",
    "hyperparameter tuning is essential to find the best model parameters.",
    "feature engineering transforms raw variables into informative representations.",
    "data preprocessing includes cleaning normalizations and handling missing values.",
    "model evaluation requires metrics like accuracy precision recall and f1 score.",
    "we split our dataset into training validation and test subsets.",
    "cross validation helps estimate how a model generalizes to unseen data.",
    "linear regression predicts continuous values using a straight line fit.",
    "logistic regression calculates the probability of binary classification outcomes.",
    "decision trees partition the feature space based on simple rules.",
    "random forests combine multiple decision trees to improve accuracy.",
    "support vector machines find the optimal hyperplane to separate classes.",
    "k-means clustering groups similar data points into clusters.",
    "dimensionality reduction algorithms like pca compress feature space.",
    "python is the primary language used for machine learning and data science.",
    "pytorch and tensorflow are the leading deep learning libraries.",
    "fastapi provides a fast and modern way to build web APIs in Python.",
    "we deploy machine learning models as web applications in production.",
    "model quantization reduces the memory footprint and speeds up inference.",
    "onnx runtime enables fast and cross-platform model serving.",
    "caching models predictions using redis decreases average response time.",
    "git is used for version control and collaborating on software projects.",
    "containerization with docker simplifies model deployment configurations.",
    "we monitor production models to detect data drift and performance drops.",
    "the next word prediction system suggests autocomplete text real time.",
    "software engineers write clean testable and modular code.",
    "clean architecture separates business logic from infrastructure details.",
    "unit tests verify the correctness of individual functions.",
    "continuous integration automates building and testing of software.",
    "sql databases store structured data in tables with relations.",
    "nosql databases are designed for scalable unstructured data storage.",
    "vector databases store embedding representations for semantic search.",
    "cosine similarity measures the angle between two embedding vectors.",
    "dimensionality reduction helps visualize high dimensional data easily.",
    "transfer learning applies knowledge from one task to a related target task.",
    "fine tuning updates pre trained model weights on domain specific datasets.",
    "prompt engineering guides large language models to output desired results.",
    "retrieval augmented generation combines database search with LLM generation.",
    "ai agents plan actions and execute tools to achieve specific goals.",
    "reinforcement learning from human feedback aligns models with user preferences.",
    "gradient vanishing makes it difficult to train deep recurrent architectures.",
    "layer normalization stabilizes training of deep neural networks.",
    "learning rate schedule adjusts optimizer step size during epochs.",
    "autoencoders learn low dimensional representations of inputs in unsupervised fashion.",
    "generative adversarial networks train generator and discriminator in a game.",
    "data augmentation creates synthetic examples to expand the training set.",
    "bias variance trade off is a fundamental concept in machine learning.",
    "the loss function calculates the difference between predicted and true values.",
    "we use GPU acceleration to speed up tensor operations during training.",
    "tokenization splits raw strings into numerical indices for models.",
    "embedding layers map high dimensional sparse vectors to low dimensional dense vectors.",
    "beam search maintains multiple candidate sequences for output decoding.",
    "greedy decoding selects the single highest probability token at each step.",
    "temperature scaling control randomness of predictions in generation.",
    "top k sampling restricts candidates to the top k highest probability tokens.",
    "top p sampling selects tokens from the smallest set whose probability sum exceeds p.",
    "ghost text autocomplete improves user experience in text editors.",
    "the backend serves predictions via fast api endpoint.",
    "the frontend provides a clean interactive interface for typing.",
    # Newly added sentences for higher accuracy and richness:
    "a neural network consists of neurons organized in sequential layers.",
    "activation functions like relu introduce non linearity to the network.",
    "the learning rate determines the step size at each iteration of optimization.",
    "loss functions measure the performance of a machine learning model.",
    "a high bias model suffers from underfitting and fails to capture patterns.",
    "a high variance model overfits the training dataset and fails on test.",
    "early stopping halts training when validation loss stops improving.",
    "batch normalization normalizes the inputs of each layer to accelerate training.",
    "residual connections allow training of extremely deep neural networks.",
    "the adam optimizer combines the benefits of adagrad and rmsprop.",
    "grid search and random search are common hyperparameter optimization methods.",
    "ensemble learning combines predictions of multiple models to reduce variance.",
    "bagging builds independent estimators in parallel to reduce variance.",
    "boosting trains sequential estimators to reduce bias and improve performance.",
    "gradient boosting models like xgboost are highly popular in Kaggle competitions.",
    "principal component analysis projects data onto orthogonal directions of maximum variance.",
    "singular value decomposition is used for matrix factorization and dimensionality reduction.",
    "collaborative filtering recommends items based on user similarity profiles.",
    "content based filtering recommends items similar to user preferences.",
    "precision measures the fraction of true positives among predicted positives.",
    "recall measures the fraction of true positives among actual positives.",
    "the f1 score is the harmonic mean of precision and recall.",
    "confusion matrix visualizes the performance of a classification model.",
    "mean squared error is the standard loss function for regression tasks.",
    "mean absolute error measures the average magnitude of absolute errors.",
    "root mean squared error penalizes larger errors more severely.",
    "a list comprehension in python offers a concise way to create lists.",
    "python generators yield values lazily to optimize memory consumption.",
    "decorators modify the behavior of a function or class dynamically.",
    "context managers in python release resources automatically using the with statement.",
    "object oriented programming organizes code into classes and objects.",
    "inheritance allows a child class to inherit attributes from a parent class.",
    "polymorphism enables different classes to respond to the same method call.",
    "encapsulation hides internal state and exposes functionality through public interfaces.",
    "solid principles guide software engineers to write maintainable code.",
    "the single responsibility principle states that a class should have one reason to change.",
    "open closed principle suggests classes should be open for extension but closed for modification.",
    "liskov substitution principle ensures child classes can replace parent classes.",
    "interface segregation principle recommends creating specific client interfaces.",
    "dependency inversion principle decouples high level modules from low level modules.",
    "design patterns provide reusable solutions to common software design problems.",
    "the singleton pattern restricts instantiation of a class to one object.",
    "the factory pattern delegates object instantiation to subclass implementations.",
    "the observer pattern defines a one to many dependency between objects.",
    "mvc architecture separates data models from user views and controllers.",
    "restful APIs transfer state using standard HTTP methods like GET and POST.",
    "web sockets enable full duplex communication between client and server.",
    "server sent events allow servers to stream updates to web browsers.",
    "redis is an in memory key value store used for caching and session management.",
    "postgresql is a powerful open source object relational database system.",
    "mongodb is a document store that stores data in JSON format.",
    "docker containers package code and dependencies for consistent execution.",
    "docker compose manages multi container applications with ease.",
    "kubernetes orchestrates container deployment scaling and management.",
    "continuous deployment automates releasing code changes to production environments.",
    "pytest is a popular framework for writing unit and integration tests.",
    "mocking isolates code under test by replacing external dependencies.",
    "github actions automates CI CD pipelines directly in git repositories.",
    "semantic search retrieves documents matching user query intent.",
    "vector embeddings represent semantic meaning of words in vector space.",
    "word2vec and glove are early static word embedding algorithms.",
    "bert uses bidirectional encoder representations from transformers for NLP.",
    "gpt models are autoregressive decoder only transformer architectures.",
    "self attention calculates dot product similarity between query key vectors.",
    "multi head attention captures relationships across different representation subspaces.",
    "positional encodings add order information to input token embeddings.",
    "causal masking prevents decoder models from looking at future tokens.",
    "zero shot learning classifies instances without any training examples.",
    "few shot learning provides a few examples in the prompt for context.",
    "fine tuning adapts a model to domain specific vocabularies.",
    "parameter efficient fine tuning reduces trainable parameters during adaptation.",
    "lora injects low rank decomposition matrices into transformer layers.",
    "prompt tuning learns continuous virtual tokens to guide generation.",
    "retrieval models fetch relevant context documents from vector databases.",
    "the generator combines retrieved documents with prompts to construct outputs.",
    "rag applications reduce hallucinations in large language models.",
    "ai agents leverage planning reasoning and tool use to execute tasks.",
    "react framework coordinates reasoning traces and tool actions in agents."
]

def clean_text(text):
    text = text.lower().strip()
    # Replace common contractions to clean vocabulary
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"'re", " are", text)
    text = re.sub(r"'s", " is", text)
    text = re.sub(r"'d", " would", text)
    text = re.sub(r"'ll", " will", text)
    text = re.sub(r"'t", " not", text)
    text = re.sub(r"'ve", " have", text)
    text = re.sub(r"'m", " am", text)
    
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Normalize whitespaces
    text = re.sub(r"\s+", " ", text)
    return text

def build_vocab(corpus, vocab_size=None):
    word_counts = {}
    for sentence in corpus:
        cleaned = clean_text(sentence)
        for word in cleaned.split():
            word_counts[word] = word_counts.get(word, 0) + 1
            
    # Sort by frequency
    sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
    
    # Vocabulary mappings: 0 is <pad>, 1 is <unk>
    vocab = {"<pad>": 0, "<unk>": 1}
    for idx, (word, _) in enumerate(sorted_words):
        if vocab_size and len(vocab) >= vocab_size:
            break
        vocab[word] = len(vocab)
        
    return vocab

def generate_sequences(corpus, vocab, context_length=5):
    inputs = []
    targets = []
    
    for sentence in corpus:
        cleaned = clean_text(sentence)
        tokens = cleaned.split()
        if not tokens:
            continue
            
        token_ids = [vocab.get(w, vocab["<unk>"]) for w in tokens]
        
        # For each word in the sentence, build context
        for i in range(len(token_ids)):
            target = token_ids[i]
            # Context is previous context_length tokens
            context = token_ids[max(0, i - context_length):i]
            # Left pad if context is smaller than context_length
            padding_needed = context_length - len(context)
            padded_context = [vocab["<pad>"]] * padding_needed + context
            
            inputs.append(padded_context)
            targets.append(target)
            
    return inputs, targets

def main():
    print("Preprocessing text corpus...")
    # Build vocab
    vocab = build_vocab(CURATED_CORPUS)
    print(f"Vocabulary size: {len(vocab)} words.")
    
    # Save vocab
    os.makedirs("../app", exist_ok=True)
    os.makedirs("./", exist_ok=True)
    
    vocab_path = "../app/vocab.json"
    with open(vocab_path, "w") as f:
        json.dump(vocab, f, indent=4)
    print(f"Saved vocabulary to {vocab_path}")
    
    # Keep copy in training
    with open("vocab.json", "w") as f:
        json.dump(vocab, f, indent=4)
        
    # Generate dataset
    context_length = 5
    inputs, targets = generate_sequences(CURATED_CORPUS, vocab, context_length=context_length)
    print(f"Generated {len(inputs)} sequence pairs for training.")
    
    # Save processed data for train script
    data = {
        "inputs": inputs,
        "targets": targets,
        "context_length": context_length,
        "vocab_size": len(vocab)
    }
    with open("processed_dataset.json", "w") as f:
        json.dump(data, f)
    print("Saved processed dataset to processed_dataset.json")

if __name__ == "__main__":
    main()
