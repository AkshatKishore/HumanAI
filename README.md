![HumanAI Logo](https://github.com/AkshatKishore/HumanAI/assets/141553703/66028614-48da-4d8a-9f5a-4bc436d076a0)
# HumanAI
This document contains technical details on the development of 3 publicly available ML models: Palpitate, Harmonious-Minds, and Eini. 
The complete methodology, dataset details, mathematical explication, and deployment details can be found in this paper: https://ijisrt.com/assets/upload/files/IJISRT23FEB247.pdf 

## (1) Eini - Mental Health Chatbot
### 1. Data Preprocessing and Tokenization
#### 1.1 Tokenization
The first step in building the Eini chatbot involves tokenization, where a sentence is split into an array of words or tokens. The tokenize function from the NLTK library is used for this purpose. Tokenization is essential as it provides the chatbot with a structured input that can be processed further.

```
def tokenize(sentence):
    return nltk.word_tokenize(sentence)
```

#### 1.2 Stemming
Stemming is the process of finding the root form of a word. The stem function, utilizing the Porter Stemmer algorithm, is applied to each word in the tokenized sentence. Stemming ensures that variations of words are represented by their common root, reducing the dimensionality of the input space.

```
def stem(word):
    return stemmer.stem(word.lower())
```

#### 1.3 Bag-of-Words Representation
The bag-of-words representation is created using the bag_of_words function. This function converts a tokenized sentence into a numerical array, where each element represents the presence or absence of a word in the sentence. This representation is crucial for training the neural network.

```
def bag_of_words(tokenized_sentence, words):
    sentence_words = [stem(word) for word in tokenized_sentence]
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in sentence_words: 
            bag[idx] = 1
    return bag
```

### 2. Neural Network Architecture
#### 2.1 Model Definition
The neural network architecture for Eini is defined using PyTorch. It consists of an input layer, two hidden layers, and an output layer. The ReLU activation function is applied after each hidden layer. The model is defined in the NeuralNet class.

```
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size) 
        self.l2 = nn.Linear(hidden_size, hidden_size) 
        self.l3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        return out2.2 
```

#### 2.2 Training the Model
The model is trained using a dataset created from the provided intents.json file. The dataset is preprocessed to create a bag-of-words representation for each pattern_sentence. The neural network is trained using cross-entropy loss and Adam optimization.

### 3. Chatbot Implementation
#### 3.1 Loading the Model
The saved model is loaded during the chatbot's runtime to make predictions.

```
with open('intents.json', 'r') as json_data:
    intents = json.load(json_data, strict=False)

FILE = "data.pth"
data = torch.load(FILE)
input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()
3.2 Chat Function
The get_response function takes a user's message, tokenizes it, creates a bag-of-words representation, and feeds it to the trained neural network. The model predicts the tag of the message, and if the confidence is above a certain threshold, a response is generated from the predefined intents.
def get_response(msg):
    sentence = tokenize(msg)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                return random.choice(intent['responses'])
    else:
        return "I do not understand..."
```

## (2) Harmonious-Minds Mental Health Assessor
### 1. Data Preprocessing and Cleaning
#### 1.1 Handling Missing Data
The first step involves handling missing data in the dataset. The variables "Timestamp," "comments," "state," "tech_company," and "no_employees" are dropped as they are not crucial for the model. Missing values in other columns are imputed based on the data type. For categorical variables, missing values are replaced with the string "NaN," while for numerical variables, the median value is used.

```
# Drop unnecessary columns
train_df = train_df.drop(['comments', 'state', 'Timestamp', 'tech_company', 'no_employees'], axis=1)

# Define default values for different data types
defaultInt = 0
defaultString = 'NaN'
defaultFloat = 0.0

# Create lists for different data types
intFeatures = ['Age']
stringFeatures = ['Gender', 'Country', 'self_employed', 'family_history', 'treatment', 'work_interfere', 'no_employees', 'remote_work', 'tech_company', 'anonymity', 'leave', 'mental_health_consequence', 'phys_health_consequence', 'coworkers', 'supervisor', 'mental_health_interview', 'phys_health_interview', 'mental_vs_physical', 'obs_consequence', 'benefits', 'care_options', 'wellness_program', 'seek_help']
floatFeatures = []

# Impute missing values based on data type
for feature in train_df:
    if feature in intFeatures:
        train_df[feature] = train_df[feature].fillna(defaultInt)
    elif feature in stringFeatures:
        train_df[feature] = train_df[feature].fillna(defaultString)
    elif feature in floatFeatures:
        train_df[feature] = train_df[feature].fillna(defaultFloat)
```

#### 1.2 Cleaning Gender Data
The "Gender" column is cleaned by converting all entries to lowercase and categorizing them into three main groups: male, female, and trans. This step ensures consistency and reduces variations in gender representations.
```
# Clean 'Gender' column
gender = train_df['Gender'].str.lower()

male_str = ["male", "m", "male-ish", "maile", "mal", "male (cis)", "make", "male ", "man","msle", "mail", "malr","cis man", "Cis Male", "cis male"]
trans_str = ["trans-female", "something kinda male?", "queer/she/they", "non-binary", "nah", "all", "enby", "fluid", "genderqueer", "androgyne", "agender", "male leaning androgynous", "guy (-ish) ^_^", "trans woman", "neuter", "female (trans)", "queer", "ostensibly male, unsure what that really means"]           
female_str = ["cis female", "f", "female", "woman",  "femake", "female ","cis-female/femme", "female (cis)", "femail"]

for (row, col) in train_df.iterrows():
    if str.lower(col.Gender) in male_str:
        train_df['Gender'].replace(to_replace=col.Gender, value='male', inplace=True)
    if str.lower(col.Gender) in female_str:
        train_df['Gender'].replace(to_replace=col.Gender, value='female', inplace=True)
    if str.lower(col.Gender) in trans_str:
        train_df['Gender'].replace(to_replace=col.Gender, value='trans', inplace=True)
```

#### 1.3 Handling Outliers in Age
Outliers in the "Age" column are addressed by replacing values below 18 and above 120 with the median age. Additionally, an "age_range" column is created to categorize individuals into age groups.

```
# Complete missing age with median
train_df['Age'].fillna(train_df['Age'].median(), inplace=True)

# Replace outliers in Age column
s = pd.Series(train_df['Age'])
s[s < 18] = train_df['Age'].median()
train_df['Age'] = s
s = pd.Series(train_df['Age'])
s[s > 120] = train_df['Age'].median()
train_df['Age'] = s

# Create age range categories
train_df['age_range'] = pd.cut(train_df['Age'], [0, 20, 30, 65, 100], labels=["0-20", "21-30", "31-65", "66-100"], include_lowest=True)
```

### 2. Feature Encoding
#### 2.1 Label Encoding
Label encoding is performed on categorical features to convert them into numerical values. Each unique category in a feature is assigned a numerical label.

```
labelDict = {}
for feature in train_df:
    le = preprocessing.LabelEncoder()
    le.fit(train_df[feature])
    le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    train_df[feature] = le.transform(train_df[feature])
    labelKey = 'label_' + feature
    labelValue = [*le_name_mapping]
    labelDict[labelKey] = labelValue
```

#### 2.2 Dropping Redundant Columns
The "Country" column is dropped from the dataset as it is not considered a crucial feature for the model.

```
# Drop 'Country' column
train_df = train_df.drop(['Country'], axis=1)
```

### 3. Model Training and Prediction
#### 3.1 Model Selection
Various machine learning models are considered for training the Harmonious-Minds Mental Health Assessor. Models such as Logistic Regression, Decision Tree, Random Forest, Extra Trees, Neural Network, Bagging, Naive Bayes, and Stacking are imported for experimentation.
#### 3.2 Feature Scaling
As a preprocessing step, feature scaling is performed using the Min-Max Scaler to normalize numerical features.
scaler = MinMaxScaler()

```
train_df['Age'] = scaler.fit_transform(train_df[['Age']])
```

#### 3.3 Model Training
The dataset is split into training and testing sets. A K-Nearest Neighbors (KNN) classifier is selected and trained on the training set. Hyperparameter tuning is performed using Randomized Search to find the optimal values.

![Untitled2781](https://github.com/AkshatKishore/HumanAI/assets/141553703/286ccfb1-4b7d-44e2-8d8e-89250ac56570)

#### 3.4 Making Predictions
Finally, the trained KNN model is used to make predictions for a given set of input features, including age, sex, family history, benefits, care options, anonymity, leave, work interference, coworkers, and remote work.

## (3) Palpitate - Heart Disease Forecaster
### 1. Data Preprocessing and Cleaning
#### 1.1 Loading and Splitting Data
The first step involves loading the heart disease dataset from the "heart.csv" file using Pandas. The dataset is then split into features (x_data) and the target variable (y).

```
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Load the heart disease dataset
df = pd.read_csv("heart.csv")

# Separate target variable and features
y = df['condition']
x_data = df.drop(['condition'], axis=1)
```

#### 1.2 Data Splitting for Training
The dataset is further split into training and testing sets using the train_test_split function from scikit-learn.

### 2. Model Selection and Training
#### 2.1 Naive Bayes Classifier
The chosen model for predicting heart disease is the Naive Bayes classifier. Specifically, the Gaussian Naive Bayes model is implemented for this classification task.

```
from sklearn.naive_bayes import GaussianNB
# Initialize Gaussian Naive Bayes model
nb = GaussianNB()

# Fit the model on the training data
nb.fit(x_train, y_train)
```

![23385Capture6](https://github.com/AkshatKishore/HumanAI/assets/141553703/ed7119ea-6130-4f3e-a4da-ec294305603b)

#### 2.2 Model Accuracy
The accuracy of the Naive Bayes model is calculated using the test set.

```
# Calculate accuracy on the test set
acc = nb.score(x_test, y_test) * 100
print("Accuracy of Naive Bayes: {:.2f}%".format(acc))
```

### 3. Making Predictions
#### 3.1 Input Data
The input features for prediction, including age, sex, chest pain type, resting blood pressure, serum cholesterol, fasting blood sugar, electrocardiographic results, maximum heart rate, exercise-induced angina, ST depression, slope of the peak exercise ST segment, number of major vessels, and thalassemia type, are collected.
#### 3.2 Input Data Encoding
Sex information is encoded into numerical values, where 1 represents Male and 0 represents Female.

```
if (sex == 'M') or (sex == 'Male') or (sex == 'Man') or (sex == 'male') or (sex == 'man') or (sex == 'm'):
    sex_input = 1
elif (sex == 'F') or (sex == 'Female') or (sex == 'Woman') or (sex == 'Girl') or (sex == 'female') or (sex == 'woman') or (sex == 'woman') or (sex == 'f'):
    sex_input = 0
```

#### 3.3 Making Predictions
The input data is structured and passed to the Naive Bayes model for making predictions.

```
inp = [[age, sex_input, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]]
result = nb.predict(inp)
```
