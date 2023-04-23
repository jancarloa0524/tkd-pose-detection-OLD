# First, read collected data and process
import pandas as pd # allows us to work with tabular data
from sklearn.model_selection import train_test_split # allows us to create our TTS
# Then, train the machine learning classification model
from sklearn.pipeline import make_pipeline # allows us to build a ML pipeline
from sklearn.preprocessing import StandardScaler # normalizes our data by subtract mean from Standard Deviation
# A bunch of algorithms we can use for classification
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
# finally, eval and serialize model
from sklearn.metrics import accuracy_score # accuracy metric for ml
import pickle # a popular library for saving models down to disks

# READ COLLECTED DATA AND PROCESS
df = pd.read_csv('coords.csv') # setting up dataframe

X = df.drop('class', axis=1) #features
y = df['class'] # target value

# Extract values from the TTS by passing through X, y, setting up our split (70//30), and random state that ensures we get similiar results whenever run
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=1234)

# train machine learning classification model
pipelines = {
    # These will be seperate ML Pipelines
    'lr':make_pipeline(StandardScaler(), LogisticRegression()),
    'rc':make_pipeline(StandardScaler(), RidgeClassifier()),
    'rf':make_pipeline(StandardScaler(), RandomForestClassifier()),
    'gb':make_pipeline(StandardScaler(), GradientBoostingClassifier())
}

fit_models = {}
for algo, pipeline in pipelines.items():
    model = pipeline.fit(X_train, y_train)
    fit_models[algo] = model
# We're looping through each of our pipelines. For each pipeline, are using the fit method (or the training method) to pass in our training data and store that trained model in our model variable. We take that model and store it in our `fit_models` dictionary using the algo key

# print(fit_models)

# eval and serialize model
for algo, model in fit_models.items():
    yhat = model.predict(X_test) # yhat is where you store predictions
    print(algo, accuracy_score(y_test, yhat)) # we are pinning our predictions against our test data to test performance

with open('tkd.pkl', 'wb') as f: # opening a file called body_lanugage.pkl in write binary mode. We dump our best model into the file. 
    pickle.dump(fit_models['lr'], f)