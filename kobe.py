import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import altair as alt
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss, make_scorer
from sklearn import tree
import cv2
from kobe_def import draw_court


st.title('Kobe Bryant Shots')
st.markdown('This is an analysis of all the shots Kobe Bryant has made throughout his career, including missed shots. This is identified with the shot_made_flag.')
st.markdown('**Data source**: https://www.kaggle.com/c/kobe-bryant-shot-selection/overview')
st.markdown('**Problem:** There are 5000 shot_made_flag values are missing, so the task is estimate the probability that the shot was a sucess.')
st.markdown('**Approach:** Clean the dataframe of redundant features, visualize the data for any outliers, engineer additional features that maybe useful, develop a binary classifier for the shot_made_flag probability.')

@st.cache(allow_output_mutation=True)
def load_data():
    data = pd.read_csv('PycharmProjects/Streamlit/data.csv')
    return data

data_load_state = st.text('Loading data...')
df = load_data()
data_load_state.text("Done! (using st.cache)")

if st.checkbox('Show raw data (first 100)'):
    st.subheader('Raw data (first 100)')
    st.write(df.head(100))
    
st.write(df['shot_zone_area'].unique())

st.markdown('The shape of the dataframe is: '+str(df.shape))

st.subheader('Cleaning the data')
st.markdown('* Kobe only played for the Los Angeles Lakers for his entire career, so the features "team_id" and "team_name" are not needed.')
st.markdown('* The "matchup" feature is redundant with "opponent", so this can also be dropped.')
st.markdown('* The "game_date" feature can be split by year, month, day')
st.markdown('* lat is linear with loc_y, and lon is linear with loc_x. Is is not necessary to keep both, so I will drop lat and lon.')
st.markdown('* Sort data by game_date and game_event_id.')
st.markdown('* Drop shot_id and game_event_id.')

# cleaning data
df['game_date'] = pd.to_datetime(df['game_date'])
df.sort_values(by=['game_date','game_event_id'], inplace=True)
df['year'] = df['game_date'].dt.year
df['month'] = df['game_date'].dt.month
df['day'] = df['game_date'].dt.day
df = df.drop(['team_id','team_name','matchup','game_date','lat','lon','shot_id','game_event_id'], axis=1)
df['period_minutes_remaining'] = df['minutes_remaining'] + df['seconds_remaining']/60


if st.checkbox('Show cleaned dataframe'):
    st.write(df.head())

seasons = df.season.unique().tolist()
    
start_season, end_season = st.select_slider(
     'Select a season',
     options=seasons,
     value=(seasons[0], seasons[-1]))
st.write('You selected seasons between', start_season, 'and', end_season)
index1 = seasons.index(start_season)
index2= seasons.index(end_season)

action_df = df[df['season'].isin(seasons[index1:index2])]['action_type']\
    .value_counts().reset_index()\
    .rename(columns={'index':'action type', 'action_type':'Shots Attempted'})

# st.write(alt.Chart(action_df).mark_bar().encode(
#     y=alt.X('action type', sort=None),
#     x='Shots Attempted',
# ))



# create shot location scatter plot
shotloc_df = df[df['season'].isin(seasons[index1:index2])]
fig, ax = plt.subplots()
ax.scatter(shotloc_df['loc_x'], shotloc_df['loc_y'], s=2, alpha =0.3, c=shotloc_df['shot_made_flag'])
plt.ylabel('loc_y')
plt.xlabel('loc_x')
plt.xlim(-250, 250)
plt.ylim(-50, 420)
ax.set_aspect(500/470)
ax.legend([0,1,2])
draw_court(outer_lines=True)
st.pyplot(fig)

fig, ax = plt.subplots(figsize=(7,10))
ax.barh(action_df['action type'].values, action_df['Shots Attempted'].values, align='center')
plt.xscale('log')
plt.xlabel('Shot Attempts')
st.pyplot(fig)


figure, axes = plt.subplots(2, 2,figsize=(10,6))
binsize=100
shotloc_df[shotloc_df['period']==1]['period_minutes_remaining'].hist(bins=binsize, ax=axes[0,0])
shotloc_df[shotloc_df['period']==2]['period_minutes_remaining'].hist(bins=binsize, ax=axes[0,1])
shotloc_df[shotloc_df['period']==3]['period_minutes_remaining'].hist(bins=binsize, ax=axes[1,0])
shotloc_df[shotloc_df['period']==4]['period_minutes_remaining'].hist(bins=binsize, ax=axes[1,1])
figure.text(0.5, 0.04, 'Minutes Elapsed in the Quarter', ha='center')
figure.text(0.05, 0.5, 'Average Number of Shots / '+str(np.round(12/binsize*60,1))+'s', va='center', rotation='vertical')
axes[0,0].legend('1st')
axes[0,1].legend('2nd')
axes[1,0].legend('3rd')
axes[1,1].legend('4th')
st.pyplot(figure)


#-----------------------------------------------
st.subheader('Machine Learning')
st.markdown('Models are evaluated by log-loss: -log P(yt|yp) = -(yt log(yp) + (1 - yt) log(1 - yp))')
LogLoss = make_scorer(log_loss, greater_is_better=False, needs_proba=True)

st.markdown('The data is separated into a training and test set based on if the shot_made_flag = null.')
st.markdown('Split the training data into 80% for training, 20% for validation.')
st.markdown('**Strategy:** Use grid search to find optimal hyperparameters and k-fold cross-validation.')

## Split into train/test set based on missing value
X_test = df[df['shot_made_flag'].isnull()].drop('shot_made_flag',axis=1)
X_train_full = df[df['shot_made_flag'].notnull()]
y_train_full = X_train_full.pop('shot_made_flag')

feature_names = df.drop('shot_made_flag', axis=1).columns.tolist()
cat_attribs = df.select_dtypes(include=object).columns.tolist()
num_attribs = df.select_dtypes(exclude=object).drop('shot_made_flag', axis=1).columns.tolist()

#--------------------------
st.subheader('Logistic Regression')

num_pipeline = Pipeline([
 ('imputer', SimpleImputer(strategy="median")) # even though there are no missing values
    ,('std_scaler', StandardScaler())
])
full_pipeline = ColumnTransformer([
("num", num_pipeline, num_attribs),
("cat", OrdinalEncoder(), cat_attribs),
])

X_train_transformed = full_pipeline.fit_transform(X_train_full)
X_test_transformed = full_pipeline.fit_transform(X_test)
X_train, X_valid, y_train, y_valid = train_test_split(X_train_transformed, y_train_full, test_size=0.2, random_state=42)


logclf = LogisticRegression(random_state=0)
scores = cross_val_score(logclf, X_train_transformed, y_train_full,
scoring=LogLoss, cv=10)
st.write(str(scores.mean()) + '+-' + str(scores.std()))


#-------------
st.subheader('k-Nearest Neighbours Classifier')

num_pipeline = Pipeline([
 ('imputer', SimpleImputer(strategy="median")) # even though there are no missing values
    ,('std_scaler', StandardScaler())
])
full_pipeline = ColumnTransformer([
("num", num_pipeline, num_attribs),
("cat", OneHotEncoder(), cat_attribs),
])

X_train_transformed = full_pipeline.fit_transform(X_train_full)
X_test_transformed = full_pipeline.fit_transform(X_test)
X_train, X_valid, y_train, y_valid = train_test_split(X_train_transformed, y_train_full, test_size=0.2, random_state=42)


knn = KNeighborsClassifier()
scores = cross_val_score(knn, X_train_transformed, y_train_full,
scoring=LogLoss, cv=10)
st.write(str(scores.mean()) + '+-' + str(scores.std()))



#------------------
st.subheader('Tree-Based Methods')
st.markdown('* Transform string features to ordinal encoding (0,1,2...).')
st.markdown('* Does not require feature scaling.')
st.markdown('* Finds the best feature and threshold that minimizes the impurity down the tree (gini).')



## Use an ordinal encoder to label the string features
encoder = OrdinalEncoder()

num_pipeline = Pipeline([
 ('imputer', SimpleImputer(strategy="median")) # even though there are no missing values
])
full_pipeline = ColumnTransformer([
("num", num_pipeline, num_attribs),
("cat", encoder, cat_attribs),
])

X_train_transformed = full_pipeline.fit_transform(X_train_full)
X_test_transformed = full_pipeline.fit_transform(X_test)
X_train, X_valid, y_train, y_valid = train_test_split(X_train_transformed, y_train_full, test_size=0.2, random_state=42)





#------------------------------
st.subheader('Decision Tree Classifier')

param_grid = [
{'max_depth': [2,3,4,5], 'max_features':[2,4,8,16,None], 'max_leaf_nodes':[2,3,4,None]}
]
st.markdown('Grid Search Parameters with ' + str(param_grid))
dtree = DecisionTreeClassifier(random_state=0)
dtreeCV = GridSearchCV(dtree, param_grid, cv=10,
                       scoring=LogLoss,
                       return_train_score=True)
dtreeCV.fit(X_train_transformed, y_train_full)
st.text(dtreeCV.best_params_)
st.text([dtreeCV.best_score_, dtreeCV.best_estimator_])


fig = plt.figure(figsize=(7,7))
_ = tree.plot_tree(dtreeCV.best_estimator_, 
                   feature_names=num_attribs+cat_attribs,  
                   class_names=['0','1'],
                   filled=True)
fig.canvas.draw()
figdata = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
figdata = figdata.reshape(fig.canvas.get_width_height()[::-1] + (3,))
st.image(figdata)


#-----------------------------------------------
st.subheader('Random Forest Classifier')
st.markdown('An ensemble of decision trees where subsets of data is divided amongst tree. The ensemble votes for the most likely outcome.')

param_grid = [
{'max_depth': [2,4,8], 'max_features':[2,4,8,16]}
]
st.markdown('Grid Search Parameters with ' + str(param_grid))
RFclf = RandomForestClassifier(random_state=0, n_estimators=20)
RFclfCV = GridSearchCV(RFclf, param_grid, cv=10,
                       scoring=LogLoss,
                       return_train_score=True)
RFclfCV.fit(X_train_transformed, y_train_full)
st.text(RFclfCV.best_params_)
st.text([RFclfCV.best_score_, RFclfCV.best_estimator_])