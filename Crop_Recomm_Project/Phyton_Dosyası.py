import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import GridSearchCV, cross_validate, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
sns.set_style('darkgrid')
from sklearn.model_selection import train_test_split ,GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# Veri seti okutulur.
df = pd.read_csv(r"C:\Users\oskac\Desktop\pyt_osman\YENİ\datasets\Crop_recommendation.csv")


# check_df fonksiyonu ile veri hızlıca bir analiz edilir.
def check_df(dataframe, head = 5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.describe([0, 0.05, 0.50, 0.95, 0.99, 1]).T)
    print("##################### nunique #####################")
    print(dataframe.nunique())
check_df(df)

# dataset target variable: label. her değer için eşit gözlem olması ML için pozitif
df.label.value_counts()


def outlier_thresholds(df, col_name, q1=0.05, q3=0.95):
    quartile1 = df[col_name].quantile(q1)
    quartile3 = df[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

outlier_thresholds(df, "K")


def check_outlier(df, col_name):
    low_limit, up_limit = outlier_thresholds(df, col_name, q1=0.05, q3=0.95)
    if df[(df[col_name] > up_limit) | (df[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

check_outlier(df, "K")


# Target variable hariç diğer variable'ların correlasyonunun incelenmesi:
# K ve P arasında önemli ölçüde korrelasyon gözlemleniyor
plt.figure(figsize=(6,4))
sns.heatmap(df.corr(),cmap='Blues',annot=True )

# K ve P değerleri arasında yüksek corr. olması
# nedeniyle corr'a graph'da detay bakılması
#sonuç: değişkenlerden birini silmeyi gerektirecek kadar corr yoktur.
plt.figure(figsize=(12,5))
plt.subplot(1, 2, 1)
# sns.distplot(df_setosa['sepal_length'],kde=True,color='green',bins=20,hist_kws={'alpha':0.3})
sns.distplot(df['K'],color="purple",bins=15,hist_kws={'alpha':0.2})
plt.subplot(1, 2, 2)
sns.distplot(df['P'],color="green",bins=15,hist_kws={'alpha':0.2})

#  categoric variable ile numeric variable corr. bakılması

df["label"] = df["label"].astype('category')
df.dtypes

# categoric variable'ı numeric'e dönüştürmek
dff = df.drop("label", axis='columns')
dff["label_encode"] = df["label"].cat.codes
plt.figure(figsize=(10,8))
sns.heatmap(dff.corr(),cmap='Greens',annot=True )


# N, P, K value karşılaştırmaları
import plotly.graph_objects as go

crop_summary = pd.pivot_table(df,index=['label'],aggfunc='mean')

fig = go.Figure()
fig.add_trace(go.Bar(
    x=crop_summary.index,
    y=crop_summary['N'],
    name='Nitrogen',
    marker_color='indianred'
))
fig.add_trace(go.Bar(
    x=crop_summary.index,
    y=crop_summary['P'],
    name='Phosphorous',
    marker_color='lightsalmon'
))
fig.add_trace(go.Bar(
    x=crop_summary.index,
    y=crop_summary['K'],
    name='Potash',
    marker_color='crimson'
))

fig.update_layout(title="N, P, K values comparison between crops",
                  plot_bgcolor='white',
                  barmode='group',
                  xaxis_tickangle=-45)

fig.show()


# In[49]:


# temp., humid., rainfall value karşılaştırmaları

fig = go.Figure()
fig.add_trace(go.Bar(
    x=crop_summary.index,
    y=crop_summary['temperature'],
    name='temperature',
    marker_color='red'
))
fig.add_trace(go.Bar(
    x=crop_summary.index,
    y=crop_summary['humidity'],
    name='humidity',
    marker_color='orange'
))
fig.add_trace(go.Bar(
    x=crop_summary.index,
    y=crop_summary['rainfall'],
    name='rainfall',
    marker_color='lightblue'
))

fig.update_layout(title="temp., humid., rainfall values comparison between crops",
                  plot_bgcolor='white',
                  barmode='group',
                  xaxis_tickangle=-45)
fig.show()


# Modelleme

y = df["label"]
X = df.drop(["label"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 152)

kn_classifier = KNeighborsClassifier()
kn_classifier.fit(X_train,y_train)
y_pred =kn_classifier.predict(X_test)
pred_kn = kn_classifier.predict(X_test)
print('Training set score: {:.4f}'.format(kn_classifier.score(X_train, y_train)))
print('Test set score: {:.4f}'.format(kn_classifier.score(X_test, y_test)))

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(15,10))
sns.heatmap(cm, annot=True, fmt=".0f", linewidths=.5, square = True);
plt.ylabel('Actual label');
plt.xlabel('Predicted label');
all_sample_title = 'Confusion Matrix - score : {:.4f}'.format(kn_classifier.score(X_test, y_test))
plt.title(all_sample_title, size = 15);
plt.show(block=True)


df.head(5)

newdata= kn_classifier.predict([[60,55,44,23.004459,82.320763,7.840207,	263.964248]])
newdata

a= kn_classifier.predict([[75, 75, 55, 25.5, 86.2435, 8.5436, 185.5353]])
a

b= kn_classifier.predict([[68, 90, 40, 25.5, 58.1518, 6.5191, 120.155]])
b

print(newdata, a, b)








