<h2 align="center"> <a href="https://upgini.com/">Upgini</a> : automated feature discovery & enrichment library for supervised machine learning on tabular data </h2>
<p align="center"> <b>Automatically find and enrich ML model with relevant <i>external</i> features from scraped data and public datasets to improve machine learning model accuracy </b> </p>
<p align="center">
	<br />
    <a href="https://colab.research.google.com/github/upgini/upgini/blob/main/notebooks/kaggle_example.ipynb"><strong>Live DEMO in Colab »</strong></a> |
    <a href="https://upgini.com/">Upgini.com</a> |
    <a href="https://profile.upgini.com">Sign In</a> |
    <a href="https://upgini.slack.com/messages/C02MW49ADSN">Slack Community</a> 
 </p>

[![license](https://img.shields.io/badge/license-BSD--3%20Clause-green)](/LICENSE)
[![Python version](https://img.shields.io/badge/python_version-3.8-red?logo=python&logoColor=white)](https://www.python.org/downloads/release/python-380/)
[![PyPI Latest Release](https://img.shields.io/badge/pypi-v0.10.0-blue?logo=pypi&logoColor=white)](https://pypi.org/project/upgini/)
[![stability-release-candidate](https://img.shields.io/badge/stability-pre--release-br?logo=circleci&logoColor=white)](https://pypi.org/project/upgini/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg?logo=python&logoColor=white)](https://github.com/psf/black)
[![Slack upgini](https://img.shields.io/badge/slack-@upgini-orange.svg?logo=slack)](https://upgini.slack.com/messages/C02MW49ADSN)
[![Downloads](https://pepy.tech/badge/upgini)](https://pepy.tech/project/upgini)
## ❔ Overview

**Upgini** is a Python library for an automated features search to boost accuracy of supervised ML models on tabular data. It enriches your dataset with intelligently crafted features from a broad range of curated data sources, including public datasets and scraped data. The search is conducted for any combination of public IDs contained in your tabular dateset: IP, date, etc.
Only features that could improve the prediction power of your ML model are returned.  
**Motivation:** for most supervised ML models external data & features boost accuracy significantly better than any hyperparameters tuning. But lack of automated and time-efficient search tools for external data blocks massive adoption of external features in ML pipelines.  
We want radically simplify features search and delivery for ML pipelines to make external data a standard approach. Like a hyperparameter tuning for machine learning nowadays.

## 🚀 Awesome features
⭐️ Automatically find only features that *give accuracy improvement for ML algorithm* according to metrics: ROC AUC, RMSE, Accuracy. Not just correlated with target variable data or features, which 9 out of 10 cases gives zero accuracy improvement for production ML cases  
⭐️ Calculate *accuracy metrics and uplifts* if you'll enrich your existing ML model with found external features, right in search results   
⭐️ Check the stability of accuracy gain from external data on out-of-time intervals and verification datasets. Mitigate risks of unstable external data dependencies in ML pipelines   
⭐️ Scikit-learn compatible interface for quick data integration with your existing ML pipelines  
⭐️ Curated and updated data sources, including public datasets and scraped data  
⭐️ Support for several search key types (including <i>**SHA256 hashed email, IPv4, phone, date/datetime**</i>), more to come...  
⭐️ Supported supervised ML tasks:  
  - ☑️ [binary classification](https://en.wikipedia.org/wiki/Binary_classification)  
  - ☑️ [multiclass classification](https://en.wikipedia.org/wiki/Multiclass_classification)  
  - ☑️ [regression](https://en.wikipedia.org/wiki/Regression_analysis)  
  - 🔜 [time series prediction](https://en.wikipedia.org/wiki/Time_series#Prediction_and_forecasting)   
  - 🔜 [recommender system](https://en.wikipedia.org/wiki/Recommender_system)  
## 🏁 Quick start with kaggle example

### 🏎 Live Demo with kaggle competition data
Live Demo notebook [kaggle_example.ipynb](https://github.com/upgini/upgini/blob/main/notebooks/kaggle_example.ipynb) inside your browser:

[![Open example in google colab](https://img.shields.io/badge/run_example_in-colab-blue?style=for-the-badge&logo=googlecolab)](https://colab.research.google.com/github/upgini/upgini/blob/main/notebooks/kaggle_example.ipynb)
&nbsp;
[![Open in Binder](https://img.shields.io/badge/run_example_in-mybinder-red.svg?style=for-the-badge&logo=data:image/png;base64,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)](https://mybinder.org/v2/gh/upgini/upgini/HEAD?labpath=notebooks%2Fkaggle_example.ipynb)
&nbsp;
<!--
[![Open example in Gitpod](https://img.shields.io/badge/run_example_in-gitpod-orange?style=for-the-badge&logo=gitpod)](https://gitpod.io/#/github.com/upgini/upgini)
-->

### 🐍 Install from PyPI
```python
%pip install -Uq upgini catboost
```
### 🐳 Docker-way
Clone `$ git clone https://github.com/upgini/upgini` or download upgini git repo locally and follow steps below to build docker container 👇  
Build docker image  
 - ... from cloned git repo:
```bash
cd upgini
docker build -t upgini .
```
 - ...or directly from GitHub:
```bash
DOCKER_BUILDKIT=0 docker build -t upgini git@github.com:upgini/upgini.git#main
```
Run docker image:
```bash
docker run -p 8888:8888 upgini
```
Open http://localhost:8888?token=<your_token_from_console_output> in your browser

#### *Kaggle notebook*
Jupyter notebook with a kaggle example: [kaggle_example.ipynb](https://github.com/upgini/upgini/blob/main/notebooks/kaggle_example.ipynb). The problem being solved is a Kaggle competition [Store Item Demand Forecasting Challenge](https://www.kaggle.com/c/demand-forecasting-kernels-only). The goal is to predict future sales of different goods in different stores based on a 5-year history of sales. The evaluation metric is [SMAPE](https://en.wikipedia.org/wiki/Symmetric_mean_absolute_percentage_error).

Competition dataset was splited into train (2013-2016 year) and test (2017 year) parts. `FeaturesEnricher` was fitted on train part. And both datasets  were enriched with external features. Finally, ML algorithm was fitted both of the initial and the enriched datasets to compare accuracy improvement. With a solid improvement of the evaluation metric achieved by the enriched ML model.

## 💻 How it works?

### 1. 🔑 Get access - API key
For full access beyond demo example, you'll need API key from User profile page https://profile.upgini.com  
Pass API key via `api_key` parameter in [`FeaturesEnricher` class constructor](#4--start-your-first-data-search) or export as environment variable:  
... in python  
```python
import os
os.environ["UPGINI_API_KEY"] = "your_long_string_api_key_goes_here"
```
... in bash/zsh
```bash
export UPGINI_API_KEY = "your_long_string_api_key_goes_here"
```  
### 2. 💡 Reuse existing labeled training datasets for search
To simplify things, you can reuse your existing labeled training datasets "as is" to initiate the search. Under the hood, we'll search for relevant data using:
- *[search keys](#-search-key-types-we-support-more-is-coming)* from training dataset to match records from potential external data sources and features
- *labels* from training dataset to estimate relevancy of feature or dataset for your ML task and calculate metrics  
- *your features* from training dataset to find external datasets and features only give accuracy improvement to your existing data and estimate accuracy uplift ([optional](#-optional-find-datasets-and-features-only-give-accuracy-gain-to-your-existing-data-in-the-ml-model))  

Load training dataset into pandas dataframe and separate features' columns from label column:  
```python
import pandas as pd
# labeled training dataset - customer_churn_prediction_train.csv
train_df = pd.read_csv("customer_churn_prediction_train.csv")
train_ids_and_features = train_df.drop(columns="label")
train_label = train_df["label"]
```
### 3. 🔦 Choose at least one column as a search key
*Search keys* columns will be used to match records from all potential external data sources 👓. Define at least one search key with `FeaturesEnricher` class initialization.  
```python
from upgini import FeaturesEnricher, SearchKey
enricher = FeaturesEnricher (
    search_keys={"subscription_activation_date": SearchKey.DATE},
    keep_input=True )
```
#### ✨ Search key types we support (more is coming!)
Our team works hard to introduce new search key types, currently we support:
<table style="table-layout: fixed; text-align: left">
  <tr>
    <th> Search Key<br/>Meaning Type </th>
    <th> Description </th>
    <th> Example </th>
  </tr>
  <tr>
    <td> SearchKey.EMAIL </td>
    <td> e-mail </td>
    <td> <tt>support@upgini.com </tt> </td>
  </tr>
  <tr>
    <td> SearchKey.HEM </td>
    <td>  <tt>sha256(lowercase(email)) </tt> </td>
    <td> <tt>0e2dfefcddc929933dcec9a5c7db7b172482814e63c80b8460b36a791384e955 </tt> </td>
  </tr>
  <tr>
    <td> SearchKey.IP </td>
    <td> IP address (version 4) </td>
    <td> <tt>192.168.0.1 </tt> </td>
  </tr>
  <tr>
    <td> SearchKey.PHONE </td>
    <td> phone number, <a href="https://en.wikipedia.org/wiki/E.164">E.164 standard</a> </td>
    <td> <tt>443451925138 </tt> </td>
  </tr>
  <tr>
    <td> SearchKey.DATE </td>
    <td> date </td>
    <td> 
      <tt>2020-02-12 </tt>&nbsp;(<a href="https://en.wikipedia.org/wiki/ISO_8601">ISO-8601 standard</a>) 
      <br/> <tt>12.02.2020 </tt>&nbsp;(non standard notation) 
    </td>
  </tr>
  <tr>
    <td> SearchKey.DATETIME </td>
    <td> datetime </td>
    <td> <tt>2020-02-12 12:46:18 </tt> <br/> <tt>12:46:18 12.02.2020 </tt> <br/> <tt>unixtimestamp </tt> </td>
  </tr>
</table>

#### ⚠️ Requirements for search initialization dataset  
We do dataset verification and cleaning under the hood, but still there are some requirements to follow:  
- Pandas dataframe representation  
- Correct label column types: integers or strings for binary and multiclass lables, floats for regression  
- At least one column defined as a [search key](#-search-key-types-we-support-more-is-coming)  
- Min size after deduplication by search key column and NAs removal: *1000 records*  
- Max size after deduplication by search key column and NAs removal: *1 000 000 records*  
### 4. 🔍 Start your first data search!
The main abstraction you interact is `FeaturesEnricher`. `FeaturesEnricher` is scikit-learn compatible estimator, so you can easily add it into your existing ML pipelines. First, create instance of the `FeaturesEnricher` class. Once it created call  
- `fit` to search relevant datasets & features  
- than `transform` to enrich your dataset with features from search result  

Let's try it out!
```python
import pandas as pd
from upgini import FeaturesEnricher, SearchKey

# load labeled training dataset to initiate search
train_df = pd.read_csv("customer_churn_prediction_train.csv")
train_features = train_df.drop(columns="label")
train_target = train_df["label"]

# now we're going to create `FeaturesEnricher` class
# if you still didn't define UPGINI_API_KEY env variable - not a problem, you can do it via `api_key`
enricher = FeaturesEnricher(
    search_keys={"subscription_activation_date": SearchKey.DATE},
    keep_input=True,
    api_key="your_long_string_api_key_goes_here"
)

# everything is ready to fit! For 200к records fitting should take around 10 minutes,
# but don't worry - we'll send email notification. Accuracy metrics of trained model and uplifts
# will be shown automaticly
enricher.fit(train_ids_and_features, train_target)

```

That's all). We have fitted `FeaturesEnricher` and any pandas dataframe, with exactly the same data schema, can be enriched with features from search results. Use `transform` method, and let magic to do the rest 🪄

```python
# load dataset for enrichment
test_df = pd.read_csv("test.csv")
test_ids_and_features = test_df.drop(columns="target")
# enrich it!
enriched_test_features = enricher.transform(test_ids_and_features)
enriched_test_features.head()
```
You can get more details about `FeaturesEnricher` in runtime using docstrings, for example, via `help(FeaturesEnricher)` or `help(FeaturesEnricher.fit)`.

### ✅ Optional: find datasets and features only give accuracy gain to your existing data in the ML model
If you already have a trained ML model, based on internal features or other external data sources, you can specifically search new datasets & features only give accuracy gain "on top" of them.  
Just leave all these existing features in the labeled training dataset and Upgini library automatically use them as a baseline ML model to calculate accuracy metric uplift. And won't return any features that might not give an accuracy gain to the existing ML model feature set.  

### ✅ Optional: check stability of ML accuracy gain from search result datasets & features
You can validate data quality from your search result on out-of-time dataset using `eval_set` parameter. Let's do that:
```python
# load train dataset
train_df = pd.read_csv("train.csv")
train_ids_and_features = train_df.drop(columns="target")
train_target = train_df["target"]

# load out-of-time validation dataset
eval_df = pd.read_csv("validation.csv")
eval_ids_and_features = eval_df.drop(columns="eval_target")
eval_target = eval_df["eval_target"]
# create FeaturesEnricher
enricher = FeaturesEnricher(
    search_keys={"registration_date": SearchKey.DATE},
    keep_input=True
)

# now we fit WITH eval_set parameter to calculate accuracy metrics on OOT dataset.
# the output will contain quality metrics for both the training data set and
# the eval set (validation OOT data set)
enricher.fit(
  train_ids_and_features,
  train_target,
  eval_set = [(eval_ids_and_features, eval_target)]
)
```
#### ⚠️ Requirements for out-of-time dataset  
- Same data schema as for search initialization dataset  
- Pandas dataframe representation  

### 🧹 Search dataset validation
We validate and clean search initialization dataset uder the hood:  
✂️ Check you *search keys* columns format   
✂️ Check dataset for full row duplicates. If we find any, we remove duplicated rows and make a note on share of row duplicates  
✂️ Check inconsistent labels  - rows with the same record keys (not search keys!) but different labels, we remove them and make a note on share of row duplicates
### 🆙 Accuracy and uplift metrics calculations
We calculate all the accuracy metrics and uplifts for non-linear machine learning algorithms, like gradient boosting or neural networks. If your external data consumer is a linear ML algorithm (like log regression), you might notice different accuracy metrics after data enrichment.  

## 💸 Why it's a paid service? Can I use it for free?
The short answer is Yes! **We do have two options for that** 🤓  
Let us explain. This is a part-time project for our small team, but as you might know, search is a very infrastructure-intensive service. We pay infrustructure cost for *every search request* generated on the platform, as we mostly use serverless components under the hood. Both storage and compute.  
To cover these run costs we introduce paid plans with a certain amount of search requests, which we hope will be affordable for most of the data scientists & developers in the community.  
#### First option. Participate in beta testing
Now service is still in a beta stage, so *registered beta testers* will get an **80USD credits for 6 months**. Feel free to start with the registration form 👉 [here](https://profile.upgini.com/access-for-beta-testers)  Please note that number of slots for beta testing is limited and we wont' be able to handle all the requests.  
#### Second option. Share license-free data with community
If you have ANY data which you might consider as royalty and license-free ([Open Data](http://opendatahandbook.org/guide/en/what-is-open-data/)) and potentially valuable for supervised ML applications, we'll be happy to give **free individual access** in exchange for **sharing this data with community**.  
Just upload your data sample right from Jupyter. We will check your data sharing proposal and get back to you ASAP:
```python
import pandas as pd
from upgini import SearchKey
from upgini.ads import upload_user_ads
import os
os.environ["UPGINI_API_KEY"] = "your_long_string_api_key_goes_here"
#you can define custom search key which might not be supported yet, just use SearchKey.CUSTOM_KEY type
sample_df = pd.read_csv("path_to_data_sample_file")
upload_user_ads("test", sample_df, {
    "city": SearchKey.CUSTOM_KEY, "stats_date": SearchKey.DATE
})
```
## 🛠 Getting Help & Community
Requests and support, in preferred order  
[![Claim help in slack](https://img.shields.io/badge/slack-@upgini-orange.svg?style=for-the-badge&logo=slack)](https://upgini.slack.com/messages/C02MW49ADSN)
[![Open GitHub issue](https://img.shields.io/badge/open%20issue%20on-github-blue?style=for-the-badge&logo=github)](https://github.com/upgini/upgini/issues)  
Please try to create bug reports that are:
- _Reproducible._ Include steps to reproduce the problem.
- _Specific._ Include as much detail as possible: which Python version, what environment, etc.
- _Unique._ Do not duplicate existing opened issues.
- _Scoped to a Single Bug._ One bug per report.

## 🧩 Contributing
We are a **very** small team and this is a part-time project for us, thus most probably we won't be able:
 - implement ALL the data delivery and integration interfaces for most common ML stacks and frameworks
 - implement ALL data verification and normalization capabilities for different types of search keys (we just started with current 4)

And we might need some help from community)
So, we'll be happy about every **pull request** you open and **issue** you find to make this library **more awesome**. Please note that it might sometimes take us a while to get back to you.
**For major changes**, please open an issue first to discuss what you would like to change
#### Developing
Some convinient ways to start contributing are:  
⚙️ **Visual Studio Code** [![Open in Visual Studio Code](https://open.vscode.dev/badges/open-in-vscode.svg)](https://open.vscode.dev/upgini/upgini) You can remotely open this repo in VS Code without cloning or automaticaly clone and open it inside a docker container.  
⚙️ **Gitpod** [![Gitpod Ready-to-Code](https://img.shields.io/badge/Gitpod-Ready--to--Code-blue?logo=gitpod)](https://gitpod.io/#https://github.com/upgini/upgini) You can use Gitpod to launch a fully functional development environment right in your browser.

## 🔗 Useful links
- [Quick start guide](https://upgini.com/#quick-start)
- [Kaggle example notebook](https://github.com/upgini/upgini/blob/main/notebooks/kaggle_example.ipynb)
- [Project on PyPI](https://pypi.org/project/upgini)
- [Get API Key](https://profile.upgini.com)

<sup>😔 Found mistype or a bug in code snippet? Our bad! <a href="https://github.com/upgini/upgini/issues/new?assignees=&title=readme%2Fbug">
Please report it here.</a></sup>
