# All ML Projects:

# Rossman-Retail-Sales-Prediction

Rossmann operates over 3,000 drug stores in 7 European countries.Store sales are influenced by many factors, including promotions, competition, school and state holidays, seasonality, and locality. With thousands of individual managers predicting sales based on their unique circumstances, the accuracy of results can be quite varied.


**Problem Statement**

Rossmann operates over 3,000 drug stores in 7 European countries. Currently, Rossmann store managers are tasked with predicting their daily sales for up to six weeks in advance. Store sales are influenced by many factors, including promotions, competition, school and state holidays, seasonality, and locality. With thousands of individual managers predicting sales based on their unique circumstances, the accuracy of results can be quite varied. You are provided with historical sales data for 1,115 Rossmann stores. The task is to forecast the "Sales" column for the test set. Note that some stores in the dataset were temporarily closed for refurbishment.

**About Data set**

The given dataset is a dataset from Rossmen industry who operates over 3,000 drug stores in 7 European countries, and we have to analysis the sales of the stores and what are the factores affecting the sales. in our given dataset has 1017209 rows and 18 columnn and There are some missing values and there is no duplicate values in the dataset

Dataset link : https://www.kaggle.com/competitions/rossmann-store-sales/data

**Variables Description**

* #### Id - an Id that represents a (Store, Date) duple within the test set
* #### Store - a unique Id for each store
* #### Sales - the turnover for any given day (this is what you are predicting)
* #### Customers - the number of customers on a given day
* #### Open - an indicator for whether the store was open: 0 = closed, 1 = open
* #### StateHoliday - indicates a state holiday. Normally all stores, with few exceptions, are closed on state holidays. Note that all schools are closed on public holidays and weekends. a = public holiday, b = Easter holiday, c = Christmas, 0 = None
* #### SchoolHoliday - indicates if the (Store, Date) was affected by the closure of public schools
* #### StoreType - differentiates between 4 different store models: a, b, c, d
* #### Assortment - describes an assortment level: a = basic, b = extra, c = extended
* #### CompetitionDistance - distance in meters to the nearest competitor store
* #### CompetitionOpenSince[Month/Year] - gives the approximate year and month of the time the nearest competitor was opened
* #### Promo - indicates whether a store is running a promo on that day
* #### Promo2 - Promo2 is a continuing and consecutive promotion for some stores: 0 = store is not participating, 1 = store is participating
* #### Promo2Since[Year/Week] - describes the year and calendar week when the store started participating in Promo2
* #### PromoInterval - describes the consecutive intervals Promo2 is started, naming the months the promotion is started anew. E.g. "Feb,May,Aug,Nov" means each round starts in February, May, August, November of any given year for that store


**Data Wrangling and Visualization**

Data wrangling, also known as data preprocessing or data cleaning, refers to the process of transforming and preparing raw data into a suitable format for analysis. It involves cleaning, organizing, and transforming data to ensure its quality, consistency, and compatibility with the analytical tasks at hand.

Data visualization is the process of representing data and information graphically to facilitate understanding, exploration, and communication of insights. It involves creating visual representations such as charts, graphs, maps, and infographics to effectively convey patterns, trends, relationships, and comparisons present in the data.


**Hypothesis Testing**

*  Normality test using Shapiro-Wilk Test : tests If data is normally distributed
*  we used Independent Sample T-test to obtained P-value this test used to compare means of two sample. we imported ttest library from scipy to perform this test.here p value is less than 0.05 so we rejected the null hypothesis so our assumptionj becomes true that average sale of 2014 is greater than last year

 **Feature Engineering & Data Pre-processing**
 * Handling Missing Values
 *  Handling Outliers
 *  Categorical Encoding
 *  Feature Manipulation
 *  Data Transformation
 *  Data Scaling


**ML Model Implementation**

We have successfully implemented a diverse range of models, **including linear regression**, **decision tree regression**, **random forest regressor**, as well as **lasso and ridge regression**. After thorough evaluation, it became evident that the random forest regressor outperformed all the other models, exhibiting the highest level of predictive accuracy.

In our pursuit of even better results, we proactively worked on enhancing our model's performance by employing advanced techniques such as hyperparameter tuning and cross-validation. These practices allowed us to fine-tune the model's parameters and ensure it generalizes well to unseen data, ultimately leading to improved accuracy in our predictions.

As we have seen above that, Random Forest Regressor is performing the best with the accuracy of 93.7% followed by Decison Tree Regressor with accuracy of 87.3% so here we are choosing Random Forest Regressor for best prediction.

![image](https://github.com/irfan7210/Rossman-Retail-Sales-Prediction/assets/113547056/6193dcb0-c3fe-4f70-8e92-a2bb04820e64)

Random Forest is a supervised learning algorithm that can be used for both classification and regression tasks. A Random Forest regressor is a specific type of Random Forest that is used for regression tasks, which involve predicting a continuous output value (such as a price or temperature) rather than a discrete class label.

The algorithm works by creating an ensemble of decision trees, where each tree is trained on a random subset of the data. The final output is then obtained by averaging the predictions of all the trees. This helps to reduce overfitting and improve the overall performance of the model.

Random Forest regressor is known to be a very powerful algorithm that can handle high-dimensional data and a large number of input features. It is also relatively easy to use and interpret. It has several parameters that can be adjusted to optimize its performance, such as the number of trees in the ensemble, the maximum depth of each tree, and the minimum number of samples required to split a node.

**About model and Performance**

**Linear Regression**

It gives 79% accuracy on the test data (i.e r2_score is 0.790007 for test data). And we get 81% accuracy (r2_score = 0.8155) after using the cross-validation.

**Decision Tree**
It gives 87% accuracy on the test data (i.e r2_score is 0.8720 for test data). And 88% (i.e. r2_score = 0.8843) after tunning hyperparameter.

**Random Forest Regressor**

The random forest regressiore provides 93.77% accuracy because it is gives 0.9377 r2_score.

**L1 and L2 Regularization**
Lasso gives 14% accuracy while ridge provides approximately 79% accuracy. And elastic net is worst model for this data because it gives 9% accuracy.

After hyperparameter tunning, Lasso gives 79% (approximately) accuracy.

Among the all regression models, it is clear that Random Forest Regressor is giving the best result with the accuracy of 93.6% followed by Decison Tree Regressor with accuracy of 87.2%. So, we will use the random forest regressor to predict the sales.


**Conclusion:**


The Rossmann Store Sales challenge was a fascinating puzzle for data experts. We had to carefully choose which store features to use, and by looking at trends in the data, we picked the best ones. Using multiple learning methods together helped make our predictions more accurate, especially because sales depend on how many customers come in.

A method called "boosted trees" worked really well in our machine learning. But we had to be careful because fancy techniques could make our predictions too specific to the training data. For predicting future sales, we found that a model called "Random Forest Regressor" was super accurate, with about 93.77% correct predictions. This model looks at relationships in the data and got an impressive score of 0.9377.

Since the data changes over time, we can't just randomly pick parts to learn from. We need a smarter approach.


# Cardiovascular-Risk-Prediction
This project aims to use data from the ongoing cardiovascular study on residents of Framingham, Massachusetts to predict the 10-year risk of future coronary heart disease (CHD) for patients
The dataset consists of over 4,000 records and 15 attributes, including demographic, behavioral, and medical risk factors. The goal of this project is to develop a predictive model that accurately classifies patients based on their risk of CHD.

**Problem Statement**
Despite advances in medical technology, coronary heart disease remains a leading cause of death worldwide. The early detection of CHD risk is crucial for preventing and mitigating its impact. The current cardiovascular study on the residents of Framingham, Massachusetts provides an opportunity to use data to identify patients at risk of CHD. However, with over 4,000 records and 15 attributes, it is difficult to manually identify patients who are at high risk. This project aims to address this challenge by developing a predictive model that accurately classifies patients based on their risk of CHD. This will help to improve the early detection and prevention of CHD, reducing its impact on patients and the healthcare system.

**About Data Set**

**Demographic**
* Sex: male or female("M" or "F")

* Age: Age of the patient;(Continuous - Although the recorded ages have been truncated to whole numbers, the concept of age is continuous)

**Behavioral**
* is_smoking: whether or not the patient is a current smoker ("YES" or "NO")
* Cigs Per Day: the number of cigarettes that the person smoked on average in one day.(can be
considered continuous as one can have any number of cigarettes, even half a cigarette.)
Medical( history)
* BP Meds: whether or not the patient was on blood pressure medication (Nominal)
* Prevalent Stroke: whether or not the patient had previously had a stroke (Nominal)
* Prevalent Hyp: whether or not the patient was hypertensive (Nominal)
* Diabetes: whether or not the patient had diabetes (Nominal)
Medical(current)
* Tot Chol: total cholesterol level (Continuous)
* Sys BP: systolic blood pressure (Continuous)
* Dia BP: diastolic blood pressure (Continuous)
* BMI: Body Mass Index (Continuous)
* Heart Rate: heart rate (Continuous - In medical research, variables such as heart rate though in
fact discrete, yet are considered continuous because of large number of possible values.)
* Glucose: glucose level (Continuous)
Predict variable (desired target)
* 10-year risk of coronary heart disease CHD(binary: “1”, means “Yes”, “0” means “No”) -

**Data Vizualization, Storytelling & Experimenting with charts : Understand the relationships between variables**

Data wrangling, also known as data preprocessing or data cleaning, refers to the process of transforming and preparing raw data into a suitable format for analysis. It involves cleaning, organizing, and transforming data to ensure its quality, consistency, and compatibility with the analytical tasks at hand.

Data visualization is the process of representing data and information graphically to facilitate understanding, exploration, and communication of insights. It involves creating visual representations such as charts, graphs, maps, and infographics to effectively convey patterns, trends, relationships, and comparisons present in the data.

 **Hypothesis Testing**

 * T-test: A t-test is a statistical test used to determine whether there is a significant difference between the means of two groups.The t-test is typically used when the sample size is small, or when the population standard deviation is unknown. There are several types of t-tests, including the Student's t-test and the Welch's t-test, which are used for different types of data and research questions.
 *  chi squared test: The chi-squared test is a statistical test used to determine whether there is a significant difference between the expected frequencies and the observed frequencies in one or more categories. It is commonly used to determine if there is a significant association between two categorical variables.


 **Feature Engineering & Data Pre-processing**

*  Handling Missing Values
*  Handling Outliers
*  Categorical Encoding

**Feature Manipulation & Selection**

Initially, we explored all variables using data wrangling and data visualization to explore their relationships. Based on their importance and impact on producing CHD, we selected specific features. Subsequently, during the feature manipulation stage, we combined certain variables into a single feature and eliminated some unnecessary variables.

Finally, we identified 13 independent features that significantly influence the development of CHD. To validate their importance, we utilized the Embedded method with a random forest classifier feature importance. As depicted in the graph above, all 13 features displayed some level of importance, with none of them having a zero value

**Handling Imbalanced Dataset**

Due to the highly imbalanced nature of the given data, we employed a popular technique called SMOTE (Synthetic Minority Over-sampling Technique) in our machine learning approach. SMOTE is specifically designed to address the class imbalance problem by oversampling the minority class.

To achieve class balance, SMOTE generates synthetic samples for the minority class. This process involves interpolating between existing minority class samples. By selecting two or more nearest minority class samples and taking linear combinations of their feature values, new synthetic samples are created. These synthetic samples mimic the characteristics of the existing minority class instances, effectively increasing the overall number of minority class samples and restoring balance to the class distribution.

Through the application of SMOTE, we ensured that our machine learning models were better equipped to handle imbalanced data, leading to more accurate and reliable predictions for the minority class.

![image](https://github.com/irfan7210/Cardiovascular-Risk-Prediction/assets/113547056/1bf36b1f-1e15-4965-92c6-cff94bed932a)



 **ML Model Implementation**
 In our project, we explored a diverse set of models, namely Logistic Regression, Random Forest Classifier, SVM Classifier, Decision Tree Classifier, XGBoost, and KNN Classifier. After rigorous evaluation, it became evident that the Random Forest Classifier outperformed all the other models, exhibiting the highest level of predictive accuracy.

To further enhance the model's performance, we employed advanced techniques such as hyperparameter tuning and cross-validation. These practices allowed us to fine-tune the model's parameters and validate its effectiveness on unseen data, leading to improved accuracy in our predictions.

The selection of the Random Forest Classifier for predicting the TenYearCHD was based on its outstanding performance, closely followed by the XGBoost classifier.

Upon applying the Random Forest Classifier, we achieved remarkable accuracy scores of 100% for training data and 90% for test data. For both classes (NoCHD or 0, and CHD or 1), we obtained high precision, recall, and f1-scores, indicating excellent predictive capabilities. The roc_auc_score, which measures the area under the receiver operating characteristic curve, was also commendable, with a value of 89.3% for the test data.

It's worth noting that Random Forest is an ensemble learning technique based on bagging, which involves using multiple decision trees trained on subsets of data and aggregating their outputs through majority voting. Although it proved highly effective in our project, we acknowledge that its interpretability is limited, often referred to as a "black box" model due to its lack of transparency and explanation.


**model explainability**

The Random Forest algorithm uses the bagging technique to train multiple decision tree models on different subsets of the data chosen at random. The model then predicts whether an individual has CHD or NoCHD using the majority voting scheme.

For example, if the model trains 10 decision trees and seven of them predict a result of 1 (CHD) and three of them predict a result of 0 (NoCHD) for a particular observation, the model will give a final result of 1 (CHD).

We used lime (model explainability tool), to represent the feature importance for this model.

![image](https://github.com/irfan7210/Cardiovascular-Risk-Prediction/assets/113547056/a543f1cd-a315-4bef-b27d-b4203710b868)



**Conclusion**


In conclusion, all of the features offered in the dataset are crucial and affect the likelihood of developing CHDs. Although, we can draw some very significant features, such as:


*   The likelihood of being diagnosed with heart disease rises with advancing age.

*   Another important factor that contributes to CHDs is smoking.

*   Patients who struggle with diabetes and high cholesterol have a higher risk of CHDs.

*   The likelihood of receiving a CHD diagnosis is higher in patients with Hypertension.

*    Heart rate is another reason for devolopping CHD

*    Additionally, it has been observed that those without education are more susceptible to CHD

*   CHDs are more likely to occur in patients with high blood sugar levels.

*   The risk of CHD development is higher in patients who have had "strokes."

*   The likelihood of receiving a CHD diagnosis is higher in patients with high BMIs.

*    Finally, we can say that RandomForest Classifier outperformed all other models with an accuracy of 90% and an f1-score of 0.91. It is undoubtedly the highest score we have ever received. We can therefore safely say that RandomForest Classifier offers the best solution to our problem.


# online-retail-customer-segmentation

**Project Summary**
Customer Personality Analysis is a comprehensive examination of a company's ideal customers, enabling the business to gain a deeper understanding of its customer base and tailor its products to meet the unique needs, behaviors, and concerns of different customer segments.

By conducting a Customer Personality Analysis, businesses can better identify and target specific customer segments. Instead of spending resources on marketing a new product to every customer in their database, a company can analyze which customer segment is most likely to purchase the product and focus their marketing efforts on that particular segment. This targeted approach can result in more effective marketing campaigns, higher conversion rates, and increased revenue.

The aim of the project is to analyze the transaction data set of a UK-based non-store online retail business specializing in unique all-occasion gifts. Through this analysis, the project seeks to identify various customer segments based on their purchasing behavior, demographics, purchasing frequency, and average spend. The ultimate goal is to gain insights that can aid the business in optimizing its marketing and sales strategies to better serve its customers, leading to increased customer satisfaction and revenue.

**Problem Statement**

Understanding the value of each customer is crucial for any business, and the RFM analysis is a popular method used for this purpose. RFM stands for Recency, Frequency, and Monetary Value, and it is a technique that helps businesses analyze their customers' value based on these three parameters.

By using RFM analysis, businesses can segment their customers into groups based on their recency of purchase, frequency of purchase, and the monetary value of their purchases. The resulting segments can then be ordered from the most valuable customers (those with the highest recency, frequency, and value) to the least valuable customers (those with the lowest recency, frequency, and value).

Customer segmentation is the practice of dividing the customer base into groups of individuals based on common characteristics such as age, gender, interests, and spending habits. By performing customer segmentation using RFM analysis, businesses can gain a deeper understanding of their customers' value and tailor their marketing efforts and product offerings to each segment's unique needs and preferences.


**About Dataset**

This is a transnational data set which contains all the transactions that occurred between 01/12/2010 and 09/12/2011 for a UK-based and registered non-store online retail. The dataset consists of 541909 rows and 8 columns. However, it is important to note that two of these columns contain missing information. Specifically, the CustomerID column has only 406829 values, indicating that some information is missing. The same is true for the Description column. When looking at the summary statistics generated by the describe function, it is apparent that some negative values exist in the data.

This is a transnational data set which contains all the transactions occurring between 01/12/2010 and 09/12/2011 for a UK-based and registered non-store online retail.The company mainly sells unique all-occasion gifts. Many customers of the company are wholesalers

**InvoiceNo**: Invoice number. Nominal, a 6-digit integral number uniquely assigned to each transaction. If this code starts with letter 'c', it indicates a cancellation.

**StockCode**: Product (item) code. Nominal, a 5-digit integral number uniquely assigned to each distinct product.

**Description**: Product (item) name. Nominal.

**Quantity**: The quantities of each product (item) per transaction. Numeric.

**InvoiceDate**: Invoice Date and time. Numeric, the day and time when each transaction was generated.

**UnitPrice**: Unit price. Numeric, Product price per unit in sterling.

**CustomerID**: Customer number. Nominal, a 5-digit integral number uniquely assigned to each customer.

**Country**: Country name. Nominal, the name of the country where each customer resides.

 **Data Wrangling & Data Vizualization**

 Data wrangling, also known as data preprocessing or data cleaning, refers to the process of transforming and preparing raw data into a suitable format for analysis. It involves cleaning, organizing, and transforming data to ensure its quality, consistency, and compatibility with the analytical tasks at hand.

Data visualization is the process of representing data and information graphically to facilitate understanding, exploration, and communication of insights. It involves creating visual representations such as charts, graphs, maps, and infographics to effectively convey patterns, trends, relationships, and comparisons present in the data.

 **Hypothesis Testing**

 * t_test.:A t-test is a statistical test that is used to compare the means of two groups of data to determine whether there is a significant difference between them. It is a hypothesis test that helps in determining whether the difference between the means of the two groups is due to chance or is statistically significant


**Feature Engineering & Data Pre-processing**

*  Handling Missing Values
*  Handling Outliers
*  Feature Manipulation & Selection:

we have used RFM method in the form of features, RFM analysis, recency, frequency, and monetary (also known as "monetary value") are three key customer metrics that can be used to segment customers based on their purchasing behavior. Here's a brief explanation of each metric:

Recency: This metric measures how recently a customer made a purchase. Customers who have made a purchase more recently are considered to be more valuable than those who haven't made a purchase in a while. Recency is often measured in terms of the number of days since the customer's last purchase.

Frequency: This metric measures how frequently a customer makes purchases. Customers who make more frequent purchases are considered to be more valuable than those who make fewer purchases. Frequency is often measured in terms of the number of purchases made over a certain time period, such as a year or a quarter.

Monetary value: This metric measures the total amount of money that a customer has spent on purchases. Customers who have spent more money are considered to be more valuable than those who have spent less money.

by using above features we are going to implement diffrent diffrent machine model to get optimal solution

* Data Transformation
* Data Scaling
* Dimesionality Reduction


**ML Model Implementation**

In our implementation, we explored a diverse array of clustering models, such as K-Means Clustering, the Elbow Method for optimal K selection, DBSCAN, and Hierarchy Clustering. Among these techniques, DBSCAN stood out as a powerful clustering algorithm with the ability to handle various data types and shapes effectively.

![image](https://github.com/irfan7210/online-retail-customer-segmentation/assets/113547056/3d981472-d8e1-4c93-a8f3-3b39433f6814)


When DBSCAN yielded the optimal clustering result, it showcased its capability to discern the underlying structure in your data successfully. By identifying two optimal clusters based on Recency and Monetary attributes, it revealed the presence of two distinct groups within your data. This valuable insight suggests that these groups can be meaningfully separated and analyzed independently, offering a deeper understanding of the patterns and characteristics within your dataset.

![image](https://github.com/irfan7210/online-retail-customer-segmentation/assets/113547056/a7f39a51-a20f-40d7-86c9-869da2ba022b)

**Conclusion**


*   related to customer segmentation and targeting. By analyzing customer behavior in terms of their recency of purchase, frequency of purchase, and monetary value of purchase, businesses can identify and target high-value customers, personalize marketing campaigns, and improve customer retention.


*   different clustering algorithms were applied to the dataset, including clustering on Recency, Frequency, and Monetary (RFM) with 2 optimal clusters found. 

*  DBScan method performing well on our data set with  optimal number of culuster is two on the features using Recency Monetary
