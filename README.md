# Predicting Flight Cancellations During the COVID-19 Pandemic

### Take away points
This repo is used for final project of CISC849, Fall 2020, a graduate level seminar offered at the University of Delaware. Our group consists of Matt Leinhauser and Eric (Yifan) Zhang.

We hosted a competition between automated machine learning pipeline generating tools and manually selected model. We have explored the usage of TPOT on both CPU and GPU, we also explicitly selected models from several basic machine learning algorithms. Besides the competition, we discussed data balancing technique to achieve better performance. This project could give an example of the comparison between auto ML tools and human selection on a same data science task.

### Context of this project - TPOT
This course requires each group reading a paper (or article) related to data science and presenting it formally to the entire class. We chose an article about TPOT [1], which is a Python Automated Machine Learning tool that optimizes machine learning pipelines using genetic programming.

We uploaded the slides we used for the presentation, check `TPOT Paper presentation.pptx` in this repo. In the slides, we describe and visualize the main technique involved in TPOT.

### Project idea
We were also required to conduct a final project for this course. The final project needs to relate interesting tasks to data science. First, we had to present a proposal about the final project. In this proposal, we identified the task, dataset, tentative technique/ML algorithms, and timeline.

We checked Kaggle and found a dataset called `COVID-19 Airline Flight Delays and Cancellations - Which airlines have been the most affected by COVID-19?`[2]. It has the following description:

>The United States Department of Transportation's (DOT) Bureau of Transportation Statistics tracks the on-time performance of domestic flights operated by large air carriers. The data collected is from January - June 2020 and contains relevant flight information (on-time, delayed, canceled, diverted flights) from the Top 10 United States flight carriers for 11 million flights.

As COVID-19 is still impacting the entire world (at the time of writing, December 2020), this dataset provides some insights into how it affect the airline industry. We found this topic interesting and after discussion, we planned to use this dataset to try to predict the flight cancellations based on the information contained in this dataset.

We have presented TPOT as a useful automated tool to generate ML pipelines, one drawback of TPOT is that it requires relatively a lot of time for generating. So an idea came to mind: `What about hosting a competition between TPOT and a manually-selected model?` One of our group members (Eric) focused on learning how to generate optimal machine learning pipelines using TPOT and the other group member (Matt) focused on manually selecting the best models.

### Goal
Our project had the following goals:
1.) Accurately predict airline flight cancellations during the COVID-19 pandemic from Jan. 2020 - Jun. 2020
2.) Become familiar with TPOT and see if TPOT can give a good machine learning pipeline that will help us achieve Goal 1.
3.) Have one group member not use TPOT and not look at the TPOT results and see if they can create a better machine learning pipeline than TPOT.
  a.) It is worth noting that this goal is measured in terms of accuracy on the data.

### Data description
There are 47 features for each data item, the dataset has a file called `ColumnDescription.txt` to describe each feature. We analyzed all features by two methods: 1) Manually categorize them. 2) Compute and plot features correlations. We provide categories, feature names, descriptions, and whether or not kept in belowing table, where `{1}` indicates CPU-TPOT, `{2}` indicates GPU-TPOT.

Categories | Feature      | Description | Kept
---| --- | --- | ---
Time | YEAR      | Year       |
Time| QUARTER   | 1: Jan-Mar, 2: Apr-Jun, 3: Jul-Sep, 4: Oct-Dec|{1}<br>{2}
Time| MONTH      | Month of Year       |{1}<br>{2}
Time| DAY_OF_MONTH      | Date of Month       |{1}<br>{2}
Time| DAY_OF_WEEK      | Day of Week (1: Monday, 7: Sunday)       |{1}<br>{2}
Time| FL_DATE      | Full flight date (M/DD/YYYY)       |
Identification| MKT_UNIQUE_CARRIER      | Airline Carrier Code:<br> AA: American Airlines<br>AS: Alaska Airlines<br>	B6: JetBlue<br>	DL: Delta Air Lines<br>	F9: Frontier Airlines<br>	G4: Allegiant Air<br>	HA: Hawaiian Airlines<br>	NK: Spirit Airlines<br>	UA: United Airlines<br>	WN: Southwest Airlines |{1}<br>{2}
Identification| MKT_CARRIER_FL_NUM      | Flight Number       |{1}<br>{2}
Identification| TAIL_NUM      | Aircraft Tail Number (Usually starts with 'N')       |{2}
Location| ORIGIN      | Flight Departure 3-Letter Airport Abbreviation       |{1}<br>{2}
Location| ORIGIN_CITY_NAME      | Flight Departure City, State Names       |
Location| ORIGIN_STATE_ABR      | Flight Departure 2-Letter State Abbreviation       |
Location| ORIGIN_STATE_NM      | Flight Departure State Name       |
Location| DEST      | Flight Arrival 3-Letter Airport Abbreviation       |{1}<br>{2}
Location| DEST_CITY_NAME      | Flight Arrival City, State Names       |
Location| DEST_STATE_ABR      | Flight Arrival 2-Letter State Abbreviation       |
Location| DEST_STATE_NM      | Flight Arrival State Name       |
Departure| CRS_DEP_TIME      | Scheduled Departure Time (HHMM) (Single or 2-Digit Values Represent 00:MM, e.g. 3 represents 00:03 or 12:03 AM)       |{1}<br>{2}
Departure| DEP_TIME      | Actual Departure Time (HHMM)       |{2}
Departure| DEP_DELAY      | Departure Delay (Difference Between Actual Departure Time and Scheduled Departure Time in Minutes)       |{2}
Departure| DEP_DELAY_NEW      | Departure Delay Ignoring Early Departures (Listed as 0)       |
Departure| DEP_DEL15      | Departure Delay Greater Than 15 Minutes (0: Not Greater Than 15, 1: Greater Than 15)       |{2}
Departure| DEP_DELAY_GROUP      | Departure Delay in Number of 15-minute increments Rounded Down (e.g. Early Departure (< 0) is a value of -1, 30 or 42 minutes is a value of 2)       |{2}
Departure| DEP_TIME_BLK      | Scheduled Departure Time in Hourly Block (HHMM)       |{1}
Departure| TAXI_OUT      | Time between Airplane Taxi from Gate and Takeoff (WHEELS_OFF) Time (in Minutes)       |
Departure| WHEELS_OFF      | Time of Airplane Takeoff (HHMM)       |
Arrival| WHEELS_ON      | Time of Airplane Landing (HHMM)       |
Arrival| TAXI_IN      | Time between Airplane Taxi to Gate and Landing (WHEELS_ON) Time (in Minutes)       |
Arrival| CRS_ARR_TIME      | Scheduled Arrival Time (HHMM) (Single or 2-Digit Values Represent 00:MM, e.g. 3 represents 00:03 or 12:03 AM)       |{1}<br>{2}
Arrival| ARR_TIME      | Actual Arrival Time (HHMM)       |{2}
Arrival| ARR_DELAY      | Arrival Delay (Difference Between Actual Arrival Time and Scheduled Arrival Time in Minutes)       |{2}
Arrival| ARR_DELAY_NEW      | Arrival Delay Ignoring Early Arrivals (Listed as 0)       |
Arrival| ARR_DEL15      | Arrival Delay Greater Than 15 Minutes (0: Not Greater Than 15, 1: Greater Than 15)       |{2}
Arrival| ARR_DELAY_GROUP      | Arrival Delay in Number of 15-minute increments Rounded Down (e.g. Early Arrival (< 0) is a value of -1, 30 or 42 minutes is a value of 2)       |{2}
Arrival| ARR_TIME_BLK      | Scheduled Arrival Time in Hourly Block (HHMM)       |{1}
Cancellation| CANCELLED      | 0: Flight Not Cancelled, 1: Flight Cancelled       |{1}<br>{2}
Cancellation| CANCELLATION_CODE      | Reason for Cancellation - if Cancelled, Letter Present (A: Carrier, B: Weather, C: National Aviation System, D: Security)       |
On flight| CRS_ELAPSED_TIME      | Scheduled Total Flight Time (in Minutes)       |{1}<br>{2}
On flight| ACTUAL_ELAPSED_TIME      | Actual Total Elapsed Flight Time (in Minutes)       |{2}
On flight| AIR_TIME      | Actual Total Elapsed Time Airplane in the Air (in Minutes)       |
On flight| DISTANCE      | Distance Between Departure and Arrival Airports (in Miles)       |{1}<br>{2}
On flight| DISTANCE_GROUP      | Distance Between Departure and Arrival Airports in Number of 250-Mile increments Rounded Down (e.g. 400 miles is a value of 1)       |{1}<br>{2}
Delay| CARRIER_DELAY      | Carrier Delay (in Minutes)       |
Delay| WEATHER_DELAY      | Weather Delay (in Minutes)       |
Delay| NAS_DELAY      | National Aviation System Delay (in Minutes)       |
Delay| SECURITY_DELAY      | Security Delay (in Minutes)       |
Delay| LATE_AIRCRAFT_DELAY      | Late Aircraft Delay (in Minutes)       |

### TPOT Implementation

#### CPU-based
We firstly implemented what we have read in the TPOT paper - the CPU-based TPOT [1], the notebook could be found in this repo as `CISC849_TPOT_CPU.ipynb`. Our machine is `Intel(R) Core(TM) i5-9600k CPU @ 3.70GHz`

For feature selection, we kept several features listed in the table above. We encoded all categorical features into numerical values using label encoding. Training-testing data was splited by 75%/25%. We implemented TPOT with the following parameter `tpot = TPOTClassifier(generations=5, population_size=40, verbosity=1, random_state=42, n_jobs = -1, warm_start = True, max_time_mins = 60)`.

After 6 hours running, the TPOT has selected a pipeline of `KNeighborsClassifier(n_neighbors=12, p=1, weights='distance'))` and achieved 91.89% accuracy. The exported python script can be viewed in `tpot_airflight_pipeline_CPU.py`.

#### GPU-based TPOT
We also found a article called `Faster AutoML with TPOT and RAPIDS`[4]. In this article, the author described that TPOT could be acceralted by GPU and achieve better performance with less time. We then have implemented GPU-acceralted TOPT on Google Colab. The GPU in Colab was `Tesla T4`.

We have also kept different features with previous CPU-TPOT, check above table for details. After one hour running, we have achieved 92.79% accuracy.

#### One hot encoding and data balancing
According to Buda (2018) comprehensive review on data imbalancing solutions [5], daba imbalancing could cause prediction inaccurate. In this data set, the cancelled flights only occupy ~10%, causing data imbalancing. Thus, we considered undersampling for data balancing. Additionally, we have tried one hot encoding for feature engineering. The code could be viewed in `CISC849_TPOT_GPU.ipynb`.


### Manually Selecting a Machine Learning Pipeline
Within the dataset, the CANCELLED feature is binary (0 represents a flight that was not cancelled and 1 represents a flight that was cancelled). Due to this fact, I (Matt) decided to use machine learning algorithms that are good for supervised learning binary classification problems.

#### Packages Used:
For the core ML algorithms, we used `scikit-learn`.
For plotting figures, we used `seaborn` and `matplotlib`.
For dataset manipulations, we used `pandas` to transform the dataset into a dataframe.

#### Algorithms Selected:
- Naive Bayes Classifier
-- Within Naive Bayes, there is a threshold (50%) on how likely an instance is to be classified as a value (0 or 1) in a binary classification problem. We chose to use Naive Bayes because we believe it might mirror how flights are cancelled in reality. Looking at all factors (weather, arrival times, departure times, etc.), if an airline team determines there is a greater than 50% chance of cancelling the flight, it probably will get cancelled. We hoped Naive Bayes could emulate this decision making process.

- Decision Tree Classifier
-- For the Decision Tree Classifier we used GINI Impurity. GINI impurity measures the degreee of probability of a particular variable being wrongly classified when it is randomly chosen [3]. We chose to use a decision tree because it seems like a very logical way on how to cancel a flight. For example, if it is snowing out, the tree would follow a certain path that could not be followed if it was not snowing. Every path will lead to a decision, whether to cancel the flight or not.

- K Nearest Neighbors Classifier
-- We decided to use the KNN classifier because it also seems intuitive. If three flights, with very similar attributes, are all classified as cancelled and a fourth flight comes along with similar attributes to those three flights, there is a high probability that flight will also be classified as cancelled. We also think this might mirror how flight cancellations are made in real life. For example, if Southwest Airlines, Hawaiian Airlines, and JetBlue cancel their flights from Denver, CO to Honolulu, HI because of snow, American Airlines will most likely also do the same.

- Random Forest Classifier
-- Similar to the Decision Tree Classifier, we used the Random Forest Classifier as a "more powerful" decision tree. A Random Forest Classifier is basically a tree of decision trees. The tree that has the highest accuracy is then chosen as the tree the Random Forest Classifier uses.

#### Methodology
To start, we wanted to use the default classifiers scikit-learn offered. We figured based on the results from the default configurations, we could then tune the hyperparameters to increase our accuracy as needed. To divide up the dataset, we did so two ways. First, we did a regular train-test split, thus dividing our data into a training set and a testing set. We did this using scikit-learn's `train_test_split` function. Second, we used k-fold cross validation with k=5. We decided to test both ways because the dataset is extremely unbalanced. As a whole, the dataset only contains 282,926 cancelled flights, or just over 10% of the data given to us! By using k-fold cross validation, we can verify if the accuracy we achieve from the train-test split is, well, accurate.

#### Results using training set and testing set
Using the train-test split, we achieved the following results with only the default classifiers:
ML Method | Accuracy
---| ---
Naive Bayes Classifier| 89.35%
Decision Tree Classifier| 91.61%
K-Nearest Neighbor Classifier| 90.94%
Random Forest Classifier| 93.30%

#### Results using k-fold Cross Validation
Using 5-fold cross validation, we achieved the following results with only the default classifiers:
ML Method | k=5 Accuracy|  k=10 Accuracy
---| --- | ---
Naive Bayes Classifier| 73.63% | 86.69%
Decision Tree Classifier| 68.63% | 67.44%
K-Nearest Neighbor Classifier| 73.63% | 86.69%
Random Forest Classifier| 76.25% | 75.40%

### Conclusions
Our project had three goals: 
1.) Accurately predict airline flight cancellations during the COVID-19 pandemic from Jan. 2020 - Jun. 2020
2.) Become familiar with TPOT and see if TPOT can give a good machine learning pipeline that will help us achieve Goal 1.
3.) Have one group member not use TPOT and not look at the TPOT results and see if they can create a better machine learning pipeline than TPOT.
  a.) It is worth noting that this goal is measured in terms of accuracy on the data.

We were able to achieve all three of these goals in the following ways. For Goal #1, we demonstrated that using TPOT and manually creating a machine learning pipeline, we were able to create a model that can predict airline flight cancellations during the COVID-19 pandemic. While the accuracy of our results varies, we have demonstrated that our models perform much better than a random guess. Second, we achieved goal number two by really exploring TPOT. Eric was able to get TPOT running on the CPU within Google Colab and his local machine. In addition to just using TPOT on the CPU, he also figured out how to run TPOT on a GPU using the cuML library from RAPIDS [6]. Using TPOT on the GPU sped up the time taken to create an accurate predictive model and it also predicted a different pipeline to use than the CPU run of TPOT. Finally, we were able to demonstrate that Matt created a machine learning pipeline that scored higher accuracy than the pipeline TPOT generated (which was goal #3). Coming together at the after each of us completed our parts offered us great insights into how to solve the problem at hand in a different way.

From this project, we learned how to use TPOT on both the CPU and GPU and learned how it creates effective machine learning pipelines. We also learned how to perform effective feature engineering on a dataset. If we had more time with this project, we would have liked to continue to find the most optimal features. Similarly, we learned to filter out which algorithms would be useful for this problem, and which would not be useful, by truly understanding the data in the dataset. For the future, we are interested in exploring if our model can generalize to future pandemics (and not just the COVID-19 pandemic) and seeing if we can generate more accurate results by further exploring the use of a one-hot-encoder on the dataset, performing additional feature engineering and pre-processing, and creating a method to make the dataset more balanced in terms of cancelled flights and flights that were not cancelled.

>[1] Randal S. Olson, Nathan Bartley, Ryan J. Urbanowicz, and Jason H. Moore (2016). Evaluation of a Tree-based Pipeline Optimization Tool for Automating Data Science. *Proceedings of GECCO 2016*, pages 485-492.
>
>[2] https://www.kaggle.com/akulbahl/covid19-airline-flight-delays-and-cancellations
>
>[3] https://blog.quantinsti.com/gini-index/
>
>[4] https://medium.com/rapids-ai/faster-automl-with-tpot-and-rapids-758455cd89e5
>
>[5] Buda, M., Maki, A., & Mazurowski, M. A. (2018). A systematic study of the class imbalance problem in convolutional neural networks. *Neural Networks*, 106, 249-259.
>
>[6] https://github.com/rapidsai/cuml
