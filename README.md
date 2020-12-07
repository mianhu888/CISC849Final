# CISC849Final
This repo is used for final project of CISC849, a graduate level seminar offered at the University of Delaware. Our group consists of Matt Leinhauser and Eric (Yifan) Zhang.

### Context of this project - TPOT
This course requires each group reading a paper (or article) related to data science and presenting it formally to the entire class. We chose an article about TPOT [1], which is a Python Automated Machine Learning tool that optimizes machine learning pipelines using genetic programming.

We uploaded the slides we used for the presentation, check `TPOT Paper presentation.pptx` in this repo. In the slides, we describe and visualize the main technique involved in TPOT.

### Project idea
We were also required to conduct a final project for this course. The final project needs to relate interesting tasks to data science. First, we had to present a proposal about the final project. In this proposal, we identified the task, dataset, tentative technique/ML algorithms, and timeline.

We checked Kaggle and found a dataset called `COVID-19 Airline Flight Delays and Cancellations - Which airlines have been the most affected by COVID-19?`[2]. It has the following description:

>The United States Department of Transportation's (DOT) Bureau of Transportation Statistics tracks the on-time performance of domestic flights operated by large air carriers. The data collected is from January - June 2020 and contains relevant flight information (on-time, delayed, canceled, diverted flights) from the Top 10 United States flight carriers for 11 million flights.

As COVID-19 is still impacting the entire world (at the time of writing, December 2020), this dataset provides some insights into how it affect the airline industry. We found this topic interesting and after discussion, we planned to use this dataset to try to predict the flight cancellations based on the information contained in this dataset.

We have presented TPOT as a useful automated tool to generate ML pipelines, one drawback of TPOT is that it requires relatively a lot of time for generating. So an idea came to mind: `What about hosting a competition between TPOT and a manually-selected model?` One of our group members (Eric) focused on learning how to generate optimal machine learning pipelines using TPOT and the other group member (Matt) focused on manually selecting the best models.

### Data description
There are 47 features for each data item, the dataset has a file called `ColumnDescription.txt` to describe each feature. We analyzed all features by two methods: 1) Manually categorize them. 2) Compute and plot features correlations. We provide feature names, descriptions, categories, whether or not kept, and reasons for either keeping or droping the feature(s) in belowing table.

Categories | Feature      | Description | Kept | Reasons
-------| ----------- | ----------- | ---| ---
| YEAR      | Year       |
| QUARTER   | 1: Jan-Mar, 2: Apr-Jun, 3: Jul-Sep, 4: Oct-Dec|
| MONTH      | Month of Year       |
| DAY_OF_MONTH      | Date of Month       |
| DAY_OF_WEEK      | Day of Week (1: Monday, 7: Sunday)       |
| FL_DATE      | Full flight date (M/DD/YYYY)       |
| MKT_UNIQUE_CARRIER      | Airline Carrier Code:<br> AA: American Airlines<br>AS: Alaska Airlines<br>	B6: JetBlue<br>	DL: Delta Air Lines<br>	F9: Frontier Airlines<br>	G4: Allegiant Air<br>	HA: Hawaiian Airlines<br>	NK: Spirit Airlines<br>	UA: United Airlines<br>	WN: Southwest Airlines |
| MKT_CARRIER_FL_NUM      | Flight Number       |
| TAIL_NUM      | Aircraft Tail Number (Usually starts with 'N')       |
| ORIGIN      | Flight Departure 3-Letter Airport Abbreviation       |
| ORIGIN_CITY_NAME      | Flight Departure City, State Names       |
| ORIGIN_STATE_ABR      | Flight Departure 2-Letter State Abbreviation       |
| ORIGIN_STATE_NM      | Flight Departure State Name       |
| DEST      | Flight Arrival 3-Letter Airport Abbreviation       |
| DEST_CITY_NAME      | Flight Arrival City, State Names       |
| DEST_STATE_ABR      | Flight Arrival 2-Letter State Abbreviation       |
| DEST_STATE_NM      | Flight Arrival State Name       |
| CRS_DEP_TIME      | Scheduled Departure Time (HHMM) (Single or 2-Digit Values Represent 00:MM, e.g. 3 represents 00:03 or 12:03 AM)       |
| DEP_TIME      | Actual Departure Time (HHMM)       |
| DEP_DELAY      | Departure Delay (Difference Between Actual Departure Time and Scheduled Departure Time in Minutes)       |
| DEP_DELAY_NEW      | Departure Delay Ignoring Early Departures (Listed as 0)       |
| DEP_DEL15      | Departure Delay Greater Than 15 Minutes (0: Not Greater Than 15, 1: Greater Than 15)       |
| DEP_DELAY_GROUP      | Departure Delay in Number of 15-minute increments Rounded Down (e.g. Early Departure (< 0) is a value of -1, 30 or 42 minutes is a value of 2)       |
| DEP_TIME_BLK      | Scheduled Departure Time in Hourly Block (HHMM)       |
| TAXI_OUT      | Time between Airplane Taxi from Gate and Takeoff (WHEELS_OFF) Time (in Minutes)       |
| WHEELS_OFF      | Time of Airplane Takeoff (HHMM)       |
| WHEELS_ON      | Time of Airplane Landing (HHMM)       |
| TAXI_IN      | Time between Airplane Taxi to Gate and Landing (WHEELS_ON) Time (in Minutes)       |
| CRS_ARR_TIME      | Scheduled Arrival Time (HHMM) (Single or 2-Digit Values Represent 00:MM, e.g. 3 represents 00:03 or 12:03 AM)       |
| ARR_TIME      | Actual Arrival Time (HHMM)       |
| ARR_DELAY      | Arrival Delay (Difference Between Actual Arrival Time and Scheduled Arrival Time in Minutes)       |
| ARR_DELAY_NEW      | Arrival Delay Ignoring Early Arrivals (Listed as 0)       |
| ARR_DEL15      | Arrival Delay Greater Than 15 Minutes (0: Not Greater Than 15, 1: Greater Than 15)       |
| ARR_DELAY_GROUP      | Arrival Delay in Number of 15-minute increments Rounded Down (e.g. Early Arrival (< 0) is a value of -1, 30 or 42 minutes is a value of 2)       |
| ARR_TIME_BLK      | Scheduled Arrival Time in Hourly Block (HHMM)       |
| CANCELLED      | 0: Flight Not Cancelled, 1: Flight Cancelled       |
| CANCELLATION_CODE      | Reason for Cancellation - if Cancelled, Letter Present (A: Carrier, B: Weather, C: National Aviation System, D: Security)       |
| CRS_ELAPSED_TIME      | Scheduled Total Flight Time (in Minutes)       |
| ACTUAL_ELAPSED_TIME      | Actual Total Elapsed Flight Time (in Minutes)       |
| AIR_TIME      | Actual Total Elapsed Time Airplane in the Air (in Minutes)       |
| DISTANCE      | Distance Between Departure and Arrival Airports (in Miles)       |
| DISTANCE_GROUP      | Distance Between Departure and Arrival Airports in Number of 250-Mile increments Rounded Down (e.g. 400 miles is a value of 1)       |
| CARRIER_DELAY      | Carrier Delay (in Minutes)       |
| WEATHER_DELAY      | Weather Delay (in Minutes)       |
| NAS_DELAY      | National Aviation System Delay (in Minutes)       |
| SECURITY_DELAY      | Security Delay (in Minutes)       |
| LATE_AIRCRAFT_DELAY      | Late Aircraft Delay (in Minutes)       |

### CPU-based TPOT
### GPU-based TPOT


>[1] Randal S. Olson, Nathan Bartley, Ryan J. Urbanowicz, and Jason H. Moore (2016). Evaluation of a Tree-based Pipeline Optimization Tool for Automating Data Science. *Proceedings of GECCO 2016*, pages 485-492.
>
>[2] https://www.kaggle.com/akulbahl/covid19-airline-flight-delays-and-cancellations
>[3] https://blog.quantinsti.com/gini-index/

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
