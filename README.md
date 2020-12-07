# CISC849Final
This repo is used for final project of CISC849, a grad-level course offered in University of Delaware. The group member is Matt Leinhauser and Eric (Yifan) Zhang.

### Context of this project - TPOT
This course requires each group reading a paper (or article) related to data science and presenting it formally to the entire class. We have chosen the article about TPOT [1], which is a Python Automated Machine Learning tool that optimizes machine learning pipelines using genetic programming.

We uploaded the slides we used for the presentation, check `TPOT Paper presentation.pptx` in this repo. In the slides, we described and visualized the technique involved in TPOT.

### Project idea
We were also required to conduct a final project for this course. The final project should be some interesting tasks about data science. We need to firstly present a proposal about the final project, identifying the task, dataset, tentative technique, and timeline.

We checked Kaggle and found a dataset called `COVID-19 Airline Flight Delays and Cancellations - Which airlines have been the most affected by COVID-19?`[2]. It has the following description:

>The United States Department of Transportation's (DOT) Bureau of Transportation Statistics tracks the on-time performance of domestic flights operated by large air carriers. The data collected is from January - June 2020 and contains relevant flight information (on-time, delayed, canceled, diverted flights) from the Top 10 United States flight carriers for 11 million flights.

As COVID19 is still impacting the entire world (till this Readme was written), this dataset provides some insights into how it affect the airline industry. We found this topic is interesting and after discussion, we planned to use this dataset and try to predict the flight cancellations based on the information contained in this dataset.

We have presented TPOT as a useful automated tool to generate ML pipelines, one drawback of TPOT is that it requires relatively a lot of time for generating. So an idea came out of our mind: `What about hosting a competition between TPOT and manual-selected model?` One of our member (Eric) focused on TPOT and the other member (Mat) focused on manul-selected model.

### Data description
There are 47 features for each data item, the dataset has a file called `ColumnDescription.txt` to describe each feature. We analyzed all features by two methods: 1) Manually categorize them. 2) Compute and plot features correlations. We provide feature names, descriptions, categories, keep or not, and reasons in belowing table.

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
