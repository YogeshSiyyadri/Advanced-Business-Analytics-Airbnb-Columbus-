/* Advanced Business Analytics Modelling */
/* AirBnB Data Analysis*/
/*Yogesh Siyyadri 101164082*/

/* STEP 1: Importing the Data into SAS */
%web_drop_table(WORK.Import);

FILENAME REFFILE '/home/u63735976/AirBNB/ohio.xlsx';

PROC IMPORT DATAFILE=REFFILE 
	DBMS=XLSX 
	OUT=WORK.Import;
	GETNAMES=YES;
RUN;

PROC CONTENTS DATA=WORK.Import;
RUN;


%web_open_table(WORK.Import);


/* Neighborhood  Map View */
ODS GRAPHICS / RESET WIDTH=19.2in HEIGHT=10.8in;
PROC SGMAP plotdata=WORK.IMPORT; 
	openstreetmap; 
	scatter x=longitude y=latitude / group=neighbourhood_cleansed 
	name="scatterPlot" markerattrs=(size=3 symbol=circle); 
	keylegend "scatterPlot" / title='neighbourhood_cleansed';
RUN;
ODS GRAPHICS / RESET;


/*Step for Conversion of char to numeric */
DATA WORK.Import;
    SET WORK.Import;
    /* Attempt to convert and handle possible errors */
    host_response_rate_num = input(host_response_rate, ??best32.);
    host_acceptance_rate_num = input(host_acceptance_rate, ??best32.);
    drop host_response_rate host_acceptance_rate;
    rename host_response_rate_num = host_response_rate;
    rename host_acceptance_rate_num = host_acceptance_rate;



/* STEP 2: Investigating the dependent variable */

*Checking for missing or nonsensical values;
PROC UNIVARIATE DATA=WORK.Import;
	VAR Price;
	HISTOGRAM / Normal;
RUN;

*No missing values but outliers are observed.
List of Outliers;
PROC SQL;
	SELECT id, neighbourhood_cleansed 'Area', room_type 'Description', accommodates 'Capacity', 
	bathrooms, bedrooms, beds, price
	FROM WORK.Import
    WHERE price < 28 OR price > 989
    ORDER BY price;
RUN;


*Target variable Data Transformation & Outliers Elimination; 
DATA WORK.Airbnb;
    SET WORK.Import;
    WHERE 28 < price < 989;
    Price_Log = LOG(Price);
RUN;

*New dataset Airbnb contains new rows;

*Checking the Transformed Target Variable; 
PROC UNIVARIATE DATA=WORK.Airbnb;
	VAR Price_Log;
	HISTOGRAM / NORMAL;
RUN;


/* STEP 3: Investigating the numeric predictors */

*Checking for Missing values, other summary stats;
PROC MEANS DATA=Work.Airbnb 
	(KEEP=host_response_rate host_acceptance_rate accommodates 
	bathrooms bedrooms beds availability_30 number_of_reviews review_scores_rating 
	review_scores_accuracy review_scores_cleanliness review_scores_checkin 
	review_scores_communication review_scores_location review_scores_value reviews_per_month) 
	N NMISS MIN MAX MEAN MEDIAN STD;
RUN;




    DATA Work.Airbnb;
  SET Work.Airbnb;

  /* Replace missing values in host_response_rate with the mean */
  host_response_rate = COALESCE(host_response_rate, 0.97); /* Actual mean */

  /* Replace missing values in host_acceptance_rate with the mean */
  host_acceptance_rate = COALESCE(host_acceptance_rate, 0.93); /* Actual mean */

  /* Replace missing values in bathrooms with "Zero" */
  bathrooms = COALESCE(bathrooms, 'Zero');

  /* Replace missing values in bedrooms with the mean */
  bedrooms = COALESCE(bedrooms, 2.08); /* Actual Mean */

  /* Replace missing values in beds with the mean */
  beds = COALESCE(beds, 2.60); /* Actual Mean */

  /* Replace missing values in review_scores_rating with the mean */
  review_scores_rating = COALESCE(review_scores_rating, 4.79); /* Actual mean */

  /* Replace missing values in review_scores_accuracy with the mean */
  review_scores_accuracy = COALESCE(review_scores_accuracy, 4.83); /* Actual mean */

  /* Replace missing values in review_scores_cleanliness with the mean */
  review_scores_cleanliness = COALESCE(review_scores_cleanliness, 4.77); /* Actual mean */

  /* Replace missing values in review_scores_checkin with the mean */
  review_scores_checkin = COALESCE(review_scores_checkin, 4.88); /* Actual mean */

  /* Replace missing values in review_scores_communication with the mean */
  review_scores_communication = COALESCE(review_scores_communication, 4.89); /* Actual mean */

  /* Replace missing values in review_scores_location with the mean */
  review_scores_location = COALESCE(review_scores_location, 4.71); /* Actual mean */

  /* Replace missing values in review_scores_value with the mean */
  review_scores_value = COALESCE(review_scores_value, 4.73); /* Actual mean */

  /* Replace missing values in reviews_per_month with the mean */
  reviews_per_month = COALESCE(reviews_per_month, 2.35); /* Actual mean */

RUN;



*Extracting Percentiles from predictors for setting thresholds;
PROC MEANS DATA= Work.Airbnb (KEEP = accommodates bathrooms bedrooms beds ) 
	p1 p10 p25 p50 p75 p90 p95 p99; 
RUN;


* Setting thresholds for specified variables;
DATA WORK.Airbnb;
    SET WORK.Airbnb;
    if accommodates > 16 then accommodates = 16;
    if bathrooms > 4 then bathrooms = 4;
    if bedrooms > 5 then bedrooms = 5;
    if beds > 9 then beds = 9;
RUN;




*Checking levels in numeric variables;
PROC FREQ DATA=WORK.Airbnb nlevels;
  TABLES accommodates availability_30 bathrooms bedrooms beds;
RUN;

*Collapsing numeric variables into Categories;
DATA WORK.Airbnb;
    SET WORK.Airbnb;

	/* Categorize accommodates */
    IF accommodates <= 2 THEN
        accom_cat = 'Very_Small';
    ELSE IF accommodates <= 4 THEN
        accom_cat = 'Small';
    ELSE IF accommodates <= 6 THEN
        accom_cat = 'Medium';
    ELSE
        accom_cat = 'Large';

    /* Categorize availability_30 */
    IF availability_30 = 0 THEN avail_cat = "Booked";
  	ELSE IF availability_30 <= 10 THEN avail_cat = "Low";
  	ELSE IF availability_30 <= 20 THEN avail_cat = "Medium";
  	ELSE avail_cat = "High";

    
	/* Categorize bathrooms */
	IF bathrooms = 0 THEN bath_cat = "Nil";
  	ELSE IF bathrooms IN (0.5, 1, 1.5) THEN bath_cat = "Normal";
  	ELSE IF bathrooms IN (2, 2.5) THEN bath_cat = "Extra";
  	ELSE IF bathrooms IN (3, 3.5, 4) THEN bath_cat = "Luxury";

	/* Categorize bedrooms */
	IF bedrooms = 1 THEN bedrooms_cat = "Single";
  	ELSE IF bedrooms = 2 THEN bedrooms_cat = "Double";
  	ELSE IF bedrooms > 2 THEN bedrooms_cat = "Extra";

	/* Categorize beds*/
    IF beds = 1 THEN beds_Cat = "Single";
  	ELSE IF beds = 2 THEN beds_Cat = "Double";
  	ELSE IF beds > 2 THEN beds_Cat = "Extra";

RUN;

*Feature Engineering;
DATA WORK.Airbnb;
    SET WORK.Airbnb;

    /* Calculate beds_per_person */
    IF beds NOT IN (., 0) THEN
        beds_per_person = beds / accommodates;
    ELSE
        beds_per_person = 0;

    /* Calculate bath_per_person */
    IF bathrooms NOT IN (., 0) THEN
        bath_per_person = bathrooms / accommodates;
    ELSE
        bath_per_person = 0;
RUN;


*Checking VIFs by building a stepwise model;
PROC REG DATA=WORK.Airbnb PLOTS=ALL; 
	model Price_Log= host_response_rate host_acceptance_rate minimum_nights maximum_nights 
	beds_per_person bath_per_person number_of_reviews review_scores_rating review_scores_accuracy 
	review_scores_cleanliness review_scores_checkin review_scores_communication 
	review_scores_location review_scores_value reviews_per_month/ selection=forward 
	VIF COLLIN;
RUN;

*Due to a relatively higher p-value "beds_per_person", "review_scores_communication", 
"review_scores_accuracy", and "reviews_per_month" have been excluded from the model;  

*Creating Polynomials: No higher degree polynomials are being included in the model as they 
have resulted in high VIFs;


*Checking if Predictors need any transformations;
PROC UNIVARIATE DATA=Work.airbnb (KEEP = host_response_rate host_acceptance_rate minimum_nights maximum_nights 
	bath_per_person number_of_reviews review_scores_rating);
	Histogram / Normal;
RUN;

*Creating Transformations of Select Predictors;
DATA WORK.Airbnb;
    SET WORK.Airbnb;
	min_nights_tf = log(minimum_nights + 1);
	max_nights_tf  = log(maximum_nights + 1);
	bpp_tf = sqrt(bath_per_person);
	num_reviews_tf = log(number_of_reviews + 1);
RUN;


*Despite trying log / sqrt transformations, variables like host_response_rate, host_acceptance_rate, 
review_scores_rating did not  exhibit a normal distribution; 

*Checking Scatterplots of Predictors that could not be transformed;
PROC SGSCATTER DATA=Airbnb; 
	TITLE 'Scatter Plot Matrix of Select Predictors'; 
	MATRIX Price_Log host_response_rate host_acceptance_rate review_scores_rating/ 
	START=TOPLEFT ELLIPSE = (ALPHA=0.05 TYPE=PREDICTED) NOLEGEND;
RUN;

*The scatter plots of these variables also do not represent a dicipherable relationship with price. 
So, we decided to categorize and use only these three variables;

*Categorizing Numeric Predictors that could not be transformed;

PROC FREQ DATA=WORK.Airbnb nlevels;
  TABLES host_response_rate host_acceptance_rate review_scores_rating;
RUN;

DATA WORK.Airbnb;
    SET WORK.Airbnb; 

    /* Categorizing 'host_response_rate'*/
    IF host_response_rate = 1 THEN
        resp_cat = 'Perfect';
    ELSE IF host_response_rate >= 0.97 THEN
        resp_cat = 'Great';
    ELSE IF host_response_rate >= 0.8 AND host_response_rate < 0.97 THEN
        resp_cat = 'Good';
    ELSE IF host_response_rate < 0.8 THEN
        resp_cat = 'Bad';
    ELSE
        resp_cat = 'Worst';
    
    /* Categorizing 'host_acceptance_rate'*/
    IF host_acceptance_rate = 1 THEN acc_cat = "Absolute";
    ELSE IF 0.9 <= host_acceptance_rate < 1 THEN acc_cat = "High";
    ELSE IF 0.7 <= host_acceptance_rate < 0.9 THEN acc_cat = "Moderate";
    ELSE IF 0 < host_acceptance_rate < 0.7 THEN acc_cat = "Low";
    ELSE IF host_acceptance_rate = 0 THEN acc_cat = "Zero";
    
    /* Categorizing 'review_scores_rating'*/
    if review_scores_rating = 5 then rating_cat = 'Perfect';
    else if review_scores_rating < 5 and review_scores_rating >= 4 then rating_cat = 'Great';
    else if review_scores_rating < 4 and review_scores_rating >= 3 then rating_cat = 'Good';
    else if review_scores_rating < 3 and review_scores_rating >= 2 then rating_cat = 'Okay';
    else if review_scores_rating < 2 and review_scores_rating >= 1 then rating_cat = 'Bad';
    else if review_scores_rating < 1 and review_scores_rating > 0 then rating_cat = 'Terrible';
    else if review_scores_rating = 0 then rating_cat = 'Disgusting';
RUN;   

/* Create correlation analysis */ 
PROC CORR DATA=Work.airbnb; 
VAR min_nights_tf max_nights_tf bpp_tf num_reviews_tf;
RUN;


/* Checking normality assumptions of transformed numeric variables */
PROC UNIVARIATE DATA=WORK.Airbnb;
	VAR min_nights_tf max_nights_tf bpp_tf num_reviews_tf;
	HISTOGRAM / NORMAL;
RUN;

*ScatterPlots Matrix;
PROC SGSCATTER DATA=Airbnb; 
	TITLE 'Scatter Plot Matrix'; 
	MATRIX Price_Log min_nights_tf max_nights_tf bpp_tf num_reviews_tf/ 
	START=TOPLEFT ELLIPSE = (ALPHA=0.05 TYPE=PREDICTED) NOLEGEND;
RUN;


*Investigating Character Variables;
PROC FREQ DATA=WORK.Airbnb nlevels order=freq;
  TABLES neighbourhood_cleansed property_type room_type;
RUN;


*Categorizing Character Variables;
data Work.Airbnb;
    set Work.Airbnb;

    /* Categorizing "neighbourhood_cleansed" */
  	if neighbourhood_cleansed in ('Near North Side', 'West Town', 'Lake View', 'Near West Side', 
  	'Logan Square', 'Loop', 'Lincoln Park') then hood_cat = 'High-Demand';
    else if neighbourhood_cleansed in ('Near South Side', 'Lower West Side', 'Uptown', 
    'Edgewater', 'Avondale', 'Bridgeport', 'Irving Park', 'Woodlawn', 'Rogers Park', 
    'East Garfield Park', 'Grand Boulevard', 'North Center', 'South Shore', 'Lincoln Square', 
    'Humboldt Park', 'Hyde Park', 'Douglas', 'West Ridge', 'Portage Park') 
    then hood_cat = 'Moderate-Demand';
    else hood_cat = 'Low-Demand';
    
    /* Categorizing "property_type" */
    IF property_type IN ('Entire rental unit') THEN property_cat = 'Primary';
    ELSE IF property_type IN ('Entire condo', 'Private room in rental unit', 'Entire home',
	'Private room in home', 'Entire serviced apartment', 'Private room in condo',
    'Entire guest suite', 'Room in hotel') THEN property_cat = 'Secondary';
    ELSE property_cat = 'Others';

    /* Categorizing "room_type" */
    IF room_type IN ('Entire home/apt') THEN room_cat = 'Home';
    ELSE room_cat = 'Room';
run;

/*STEP 4: Splitting the data into TRAIN and TEST datasets*/ 
PROC SURVEYSELECT DATA=Work.Airbnb 
	SAMPRATE=0.20 SEED=42 
	OUT=Split OUTALL METHOD=SRS;
RUN;

*Creating the TRAIN and TEST Data Sets;
DATA TRAIN TEST;
	SET Split; 
	IF Selected=0 THEN OUTPUT TRAIN; ELSE OUTPUT TEST; DROP Selected; 
RUN;	

/* Creating macro variables for selected numeric and categorical predictor types */

%let num_vars = min_nights_tf max_nights_tf bpp_tf num_reviews_tf;

%let cat_vars = instant_bookable accom_cat avail_cat bath_cat bedrooms_cat beds_cat resp_cat 
acc_cat rating_cat hood_cat property_cat room_cat;

%let nom_cats = instant_bookable hood_cat property_cat room_cat;

%let ord_cats = accom_cat avail_cat bath_cat bedrooms_cat beds_cat resp_cat 
acc_cat rating_cat;


%let path = /home/u63735976/AirBNB;



** PREDICTIVE MODELS **;

** Linear Regression **;


ods noproctitle;
ods graphics / imagemap=on;

proc glmselect data=WORK.TRAIN plots=all;
    class &cat_vars / param=glm;
    model Price_Log= &num_vars &cat_vars / showpvalues 
        selection= lasso (stop=cv choose = cv );
        OUTPUT PREDICTED RESIDUAL OUT = train_lasso; 
        SCORE DATA=TEST PREDICTED RESIDUAL OUT=test_lasso;
run;


/* Verifying Linear Regression Assumptions*/

/* 1. Linearity: Scatter plot of observed vs. predicted values */
proc sgplot data=train_lasso;
   scatter x=p_Price_Log y=Price_Log;
   reg x=p_Price_Log y=Price_Log / lineattrs=(color=red);
   xaxis label='Predicted Values';
   yaxis label='Observed Values';
   title 'Checking Linearity: Observed by Predicted';
run;

/* 2. Normality of Residuals: Histogram*/
proc sgplot data=train_lasso;
   histogram r_Price_Log / binwidth=0.5;
   density r_Price_Log / type=kernel;
   xaxis label='Residuals';
   yaxis label='Density';
   title 'Checking Normality: Histogram of Residuals';
run;


/* 3. Homoscedasticity: Scatter plot of residuals vs. predicted values */
proc sgplot data=train_lasso;
   scatter x=p_Price_Log y=r_Price_Log;
   reg x=p_Price_Log y=r_Price_Log / lineattrs=(color=red);
   xaxis label='Predicted Values';
   yaxis label='Residuals';
   title 'Checking Homoscedasticity : Residuals by Predicted Plot';
run;


/* 4. Independence of Residuals */
proc sgplot data=train_lasso;
   scatter x=p_Price_Log y=r_Price_Log;
   refline -1 / axis=y lineattrs=(color=black);
   refline 1/ axis=y lineattrs=(color=black);
   xaxis label='Predicted Values';
   yaxis label='R-Student (Residuals Standardized)';
   title 'Checking Independence: RStudent by Predicted Plot';
run;


/* Application of model on the Test Data */
proc reg data=test_lasso plots = all;
   model Price_Log = p_Price_Log;
run;

**Decision Tree **;


/* Task 1: Fit a decision tree on the training data */
proc hpsplit data=WORK.TRAIN seed=786;
    class &cat_vars.;
    model Price_log = &num_vars. &cat_vars.;
    partition fraction(validate=0.3 seed=420);
    output out=hpsplout;
    code file="&path./hpsplexc.sas";
run;

/* Task 2: Score the hold-out test sample using the generated scoring code */
data WORK.TEST_TREE;
    set WORK.TEST;
    %include "&path./hpsplexc.sas";
run;

/* Task 3: Evaluate the performance of the model on the training data */
data WORK.TRAIN_TREE_EVAL;
    set WORK.TRAIN;
    %include "&path./hpsplexc.sas";
    residual_train = (p_Price_log - Price_log)**2;
    sqrt_residual_train = sqrt(residual_train);
    keep Price_log residual_train sqrt_residual_train;
run;

/* Calculate RMSE for Training Data */
proc sql noprint;
    select sqrt(mean(residual_train)) as RMSE_train
    into :RMSE_train_tree
    from WORK.TRAIN_TREE_EVAL;
quit;


/* Task 4: Evaluate the performance of the model on the test data */
data WORK.TEST_TREE_EVAL;
    set WORK.TEST_TREE;
    residual_test = (p_Price_log - Price_log)**2;
    sqrt_residual_test = sqrt(residual_test);
    keep Price_log residual_test sqrt_residual_test;
run;

/* Calculate RMSE for Test Data */
proc sql noprint;
    select sqrt(mean(residual_test)) as RMSE_test
    into :RMSE_test_tree
    from WORK.TEST_TREE_EVAL;
quit;

%put RMSE for Training Data: &RMSE_train_tree;
%put RMSE for Test Data: &RMSE_test_tree;

**Random Forrests** ;

/* Task 1: Fit a Random Forest on the training data */
ods graphics on;
proc hpforest data=train
    maxtrees=500 vars_to_try=16
    seed=66 
    maxdepth=20 leafsize=6
    alpha=0.1;
    target Price_log / level=interval;
    input &num_vars. / level=interval;
    input &nom_cats. / level=nominal;
    input &ord_cats. / level=ordinal;
    ods output fitstatistics=rf_train;
    save file="&path./rfmodel_fit.bin";
run;

/* Task 2: Score the hold-out test sample */
proc hp4score data=test;
    id Price_log;
    score file="&path./rfmodel_fit.bin"
    out=rf_test;
run;

/* Task 3: Calculate and print RMSE for both training & test data */
proc sql noprint;
   /* RMSE for Training Data */
   select sqrt(mean(PredAll)) as RMSE_train
   into :RMSE_train_rf
   from rf_train;

   /* RMSE for Test Data */
   select sqrt(mean((Price_log - P_Price_Log)**2)) as RMSE_test
   into :RMSE_test_rf
   from rf_test;
quit;

%put RMSE for Training Data: &RMSE_train_rf;
%put RMSE for Test Data: &RMSE_test_rf;
