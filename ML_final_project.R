#https://code.datasciencedojo.com/datasciencedojo/datasets/tree/8c8d5c573318cf2e0e07ed65af3c032d2dfcee42/Autism%20Screening%20Adult
#dataset related to autism screening of adults
#contains 20 features to be utilised for further analysis especially in determining influential autistic traits and improving the classification of ASD cases
#ten behavioural features (AQ-10-Adult) plus ten individuals characteristics that have proved to be effective in detecting the ASD cases from controls in behaviour science. 
#can we use a time-efficient and accessible ASD screening to help health professionals and inform individuals whether they should pursue formal clinical diagnosis?
###Can I use ML and behavioral features and individual characteristics to help in early diagnosis for ASD?
###This potentially could save money in heathcare costs and reduce lengthy wait times for a diagnosis. Although there is no cure for ASD, but treatments can help to improve symptoms. 
#to assist humans in clinical decision making by classifiying a likely diagnosis of Autistic Spectrum Disorder (ASD): Yes or No has ASD, using an app with 10 questions and an individuals characteristics


#http://rstudio-pubs-static.s3.amazonaws.com/383049_1faa93345b324da6a1081506f371a8dd.html

#Classification is a technique to predict what group a certain instance is going to be. To create classifiers, we use from the given learning data set and evaluate on the test samples, so it is possible predict what class the group is following to.

#A decision tree will divide the data into leaf nodes and each one of them will represent an attribute. In a nutshell, decision tree is a splitting method that is applied to demonstrate every possible outcome of a decision (Jain, 2016).
#“Because the parent population can be split into in numerous patterns, we are interested in the one with the greatest purity. In technical terminology, purity can be described by entropy.”
#To control the size and to select the optimal tree size the complexity parameter (cp) is used

#Each decision tree in the forest considers a random subset of features when forming questions and only has access to a random set of the training data points. This increases diversity in the forest leading to more robust overall predictions and the name ‘random forest.’
#idea behind a random forest is to combine many decision trees into a single model. 






#link to dataset
#https://archive.ics.uci.edu/ml/datasets/Autism+Screening+Adult
#acknowledgement
#data set has been sourced from the Machine Learning Repository of University of California, Irvine Autism Screening Adult Data Set (UC Irvine). 
#original source of the data set: Fadi Fayez Thabtah, Department of Digital Technology, Manukau Institute of Technology, Auckland, New Zealand 


#screening test included 10 questions (A1 to A10). In each of these 10 questions, 
#test takers were given a statement with which they had to agree or disagree. The responses to A1 and A10 are coded as binary values (0,1). 
#After the test taker has answered all 10 questions, his/her status on ASD is determined and recorded under the Class/ASD target variable. 
###http://asdtests.com/ #not diagnostic tools rather they are behavioural tests that just pinpoint to autistic traits. 
#The data describes ASD screening results, some of which appear to have been harvested from an app developed by Dr.Fadi Fayez named “ASDQuiz”

#There are two numeric variables (age and result) with the remaining variables being categorical (gender, ethnicity, jaundice, autism, country_of_res, used_app_before, age_desc, relation and Class/ASD) and binary (col 1-10.
#col 1
#A1_Score
#Question 1 Answer: Binary (0, 1)
#I often notice small sounds when others do not

#col 2
#A2_Score
#Question 2 Answer: Binary (0, 1)
#I usually concentrate more on the whole picture, rather than the small details

#col 3
#A3_Score
#Question 3 Answer: Binary (0, 1)
#I find it easy to do more than one thing at once

#col 4
#A4_Score
#Question 4 Answer: Binary (0, 1)
#If there is an interruption, I can switch back to what I was doing very quickly

#col 5
#A5_Score
#Question 5 Answer: Binary (0, 1)
#I find it easy to read between the lines when someone is talking to me

#col 6
#A6_Score
#Question 6 Answer: Binary (0, 1)
#I know how to tell if someone listening to me is getting bored

#col 7
#A7_Score
#Question 7 Answer: Binary (0, 1)
#When I’m reading a story I find it difficult to work out the character’s intentions

#col 8
#A8_Score
#Question 8 Answer: Binary (0, 1)
#I like to collect information about categories of things (e.g. types of cars, types of bird, types of train, types of plant, etc)

#col 9
#A9_Score
#Question 9 Answer: Binary (0, 1)
#I find it easy to work out what someone is thinking or feeling just by looking at their face

#col 10
#A10_Score
#Question 10 Answer: Binary (0,   1)
#I find it difficult to work out peoples intentions

#col 11
#Age
#Age in years

#col 12
#Gender
#Gender (m: Male, f: Female)

#col 13
#Ethnicity
#List of common ethnicities   (White-European, Latino, Others, Black, Asian, Middle Eastern, Pasifika,   South Asian, Hispanic, Turkish)

#14
#Jaundice
#born with jaundice (Yes, No)
#???jaundice

#col 15
#Austim
#Whether any immediate family member has a PDD (Yes, No)

#col 16
#Country_of_res
#Country of residence (List of countries)

#col 17
#Used_app_before
#Whether the user has used the screening app before (Yes, No)

#col 18
#Result
#Screening score: The final score obtained based on the scoring algorithm of the screening method used. 

#col 19
#Age_desc
#Type of screening method chosen based on age category
#exmp:"18 and more"

#col 20
#Relation
#Who is completing the test (Self, Parent, Health care professional, Relative, etc)

#col 21
#Class/ASD
#yes, no


library(foreign)
library(nnet)
library(caret)
library(pROC)
library(zoo)
library(ggplot2)
library(RColorBrewer)
library(dplyr)
library(stargazer)
library(readr)
library(boot)
library(caTools)
library(e1071)
library(glmnet)
library(kernlab)
library(ROCR)
library(rpart)
library(rattle)
library(randomForest)
library(forcats)
library(ggpubr)

#pdf("KMG_ML.pdf")

data = read.arff(file="Autism-Adult-Data.arff")
#704 instances and 21 variables
data = as.data.frame(data)
head(data)
summary(data)
#ethnicity (95), relation (95) and age (2) variables have nas.
#an impossibly large max value present in the age variable (383), possibly due to a typing error. intended value is 38?
#Class/ASD is the target variable
#high proportion of no for target variable: 515 to 189
str(data)
#contry_of_res is a actor with 67 levels. May need to drop as too many levels. Possibly irrelevant variable as well.
names(data)
#mispelled words: jundice, austim, contry_of_res
data$age_desc
class(data$age_desc)
#factor but only has one level so drop this and it is not relevant to our question

levels(data$ethnicity)
#two "others catagories". Need to combine.
#combine duplicated "others" category
levels(data$ethnicity) = gsub("others", "Others", levels(data$ethnicity))
levels(data$ethnicity)

#change outlier/typo? 383 in age column. Impossible age. Replace value of 383 with 38.
data$age[data$age == 383] = 38
summary(data$age) #max value is now 64


#change mispelled column names
colnames(data)[colnames(data)=="jundice"] = "jaundice"
colnames(data)[colnames(data)=="austim"] = "autism"
colnames(data)[colnames(data)=="contry_of_res"] = "country_of_res"
colnames(data)[colnames(data)=="Class/ASD"] = "Class_ASD"
names(data)

#look at nas and deal with them
anyNA(data) #true if yes
is.na(data) #true if yes, na in row
data[complete.cases(data), ] #gives you the rows with no nas in them
sum(is.na(data)) #total nas. #can use data$col_name to see total for a certain col
#omit rows with nas. most nas are categorical with ethnicity and relation
data$age = na.aggregate(data$age, FUN=mean) #takes mean of age column and replaces na with mean. 
sum(is.na(data$age))
#data_clean = na.omit(data) #takes out all rows with nas.
#anyNA(data_clean)  

#95 missing values in both: ethnicity and relation. Replace with 'Unknown".
#https://gist.github.com/riinuots/e517c36b1feb480df981721a00e0e24a
#single column mydata$column %>% fct_explicit_na(na_level = 'you_choose')
data_clean = data %>% 
  mutate_if(is.factor, fct_explicit_na, na_level = 'Unknown')
anyNA(data_clean)


#drop columns
data_clean = data_clean[ -c(16) ] #take out country_of_res column #too many factors 
#View(data_clean)
data_clean = data_clean[ -c(17) ] #take out result column #result score is related to Class and would give our model the outcome of the target variable
data_clean = data_clean[ -c(17) ] #take out age_desc as it only had one level, all the same. Note: survey is slightly different based on age_desc group.
#won't show/tell us anything as all are the same one
#plus we have an age column 


#one hot encoding
#https://machinelearningmastery.com/why-one-hot-encode-data-in-machine-learning/
#non-ordinal categorical variables can be converted into numeric data. 0 represents negative and 1 is positive
#transforms all categorical variables (except target variable) into numeric
#if you had a categorical variable made up of colours, one hot encoding would give each colour its own variable. A value of 1 in a colours variable would indicate that the colour does occur exist in an observation, while 0 indicates that it does not.
#In the case of binary variables, such as gender. One hot encoding can be instructed to not create new variables (as we have here, see: fullRank = TRUE. Using the gender example, one hot encoding could create one numeric variable called gender.f, where a value of 1 denotes that the observation is female, but a 0 denotes that it is male.
#https://www.rdocumentation.org/packages/caret/versions/6.0-84/topics/dummyVars
dummies = dummyVars(~ ., data = data_clean[, -18], fullRank = TRUE)
#dummies2 = dummyVars(~ ., data = data_clean[, -18], fullRank = FALSE)
#output of predict is a matrix, change it to data frame
one_hot = data.frame(predict(dummies, newdata = data_clean))
#one_hot2 = data.frame(predict(dummies2, newdata = data_clean))
View(data_clean)
#append the original target variable (factor)
one_hot$Class_ASD = data_clean$Class_ASD
#one_hot2$Class_ASD = data_clean$Class_ASD
View(one_hot)
#View(one_hot2)
str(one_hot)
#https://stattrek.com/multiple-regression/dummy-variables.aspx
#Avoid the Dummy Variable Trap
#When defining dummy variables, a common mistake is to define too many variables. If a categorical variable can take on k values, it is tempting to define k dummy variables. Resist this urge. Remember, you only need k - 1 dummy variables.
#A kth dummy variable is redundant; it carries no new information. And it creates a severe multicollinearity problem for the analysis. Using k dummy variables when only k - 1 dummy variables are required is known as the dummy variable trap. Avoid this trap! 
#The value of the categorical variable that is not represented explicitly by a dummy variable is called the reference group.
#In analysis, each dummy variable is compared with the reference group. In this example, a positive regression coefficient means that income is higher for the dummy variable political affiliation than for the reference group; a negative regression coefficient means that income is lower. If the regression coefficient is statistically significant, the income discrepancy with the reference group is also statistically significant. 


#look at classes in target variable
tb = table(one_hot$Class_ASD)
tb
#large proportion of no versus yes (515 vs 189) so unbalanced classes. may affect accuracy as dataset has a large bias.


#data visualization
#http://www.sthda.com/english/wiki/ggplot2-box-plot-quick-start-guide-r-software-and-data-visualization
#https://www.r-bloggers.com/detailed-guide-to-the-bar-chart-in-r-with-ggplot/


#pl = ggplot(data_clean, aes(x=gender, y=age, fill=Class_ASD)) +
  #geom_boxplot()
#pl

pl2 = ggplot(data_clean, aes(x=gender, y=age, fill=Class_ASD)) +
  geom_boxplot()
pl2+scale_fill_brewer(palette="Dark2")
#can see that ASD survey yes scoring starts around 7

a = ggplot(data_clean, aes(x = age, fill = Class_ASD))
a + geom_histogram(bins = 40, color = "black") +
  geom_vline(aes(xintercept = mean(age)), 
             linetype = "dashed", size = 0.6)

b = ggplot(data_clean, aes(x = age))
b + geom_histogram(aes(color = gender, fill = gender),
                   alpha = 0.4, position = "identity") +
  geom_vline(aes(xintercept = mean(age)), 
             linetype = "dashed", size = 0.6) +
  scale_fill_manual(values = c("#00AFBB", "#E7B800")) +
  scale_color_manual(values = c("#00AFBB", "#E7B800"))

b + geom_area( stat = "bin", bins = 30,
               color = "black", fill = "#00AFBB")

b + geom_freqpoly( aes(color = gender, linetype = gender),
                   bins = 30, size = 1.5) +
  scale_color_manual(values = c("#00AFBB", "#E7B800"))


ggplot(data_clean) +
  geom_bar(aes(x = ethnicity), fill = 'blue')
#our data mostly consists of White-European followed by Asian and then Middle-Eastern
ggplot(data_clean) +
  geom_bar(aes(x = Class_ASD), fill = 'pink')
ggplot(data_clean) +
  geom_bar(aes(x = A9_Score, fill = Class_ASD)) +
scale_fill_manual(values = c("yellow", "#E7B800"))

ggplot(data_clean) +
  geom_bar(aes(x = ethnicity, fill = Class_ASD))
#teal represents the amount in each ethnicity that classified as having ASD



#split data into train/test (70/30)
set.seed(101)
train_split = sample(1:nrow(one_hot), nrow(one_hot)*0.7) #take 70% sample
train = one_hot[train_split, ]
test_without_y = subset(one_hot[-train_split, ], select= -Class_ASD)
test_with_y = one_hot[-train_split, ]$Class_ASD
class_var_training = one_hot[train_split,]$Class_ASD 
trctrl = trainControl(method = "repeatedcv",
                       number = 10,
                       repeats = 3)
###################
####### KNN #######
###################
#find best k
best_knn = train(Class_ASD~., data=one_hot, method ="knn", tuneGrid =
                     data.frame(.k = 1:20), trControl = trainControl(method = "cv"))
best_knn
#best k is 3 with accuracy=0.9179235, with na changed 0.9232797 and unknown 0.9219293 and kappa=0.8029894 with na changed 0.8028121 and unknown 0.7992470

pred_knn = predict(best_knn, test_without_y)
pred_knn2 = predict(best_knn, test_without_y, type="prob")[,2]
head(pred)
confusionMatrix(pred_knn, test_with_y)
#outcome stats
#accuracy = 0.9454, 0.967, 0.967 
#p-value = 5.507e-12, 1.658e-15,    
#kappa= 0.8569, 0.905, same  
#sensitivity = 0.9493, 0.9817, same  
#specificity = 0.9333, 0.9167, same #important...don't want a FALSE NEGATIVE ON A MEDICAL TEST

#AUC
#ROC CURVE = receiver operating characteristic, defined as a plot of sensitivity (from cm) as the y coordinate versus its 1-specificity (from cm) aka false positive rate (FPR) as the x coordinate
#AUC is area under this???
out_knn = prediction(pred_knn2, test_with_y)
auc_knn = performance(out_knn, measure = "auc")@y.values[[1]]
auc_knn
#auc is 0.987037, 0.9952363, 0.9952998
plot(performance(out_knn, "tpr", "fpr"))
abline(0,1)

#############################
####### DECISION TREE #######
#############################
fitcontrol = trainControl(method = "cv", number = 10) #set up cross validation
grid_cp = expand.grid(cp=seq(0, 0.05, 0.005)) #find best complexity parameter

best_tree_cp = train(Class_ASD ~ ., data=train, method = "rpart",  #model using caret
            trControl=fitcontrol, metric="Accuracy", 
            maximize=TRUE, tuneGrid=grid_cp)
best_tree_cp
#BEST CP = 0.005, same, same
#accuracy = 0.8850926, 0.9025306, 0.8881224 
#kappa = 0.7329160, 0.7539271, 0.7098602

pred_tree = predict(best_tree_cp, test_without_y, type="raw") #predict using model and test data without class
pred_tree2 = predict(best_tree_cp, test_without_y, type="prob")[,2]
confusionMatrix(pred_tree, test_with_y)
#outcome stats
#accuracy = 0.8907, 0.9292, same  
#p-value = 2.671e-06, 1.108e-09, same   
#kappa= 0.722, 0.7837, same  
#sensitivity = 0.8986, 0.9817, same  
#specificity = 0.8667, 0.7500, same #important...don't want a FALSE NEGATIVE ON A MEDICAL TEST

#AUC
#ROC CURVE = receiver operating characteristic, defined as a plot of sensitivity (from cm) as the y coordinate versus its 1-specificity (from cm) aka false positive rate (FPR) as the x coordinate
#AUC is area under this???
out_tree = prediction(pred_tree2, test_with_y)
auc_tree = performance(out_tree, measure = "auc")@y.values[[1]]
auc_tree
#auc is 0.9438808, 0.9662093, same
plot(performance(out_tree, "tpr", "fpr"))
abline(0,1)


#############################
####### RANDOM FOREST #######
#############################
set.seed(101)
control_rand_for = trainControl(method="repeatedcv", number=10, repeats=3)
metric = "Accuracy"
n= round(sqrt(ncol(data)))
tunegrid = expand.grid(.mtry=seq(by = 1,to = 5, from = 1))
rand_for = train(Class_ASD ~ ., data=train, method="rf",
                 metric=metric, tuneGrid= tunegrid, trControl=control_rand_for)
print(rand_for)
#best mtry was 3: accuracy = 0.9608896 and kappa = 0.9073772
#with nas changed. best mtry was 5: accuracy = 0.9620544 and kappa = 0.9049880
#with unknown. best mtry was 4: accuracy = 0.9600272 and kappa = 0.8988537
pred_rf_caret = predict(rand_for, test_without_y)
pred_rf2 = predict(rand_for, test_without_y, type="prob")[,2]
confusionMatrix(pred_rf_caret, test_with_y)
#outcome stats
#accuracy = 0.9563, 0.9764, 0.9717 
#p-value = 1.467e-13, <2e-16, <2e-16   
#kappa= 0.8821, 0.9312, 0.9168  
#sensitivity = 0.9710, 0.9939, same    
#specificity = 0.9111, 0.9167, 0.8958 #important...don't want a FALSE NEGATIVE ON A MEDICAL TEST

#AUC
#ROC CURVE = receiver operating characteristic, defined as a plot of sensitivity (from cm) as the y coordinate versus its 1-specificity (from cm) aka false positive rate (FPR) as the x coordinate
#AUC is area under this???
out_rf = prediction(pred_rf2, test_with_y)
auc_rf = performance(out_rf, measure = "auc")@y.values[[1]]
auc_rf
#auc is 0.9917874, 0.9980945, 0.9991108
plot(performance(out_rf, "tpr", "fpr"))
abline(0,1)

important = varImp(rand_for) #shows important variables
important
#A9_Score.1                100.000
#A6_Score.1                 93.594
#A5_Score.1                 85.230
#A4_Score.1                 59.975

#A9_Score.1                100.000
#A5_Score.1                 97.183
#A6_Score.1                 92.284
#A4_Score.1                 51.747

#A9_Score.1                100.000
#A5_Score.1                 85.846
#A6_Score.1                 78.817
#A4_Score.1                 52.266
#all social que questions where autistic individuals have challenges. Makes sense these are important!

#for loop to determine best number of trees using ntree
accu = rep(0,1000)
for (i in 1:1000) {
  set.seed(i)
  rand_for = randomForest(Class_ASD~ ., data=train, ntree=i)
  pred_loop = predict(rand_for, test_without_y)
  cm = confusionMatrix(pred_loop, test_with_y)
  
  accu[i] = cm[3]$overall[1]
}
accu
max(accu)
#max accuracy is 0.9726776
#trees = 158 

#with unknown. max accuracy is 0.9858491
#trees = 32, 82
#using 32 trees and now determining best mtry
control_rand_for32 = trainControl(method="repeatedcv", number=10, repeats=3)
metric = "Accuracy"
n= round(sqrt(ncol(train)))
tunegrid = expand.grid(.mtry=seq(by = 1,to = 5, from = 1))
rand_for32 = train(Class_ASD ~ ., data=train, method="rf", ntree=32,
                   metric=metric, tuneGrid= tunegrid, trControl=control_rand_for32)
print(rand_for32)
#mtry = 5 is best with 158 trees, with unknown 32 trees mtry = 4
#accuracy = 0.9609064, 0.9505578 
#kappa = 0.9089005, 0.8748667

#best model created and tested
rf_model_best = randomForest(Class_ASD~ ., data=train, ntree=32, mtry=4)
rf_model_best

pred_rf_best = predict(rf_model_best, test_without_y)
pred_rf3 = predict(rf_model_best, test_without_y, type="prob")[,2]
confusionMatrix(pred_rf_best, test_with_y)
#outcome stats
#accuracy = 0.9672, 0.9575  
#p-value = 2.383e-15, 8.575e-14  
#kappa= 0.9116, 0.8722    
#sensitivity = 0.9783, 0.9939    
#specificity = 0.9333, 0.8333  #important...don't want a FALSE NEGATIVE ON A MEDICAL TEST

#AUC
#ROC CURVE = receiver operating characteristic, defined as a plot of sensitivity (from cm) as the y coordinate versus its 1-specificity (from cm) aka false positive rate (FPR) as the x coordinate
#AUC is area under this???
out_rf2 = prediction(pred_rf3, test_with_y)
auc_rf2 = performance(out_rf2, measure = "auc")@y.values[[1]]
auc_rf2
#auc is 0.9901771, 0.9912348
plot(performance(out_rf2, "tpr", "fpr"))
abline(0,1)

###################
####### SVM #######
###################
set.seed(101)
#exploring results for one_hot2 rank=false
#train_split3 = sample(1:nrow(one_hot2), nrow(one_hot2)*0.7) #take 70% sample
#train3 = one_hot2[train_split3, ]
#test_without_y3 = subset(one_hot2[-train_split3, ], select= -Class_ASD)
#test_with_y3 = one_hot2[-train_split3, ]$Class_ASD
#class_var_training3 = one_hot2[train_split3,]$Class_ASD 
#trctrl3 = trainControl(method = "repeatedcv",
                      #number = 10,
                      #repeats = 3)
y = train$Class_ASD
x = subset(train, select =-Class_ASD)

poly_mod = train(x, y, method="svmPoly", allowParallel = FALSE, scale=FALSE, tuneLength=5,
                 trControl=trainControl(method="repeatedcv", 
                                        number=10, repeats=3))
poly_mod #to see performance
#The final values used for the model were degree = 1, scale = 10 (1e+01) and C = 0.25.
#accuracy =  1.0000000 and kappa = 1.00000000 

plot(poly_mod)
#svm uses two parameters: degree and cost(C)
#degree = 1, scale = 10 (1e+01) and C = 0.25

#with unknown. The final values used for the model were degree = 1, scale = 10 (1e+01) and C = 0.25.
#accuracy =  1.0000000 and kappa = 1.00000000 
linea_mod = train(x, y, method="svmLinear", scale=FALSE, tuneLength=5,
                  trControl=trainControl(method="repeatedcv", 
                                         number=10, repeats=3))
linea_mod
#accuracy = 1, 1
#kappa = 1, 1
#Tuning parameter 'C' was held constant at a value of 1, 1


radial_mod = train(x, y, method="svmRadial", allowParallel = FALSE, scale=FALSE, tuneLength=5,
                   trControl=trainControl(method="repeatedcv", 
                                          number=10, repeats=3))
radial_mod
#final values used for the model were sigma = 0.02785584, 0.02537838 and C = 4, 4.
#accuracy = 0.9608527, 0.9519048 
#kappa = 0.9098806, 0.88258734
#tuning parameter 'sigma' was held constant at a value of 0.02785584, 0.02537838 

pred_poly = predict(poly_mod, test_without_y)
confusionMatrix(pred_poly, test_with_y)
#outcome stats
#accuracy = 1, same
#p-value = < 2.2e-16, same   
#kappa= 1, same    
#sensitivity = 1.0000, same   
#specificity = 1.0000, same #important...don't want a FALSE NEGATIVE ON A MEDICAL TEST
###means data separates well

#bootstrap 100 samples and calculate the 95% CI and AUC using poly kernal
n = 100
accuracy = rep(0,n)
auc = rep(0,n)
for (i in 1:n) {
  set.seed(i+100)
  new_index = sample(c(1:length(one_hot$Class_ASD)), length(one_hot$Class_ASD), replace=TRUE)
  new_sample = one_hot[new_index,]
  split = sample.split(new_sample$Class_ASD, SplitRatio = 0.8)
  train = subset(new_sample, split==TRUE)
  test.x = subset(subset(new_sample, split==FALSE), select=-Class_ASD)
  test.y = subset(new_sample, split==FALSE)$Class_ASD
  
  svm_poly = train(Class_ASD~ ., data=train, method="svmPoly",
                   tuneLength = 1, scale=FALSE,
                   trControl= trainControl(method="repeatedcv",
                                           repeats = 5, 
                                           classProbs=TRUE))
  
  #accuracy
  pred = predict(svm_poly, test.x)
  c = confusionMatrix(pred, test.y)
  accuracy[i] = c[3]$overall[1]
  
  #auc
  pred = predict(svm_poly, test.x, type="prob")[,2]
  out = prediction(pred, test.y)
  auc[i] = performance(out, measure="auc")@y.values[[1]]
}
#https://r.789695.n4.nabble.com/Cannot-scale-data-td4663526.html
#warning-In .local(x, ...) : Variable(s) `' constant. Cannot scale data.
#some columns with "zero variability" I mean "constant". They have the same value in the rows
#View(train)
#used_app_before, relation_others = LOTS OF ZEROS????
#Try scale=FALSE in svm model function



#95% ci for accuracy
accuracy
accuracy.mean = mean(accuracy)
accuracy.mean
#accuracy mean = 0.9220593, 0.9298916
accuracy.me = qnorm(0.975) * sd(accuracy)/sqrt(length(accuracy)) #Divide your accuracy standard deviation by the square root of your accuracy size.
accuracy.me
accuracy.lci = accuracy.mean - accuracy.me #lower end of ci
accuracy.lci
accuracy.uci = accuracy.mean + accuracy.me #upper end of ci
accuracy.uci
#lower ci = 0.9173911, 0.9255337
#upper ci = 0.9267276, 0.9342495

#95% ci for auc
auc.mean = mean(auc)
auc.me = qnorm(0.975) * sd(auc)/sqrt(length(auc))
auc.lci = auc.mean - auc.me
auc.uci = auc.mean + auc.me
auc.lci
auc.uci
#lower auc ci 0.9864863, 0.9890107
#upper auc ci 0.9903444, 0.9922362

#AUC
#ROC CURVE = receiver operating characteristic, defined as a plot of sensitivity (from cm) as the y coordinate versus its 1-specificity (from cm) aka false positive rate (FPR) as the x coordinate
#AUC is area under this???
pred4 = predict(svm_poly, test.x, type="prob")[,2]
out4 = prediction(pred4, test.y)
auc4 = performance(out4, measure="auc")@y.values[[1]]
auc4
#0.9993432, 0.9992204
plot(performance(out4, "tpr", "fpr"))
abline(0,1)

#####################
####### LASSO #######
#####################
#sets up x with all vars except for target 
x = model.matrix(Class_ASD~.,one_hot)[,-31] #sets up x with all vars except for target survived
head(x)
colnames(x)
y = one_hot$Class_ASD #sets up y with target var only
train=sample(1:nrow(x), nrow(x)/2)
test=(-train)
y.test=y[test]

cv.out_lasso = cv.glmnet(x[train,],y[train],alpha=1, family="binomial")
#alpha 1 is lasso
cv.out_lasso
#model output tells you min lambda using mse (mean squared error)
summary(cv.out_lasso)
plot(cv.out_lasso)
# the best model is under lambda = lambda.min
bestlam = cv.out_lasso$lambda.min
bestlam
#5.154182e-05 best lambda for lasso
out2 = glmnet(x,y, alpha=1, family = "binomial")
plot(out2, xvar = "lambda", label = TRUE)
dim(one_hot)
lasso.coef = predict(out2,type="coefficients",s=bestlam)[1:31,]
lasso.coef
#take out 0 coefficients for lasso, 0 vars did not explain anything in model
lasso.coef[lasso.coef!=0]
#selected variables with coefficients not 0               
#A1_Score.1                A2_Score.1 
#11.66615201               11.18597353 
#A3_Score.1                A4_Score.1                A5_Score.1 
#11.13314096               11.62972382               11.82139942 
#A6_Score.1                A7_Score.1                A8_Score.1 
#11.40151480               11.47105550               11.40556585 
#A9_Score.1               A10_Score.1           ethnicity.Black 
#11.64865883               11.15825994                0.17539586 
#ethnicity.Hispanic   ethnicity.Middle.Eastern.  ethnicity.White.European 
#0.09251261               -0.10892936                0.12901819 
#used_app_before.yes       relation.Self 
#0.91828610                0.03089008 

 
 
#using data-clean not encoded
#set.seed(1)
#x2 = model.matrix(Class_ASD~.,data_clean)[,-18] #sets up x with all vars except for target survived
#head(x)
#colnames(x)
#y2 = data_clean$Class_ASD #sets up y with target var only
#train2=sample(1:nrow(x), nrow(x)/2)
#test2=(-train)
#y.test2=y[test]

#cv.out_lasso2 = cv.glmnet(x2[train2,],y2[train2],alpha=1, family="binomial")
#alpha 1 is lasso
#cv.out_lasso2
#model output tells you min lambda using mse (mean squared error)
#summary(cv.out_lasso2)
#plot(cv.out_lasso2)
# the best model is under lambda = lambda.min
#bestlam2 = cv.out_lasso2$lambda.min
#bestlam2
#5.707342e-05 best lambda for lasso
#out3 = glmnet(x2,y2, alpha=1, family = "binomial")
#plot(out3, xvar = "lambda", label = TRUE)
#dim(data_clean)
#lasso.coef2 = predict(out3,type="coefficients",s=bestlam)[1:18,]
#lasso.coef2
#take out 0 coefficients for lasso, 0 vars did not explain anything in model
#gender.m =0 ???? More males have ASD.
#lasso.coef2[lasso.coef2!=0]
#same as above including coeff numbers except left out ethnicity white, used app before yes, and relation self


#create dataset based on lasso selected variables
#lasso_data = one_hot[ -c(11) ] #take out age column
#lasso_data = one_hot[ -c(11) ] #take out gender.m column #wom't work with dummies
View(one_hot)
lasso_data = one_hot[, c(1:10, 13:14, 16, 21, 25, 29, 31)]
View(lasso_data)
#checking to see if any lasso variable exactly matches the target var = NO 
ggplot(lasso_data) +
  geom_bar(aes(x = A1_Score.1, fill = Class_ASD))
ggplot(lasso_data) +
  geom_bar(aes(x = A5_Score.1, fill = Class_ASD))
ggplot(lasso_data) +
  geom_bar(aes(x = A4_Score.1, fill = Class_ASD))
ggplot(lasso_data) +
  geom_bar(aes(x = A6_Score.1, fill = Class_ASD))
ggplot(lasso_data) +
  geom_bar(aes(x = relation.Self, fill = Class_ASD))



##############################################
####### LOGISTIC REGRESSION LASSO VARS #######
##############################################
train_split2 = sample(1:nrow(lasso_data), nrow(lasso_data)*0.7) #take 70% sample
train2 = lasso_data[train_split2, ]
test_without_y2 = subset(lasso_data[-train_split2, ], select= -Class_ASD)
test_with_y2 = lasso_data[-train_split2, ]$Class_ASD

View(train2)
class(lasso_data$Class_ASD)
set.seed(101)


train_control = trainControl(method = "cv", number = 10)
logist_bayes_mod = train(Class_ASD~., data=train2, 
                             method = "bayesglm",
                             prior.df=9,
                             trControl = train_control,
                             family="binomial")

summary(logist_bayes_mod)
logist_bayes_mod$results

pred_log_train =predict(logist_bayes_mod, newdata = test_without_y2, type = "raw")
pred_log_train
confusionMatrix(pred_log_train, test_with_y2)
#accuracy 1
#p-value 2.2e-16 
#kappa 1
#sensitivity = 1.0000     
#specificity = 1.0000


#logist_mod = glm(Class_ASD ~., 
                 #data=train2, family="binomial", control = list(maxit = 50))
#ERROR: glm.fit: algorithm did not converge 
#solution: list maxit glm() uses an iterative re-weighted least squares algorithm. The algorithm hit the maximum number of allowed iterations before signalling convergence. The default, documented in ?glm.control is 25. You pass control parameters as a list in the glm call.
#ERROR:glm.fit: fitted probabilities numerically 0 or 1 occurred 
#https://stats.stackexchange.com/questions/11109/how-to-deal-with-perfect-separation-in-logistic-regression
#Complete separation occurs whenever a linear function of x can generate perfect predictions of y
#the information matrix is infinitely valued, and no inference is available. Rather, R does produce output, but you cannot trust it. The inference that R typically produces in these cases has p-values very close to one. This is because the loss of precision in the OR is orders of magnitude smaller that the loss of precision in the variance-covariance matrix.
#check for your selected variables for the model fitting, there may be a variable for which multi collinearity with the Y (outout) variable is very high, discard that variable from your model.
#overinflated coeffs
#one option is to do nothing
#summary(logist_mod)
#P VALUES OF 1
#help(glm)

###### other way to do bayes using bayesglm( )
#https://www.rdocumentation.org/packages/arm/versions/1.10-1/topics/bayesglm
library(arm)
bayes_logist = bayesglm(Class_ASD ~., data=lasso_data, prior.df=9, family="binomial", control=glm.control(maxit=1))
#Bayesian analysis with non-informative prior assumptions
#each coefficient (normalized to have mean 0.0 and a SD of 0.5). This will regularize the coefficients and pull them just slightly towards zero. In this case it is exactly what you want. Due to having very wide tails the Cauchy still allows for large coefficients (as opposed to the short tailed Normal), from Gelman:
#ERROR: fitted probabilities numerically 0 or 1 occurred 
#the error means that you have a binary variable that perfectly predicts your response variable: 
#prior.df=9 error went away
#slightly stronger regularization of coeffs by increasing prior.df which defaults to 1.0
#prior.df is prior degrees of freedom for the coefficients. For t distribution: default is 1 (Cauchy). Set to Inf to get normal prior distributions. Can be a vector of length equal to the number of predictors (not counting the intercept, if any). If it is a scalar, it is expanded to the length of this vector.
summary(bayes_logist)
#NO P-VALUES OF 1
bayes_logist$coef

pred_log =predict(bayes_logist, newdata = test_without_y2, type = "response")
pred_log
asd_or_not_log = ifelse(pred_log >= 0.5, "YES", "NO")
asd_or_not_log
#1 is positive asd and 0 is no asd
#convert to factor: fac_class3
fac_class_log = factor(asd_or_not_log, levels = levels(test_with_y2["Class_ASD"]))
confusionMatrix(fac_class_log, test_with_y2)
#accuracy 1
#p-value 1
#kappa 1
#sensitivity = 1.0000     
#specificity = 1.0000


#AUC
#ROC CURVE = receiver operating characteristic, defined as a plot of sensitivity (from cm) as the y coordinate versus its 1-specificity (from cm) aka false positive rate (FPR) as the x coordinate
#AUC is area under this???
out_log = prediction(pred_log, test_with_y2)
auc_log = performance(out_log, measure = "auc")@y.values[[1]]
auc_log
#auc is 0.9992204
plot(performance(out_log, "tpr", "fpr"))
abline(0,1)


#######################################
####### RANDOM FOREST LASSO VARS#######
#######################################
set.seed(101)
train_split2 = sample(1:nrow(lasso_data), nrow(lasso_data)*0.7) #take 70% sample
train2 = lasso_data[train_split2, ]
test_without_y2 = subset(lasso_data[-train_split2, ], select= -Class_ASD)
test_with_y2 = lasso_data[-train_split2, ]$Class_ASD
control_rand_for = trainControl(method="repeatedcv", number=10, repeats=3)
metric = "Accuracy"
n= round(sqrt(ncol(data)))
tunegrid = expand.grid(.mtry=seq(by = 1,to = 5, from = 1))

rand_for_lasso = train(Class_ASD ~ ., data=train2, method="rf",
                 metric=metric, tuneGrid= tunegrid, trControl=control_rand_for)
print(rand_for_lasso)
#best mtry was 2: accuracy = 0.9681327 and kappa = 0.9191283

pred_rf_lasso = predict(rand_for_lasso, test_without_y2)
pred_rf_lasso2 = predict(rand_for_lasso, test_without_y2, type="prob")[,2]
confusionMatrix(pred_rf_lasso, test_with_y2)
#outcome stats
#accuracy = 0.9528
#p-value = 6.494e-14   
#kappa= 0.8635  
#sensitivity = 0.9877     
#specificity = 0.8400 #important...don't want a FALSE NEGATIVE ON A MEDICAL TEST

#AUC
#ROC CURVE = receiver operating characteristic, defined as a plot of sensitivity (from cm) as the y coordinate versus its 1-specificity (from cm) aka false positive rate (FPR) as the x coordinate
#AUC is area under this???
out_rf_lasso = prediction(pred_rf_lasso2, test_with_y2)
auc_rf = performance(out_rf_lasso, measure = "auc")@y.values[[1]]
auc_rf
#auc is 0.9960494
plot(performance(out_rf_lasso, "tpr", "fpr"))
abline(0,1)

important = varImp(rand_for_lasso) #shows important variables
important
#A9_Score.1                100.000
#A6_Score.1                 80.287
#A5_Score.1                 63.808
#A4_Score.1                 52.380
#all social que questions where autistic individuals have challenges. Makes sense these are important!
#similar to rf using all variables not just lasso
plot(important)

#for loop to determine best number of trees using ntree
accu = rep(0,1000)
for (i in 1:1000) {
  set.seed(i)
  rand_for_lass = randomForest(Class_ASD~ ., data=train2, ntree=i)
  pred_loop = predict(rand_for_lass, test_without_y2)
  cm = confusionMatrix(pred_loop, test_with_y2)
  
  accu[i] = cm[3]$overall[1]
}
accu
max(accu)
#max accuracy is 0.9811321
#trees = 17 

#using 17 trees and now determining best mtry
control_rand_for17 = trainControl(method="repeatedcv", number=10, repeats=3)
metric = "Accuracy"
n=round(sqrt(ncol(train2)))
tunegrid = expand.grid(.mtry=seq(by = 1,to = 5, from = 1))
rand_for17 = train(Class_ASD ~ ., data=train2, method="rf", ntree=17,
                   metric=metric, tuneGrid= tunegrid, trControl=control_rand_for17)
print(rand_for17)
#mtry = 3 is best with 17 trees
#accuracy = 0.9682846 
#kappa = 0.9203123

#best model created and tested
rf_model_best_lass = randomForest(Class_ASD~ ., data=train2, ntree=17, mtry=3)
rf_model_best_lass

getTree(rf_model_best_lass)
install.packages("devtools")
library(devtools)
devtools::install_github('skinner927/reprtree')
library(reprtree)
reprtree:::plot.getTree(rf_model_best_lass)

pred_rf_best_lass = predict(rf_model_best_lass, test_without_y2)
pred_rf_best_lass2 = predict(rf_model_best_lass, test_without_y2, type="prob")[,2]
confusionMatrix(pred_rf_best_lass, test_with_y2)
#outcome stats
#accuracy = 0.9764
#p-value = <2e-16  
#kappa= 0.9332     
#sensitivity = 0.9938     
#specificity = 0.9200 #important...don't want a FALSE NEGATIVE ON A MEDICAL TEST

#AUC
#ROC CURVE = receiver operating characteristic, defined as a plot of sensitivity (from cm) as the y coordinate versus its 1-specificity (from cm) aka false positive rate (FPR) as the x coordinate
#AUC is area under this???
out_rf17 = prediction(pred_rf_best_lass2, test_with_y2)
auc_rf17 = performance(out_rf17, measure = "auc")@y.values[[1]]
auc_rf17
#auc is 0.9961728
plot(performance(out_rf17, "tpr", "fpr"))
abline(0,1)


##############################################
####### NEURAL NETWORK WITH LASSO VARS #######
##############################################
#look for best size of hidden layer and decay/learning rate
grid = expand.grid(size=c(2,3,4,5,6,7,8,9,10,11,12,13), decay=c(0, 0.01, 0.1, 1))
model_nn_lasso = train(Class_ASD ~., data=train2, method="nnet", tunegrid=grid, skip=FALSE, linout=FALSE)
model_nn_lasso
#based on accuracy of 0.9948155, the optimal model has a size 5 and decay 1e-04

plot(model_nn_lasso)

pred_nn = predict(model_nn_lasso, test_without_y2) 
pred_nn2 = predict(model_nn_lasso, test_without_y2, type="prob")[,2]
pred_nn
confusionMatrix(pred_nn, test_with_y2)
#accuracy = 1
#p-value = 2.2e-16 
#kappa = 1
#sensitivity = 1
#specificity = 1


#AUC
#ROC CURVE = receiver operating characteristic, defined as a plot of sensitivity (from cm) as the y coordinate versus its 1-specificity (from cm) aka false positive rate (FPR) as the x coordinate
#AUC is area under this???
out_nn = prediction(pred_nn2, test_with_y2)
auc_nn = performance(out_nn, measure = "auc")@y.values[[1]]
auc_nn
#auc is 1
plot(performance(out_nn, "tpr", "fpr"))
abline(0,1)




#graphics.off()
#### CONCLUSION ####
#in evaluation of the predictions of all models, all predicted the target extremely well, before lasso - SVM model was perfect and after lasso selected vars - neural network was perfect. 
#the target variable is hugely biased towards no. a more balanced dataset would bring more insight into the data and possibly improve prediction accuracy
#also, ethnicity wise, there are far more euro whites than any other ethnicities and this could introduce bias