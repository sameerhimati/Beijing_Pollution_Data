# here you will need to set your working directory to
# the place where you downloaded the data to
# in this directory you should have the following two files:
# trainingdata.csv
# test_predictors.csv
# those you will download from the kagge website
getwd()

setwd('~/Desktop/Prediction')

# read in the
d.train = read.csv('~/Desktop/Prediction/trainingdata2.csv')
d.test = read.csv('~/Desktop/Prediction/test_predictors2.csv')
numpy = read.csv('~/Desktop/Prediction/sameerSubmission.csv')
attach(d.train)
# in the following we will generate predictions using the sample mean
# the following code repeats the sample mean of y 30000 times
# you will need to replace the following by better ways of generating predictions later

# now bring in the right format for submission
# the first column wiht name idshould be the number of the observation
# the second line is your prediction
mean_pred = numpy
example_pred = data.frame(cbind(1:30000,mean_pred))
names(example_pred) = c('id','y')

# this will create a .csv file in your working directory
# this is the file you should upload to the competition website
write.csv(example_pred,file='linmod.csv',row.names=FALSE)

