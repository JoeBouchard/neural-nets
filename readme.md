Notes on the code used for the CS 5073 Project:

All code files are in the Autoencoder directory. The testingdata, trainingdata, and Happiness_data.csv files and folders are for future work.

The file "latest_autoencoder" includes a class for the standard autoencoder
and a class for the sequential autoencoder. It also includes a function to 
test one of each on the MNIST clothing dataset; to run that function one
must uncomment the "layers.Reshape" in line 48. For other uses that line
should remain commented.

The file "data_gen_and_test" includes all functions used to generate data
used in the report. Noteworthy functions include:

"testable": This function runs many autoencoders, both standard and
sequential and puts their results into a text file. When the text is copied
into a LATEX compiler such as Overleaf.com, it becomes a pair of nicely
formatted table.

"makecsv": This function runs 100 sequential and 100 standard autoencoders
and records the average mean squared error and standard deviation of each
model in a comma-separated-values file (readable by Excel). The variable
"epochs" is the number of epochs the sequential autoencoder will
train its first sub-autoencoder before training the next. The variable
"count" is the number of repetitions, and also the number of times data will
be recorded. For best sequential autoencoder performance, epochs>10.

"interpret": This runs a single model, recording the sum-squared error of
each data element individually into a csv file. If it is running a sequential
autoencoder, it will train the first sub-autoencoder for the specified amount
while recording its performance (without any training of the second) and then
trains the second sub-autoencoder for the specified amount, recording the
performance (this time of the whole model). It isn't built to handle a
sequential autoencoder with more than two sub-autoencoders.

To run any of these functions, either find their calls in the code and
uncomment them or write your own function calls with the particular
parameters you want to use.