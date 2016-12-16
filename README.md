## Introduction
This is an implementation of document classification using centroid-based classifier and ridge regression classifier.

## How to Run

Use

    make centroid

to compile the centroid-based classifier, while use

    make regression

to complile the ridge regression classifier.


## Details of the Centroid-Based Classifier
- Three representation forms, binary frequency, term frequency and tf-idf. are supported when vectorizing document data.
- Only cosine similarity is supported. The cosine similarity is calculated via
      cos(x, c) = < x, c >/norm(c, 2)
where x is a vectorized data object, c is a centroid of a binary classifier calculated using the training data, `<.,.>` is the inner product of a object and a centroid, and `norm(., 2)` is the l2-norm of a centroid. For any vectorized data object, it has been normalized before participating in computation so that norm(x,2) == 1.
- Centroid is obtained as the mean of the corresponding normalized, vectorized training objects.
- Fore each test object, the algorithm looks for the nearest centroid and label the object using the label of classifier who has the nearest centroid.

# Details of the Ridge Regression Classifier
- The coordinate descent method is employed to find the solution, i.e. the weight coefficient w in

      w = argmin norm(Xw-y, 2) + 𝝺 norm(w, 2)
             w
and the zero vector is chosen as the initial value of w.
- The program using training data X to train a binary classifier for each class and then label a test object, p, as the class whose w lead the max value of `< p, w >`.
- For each binary classifier, the element in y is 1 if the corresponding training object belongs to that class, and is -1 otherwise.
- Defaultly, the classifier tries to classify a validation data set for the sake of choosing a proper value of 𝝺, and then classify the test objects using 𝝺 with the chosen value.

# Parameters of the `main` program
- data-file: the file record each document's the name, freatures, frequency of freatures, which includes both of training and test objects. Each of line has three elements which are separated by a space. The first element is the name or id of the document, the second element is the name or id of one of the docuemnt's features, and the third element is the frequency of that freature in this document. It should be the term frequency if you would like to use term frequency or tf-idf in the process of classification.
- rlabel-file: each line has multiple elements separated by a space. The first element is the id or name of a document, and the last element is the real name of that document. If you do not want to output these features, just pass a non-existing file as the parameter.
- train-file: each line is the id or name of the training objects. It should be consistent with the datai in data-fiel.
- test-file: same to the above one but it is used for the test objects.
- class-file: each line has two elements separated by a space. The first element is the id or name of the document, and the second element is the id or name of class to which this docuemnt belongs. Training objects whose class information does not exist in thie file would be ignored. If you want to evaluate the classification solution, the class information of the test objects also should be provided in this file.
- feature-file: the content that each feature represents. The line number is the corresponding id of features in the data-file. Then progream will output the features with highest weights if you provide this parameters. If you do not want to output these features, just pass a non-existing file as the parameter.
- representation-form: the way to represent the vectorized document data. Binary for bianry frequency, tf for term frequency, tfidf for tf-idf.
- output-file: the file that will contain the classfication solution. Each line containt two elements separated by a comma. The first element is a test document's name or id who is provided in rlabel-file or that in data-file if no legal rlabel-file is provided.
- val-train-file: this is for ridge regression. It has the similar structure to train-file, and is used to training a classifier for the validation test data before classifying the test objects in test-file. The classification solution will be evaluated in order to determine the proper value of 𝝺. It can be a subset of or exactly the training objects.
- val-test-file: this is for ridge regression. It has the similar structure to test-file. However, due to that the calssfication to the objects provided in this file will be evaluated, objects provided in this file would be ignored if there is their classification iformation in class-file.

I know some of these parameters are quite strange. They are formulated by the instructor the data mining course. So ...

## Finally

Contact me if you have any question.

Pei Xu, xuxx0884@umn.edu
