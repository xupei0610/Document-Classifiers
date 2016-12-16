#!/bin/sh

#  run.sh
#  Document Classifier
#
#  Created by Pei Xu on 12/4/16.
#  Copyright © 2016 Pei Xu. All rights reserved.

echo "This is an autoscript to run test for the Ridge Regression Classifier coded by Pei Xu."
echo "It will test the following combaination of parameters:"
echo "  Penalty Paramters: {0.01, 0.05, 0.1, 0.5, 1.0, 10.0}"
echo "  Token Model: {bag-of-words, 5-character-grams}"
echo "  Freq. Represent. Form: {TF, BINARY, TFIDF}"

DATA_FILE[0]="_word"
DATA_FILE[1]="_char"

FREQ_FORM[0]="TF"
FREQ_FORM[1]="BINARY"
FREQ_FORM[2]="TFIDF"

DATA_FOLDER="data-files"
PREFIX="20newsgroups"

LOG_FOLDER="log"


val_train_file="${DATA_FOLDER}/${PREFIX}_ridge.train"
val_test_file="${DATA_FOLDER}/${PREFIX}_ridge.val"
input_rlabel_file="${DATA_FOLDER}/${PREFIX}.rlabel"
train_file="${DATA_FOLDER}/${PREFIX}.train"
test_file="${DATA_FOLDER}/${PREFIX}.test"
class_file="${DATA_FOLDER}/${PREFIX}.class"


run_test() {

for s in "${DATA_FILE[@]}"
do
for c in "${FREQ_FORM[@]}"
do

outfile_suf="$(date +"%H-%M-%S_%m-%d-%Y")"

input_file="${DATA_FOLDER}/${PREFIX}${s}.ijv"
f_label_file="${DATA_FOLDER}/${PREFIX}${s}.clabel"
output_file="${LOG_FOLDER}/result_ridge${s}_${c}_${outfile_suf}"
log_file="${LOG_FOLDER}/log_ridge${s}_${c}_${outfile_suf}"

./regression ${input_file} ${input_rlabel_file} ${train_file} ${test_file} ${class_file} ${f_label_file} ${c} ${output_file} ${val_train_file} ${val_test_file} 2>&1 | tee ${log_file}


outfile_suf="$(date +"%H-%M-%S_%m-%d-%Y")"
output_file="${LOG_FOLDER}/result_centroid${s}_${c}_${outfile_suf}"
log_file="${LOG_FOLDER}/log_centroid${s}_${c}_${outfile_suf}"
./centroid ${input_file} ${input_rlabel_file} ${train_file} ${test_file} ${class_file} ${f_label_file} ${c} ${output_file} 2>&1 | tee ${log_file}

done
done
}

while
echo "\nInput [Y]/[N] to start or exit test: "
read inp
do
if [ "$inp" == "Y" ] || [ "$inp" == "y" ]; then
break
elif [ "$inp" == "N" ] || [ "$inp" == "n" ]; then
exit 1
fi
done

run_test
