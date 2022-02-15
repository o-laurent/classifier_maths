linear_classification : fmatrix.cu cuda_stuff.cu stable_softmax.cu read_csv.cu evaluate_accuracy.cu classifier_math.cu normalization.cu linear_classification.cu preprocess_data.cu
	nvcc -arch=sm_86 -g -G -lcublas -lcusolver linear_classification.cu normalization.cu stable_softmax.cu read_csv.cu preprocess_data.cu evaluate_accuracy.cu fmatrix.cu classifier_math.cu cuda_stuff.cu -o linear_classification

check: linear_classification
	cuda-gdb -batch -x tmp.txt ./linear_classification