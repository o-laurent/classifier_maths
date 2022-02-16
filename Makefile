all: lin_cl batch_lin_cl shared

lin_cl: src/fmatrix.cu src/cuda_stuff.cu src/stable_softmax.cu src/read_csv.cu src/evaluate_accuracy.cu src/classifier_math.cu src/normalization.cu src/linear_classification.cu src/preprocess_data.cu
	nvcc -arch=sm_86 -g -G -lcublas -lcusolver src/linear_classification.cu src/normalization.cu src/stable_softmax.cu src/read_csv.cu src/preprocess_data.cu src/evaluate_accuracy.cu src/fmatrix.cu src/classifier_math.cu src/cuda_stuff.cu -o linear_classification

batch_lin_cl: src/fmatrix.cu src/cuda_stuff.cu src/stable_softmax.cu src/read_csv.cu src/evaluate_accuracy.cu src/classifier_math.cu src/normalization.cu src/linear_classification_batch.cu src/preprocess_data.cu
	nvcc -arch=sm_86 -g -G -lcublas -lcusolver src/linear_classification_batch.cu src/normalization.cu src/stable_softmax.cu src/read_csv.cu src/preprocess_data.cu src/evaluate_accuracy.cu src/fmatrix.cu src/classifier_math.cu src/cuda_stuff.cu -o batch_linear_classification

shared:
	nvcc -arch=sm_86 -g -G -Xcompiler -fPIC -shared -lcublas -lcusolver src/linear_classification_shared.cu src/normalization.cu src/stable_softmax.cu src/read_csv.cu src/preprocess_data.cu src/evaluate_accuracy.cu src/fmatrix.cu src/classifier_math.cu src/cuda_stuff.cu -o lclass.so

check: lin_cl
	cuda-gdb -batch -x tmp.txt ./linear_classification

batch_check: batch_lin_cl
	cuda-gdb -batch -x tmp.txt ./batch_linear_classification