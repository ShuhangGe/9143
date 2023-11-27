### ECE-GY 9143 - High Performance Computing for Machine Learning
### Instructor: Parijat Dubey
### Makefile for Part A Assignment 5


SDK_INSTALL_PATH :=  /usr/local/cuda-11.3
NVCC=$(SDK_INSTALL_PATH)/bin/nvcc
CUDNN_PATH := cudnn
LIB       :=  -L$(SDK_INSTALL_PATH)/lib64 -L$(SDK_INSTALL_PATH)/samples/common/lib/linux/x86_64  -L $(CUDNN_PATH)/lib64
#INCLUDES  :=  -I$(SDK_INSTALL_PATH)/include -I$(SDK_INSTALL_PATH)/samples/common/inc
OPTIONS   :=  -O3 
#--maxrregcount=100 --ptxas-options -v 

# TAR_FILE_NAME  := YourNameCUDA1.tar
EXECS :=  C1 C2 C3
all:$(EXECS)

#######################################################################
clean:
	rm -f $(EXECS) *.o
#######################################################################
timer.o : timer.cu timer.h
	${NVCC} $< -c -o $@ $(OPTIONS)
	
C1 : C1.cu timer.o
	${NVCC} $< -o $@ $(LIB) timer.o $(OPTIONS)

C2 : C2.cu timer.o
	${NVCC} $< -o $@ $(LIB) timer.o $(OPTIONS)

C3 : C3.cu timer.o
	${NVCC} $< -o $@ $(LIB) timer.o $(OPTIONS)



