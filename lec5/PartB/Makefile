### ECE-GY 9143 - High Performance Computing for Machine Learning
### Instructor: Parijat Dubey
### Makefile for Part A Assignment 5


SDK_INSTALL_PATH :=  /usr/local/cuda
NVCC=$(SDK_INSTALL_PATH)/bin/nvcc
LIB       :=  -L$(SDK_INSTALL_PATH)/lib64 -L$(SDK_INSTALL_PATH)/samples/common/lib/linux/x86_64
#INCLUDES  :=  -I$(SDK_INSTALL_PATH)/include -I$(SDK_INSTALL_PATH)/samples/common/inc
OPTIONS   :=  -O3 
#--maxrregcount=100 --ptxas-options -v 

# TAR_FILE_NAME  := YourNameCUDA1.tar
EXECS :=  Q2 Q3
all:$(EXECS)

#######################################################################
clean:
	rm -f $(EXECS) *.o
#######################################################################
timer.o : timer.cu timer.h
	${NVCC} $< -c -o $@ $(OPTIONS)
	
Q2 : Q2.cu timer.o
	${NVCC} $< -o $@ $(LIB) timer.o $(OPTIONS)

Q3 : Q3.cu timer.o
	${NVCC} $< -o $@ $(LIB) timer.o $(OPTIONS)



