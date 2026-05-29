CXX = g++ 
CXXFLAGS = -g -std=c++11 -Wall -Wno-sign-compare -O3

NVCXX = /usr/local/cuda/bin/nvcc 
NVCXXFLAGS = -g --ptxas-options=-v -std=c++11  -O3


CUDALIB = /usr/local/cuda/lib64
CUDAINC = /usr/local/cuda/include
SRCDIR = src
OBJDIR = obj
CUOBJDIR = cuobj
BINDIR = bin

INCS := $(wildcard $(SRCDIR)/*.h)
SRCS := $(wildcard $(SRCDIR)/*.cc)
OBJS := $(wildcard $(OBJDIR)/*.o)
CUSRCS := $(wildcard $(SRCDIR)/*.cu)
CUOBJS := $(CUSRCS:$(SRCDIR)/%.cu=$(CUOBJDIR)/%.o)

all:  bin/reduce

bin:
	mkdir -p bin


bin/reduce: $(OBJS) $(CUOBJS) 
	mkdir -p bin
	@echo "OBJ: "$(OBJS)
	@echo "CUOBJ: "$(CUOBJS)
	$(CXX) $^ -o $@ $(CXXFLAGS) -L$(CUDALIB) -lcudart -Iinclude -I$(CUDAINC) 
			    @echo "Compiled "$<" successfully!"


1: bin/reduce
	./bin/reduce /data/hw5/input_64.dat /data/hw5/output_64.dat

2: bin/reduce
	./bin/reduce /data/hw5/input_4194304.dat /data/hw5/output_4194304.dat

3: bin/reduce
	./bin/reduce /data/hw5/input_8388608.dat /data/hw5/output_8388608.dat

4: bin/reduce
	./bin/reduce /data/hw5/input_16777216.dat /data/hw5/output_16777216.dat


.PHONY:	test clean

$(CUOBJS): $(CUOBJDIR)/%.o : $(SRCDIR)/%.cu
		mkdir -p cuobj
	    @echo $(NVCXX) $(NVCXXFLAGS) "-Iinclude -c" $< "-o" $@
	    @$(NVCXX) $(NVCXXFLAGS) -Iinclude -c $< -o $@
			    @echo "CUDA Compiled "$<" successfully!"

clean: 
	rm -f $(CUOBJS) $(CUOBJS:%.o=%.d) 
	rm -rf bin cuobj

