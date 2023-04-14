exe = bfs

cc = "/usr/local/cuda-11.6/bin/nvcc" 
flags = -I. -O3 --compiler-options -Wall -Xptxas -v

ifeq ($(debug), 1)
	flags+= -DDEBUG 
endif

objs = $(patsubst %.cu,%.o,$(wildcard *.cu))

deps = $(wildcard ./*.cuh) \
		$(wildcard *.h) \
		Makefile

%.o:%.cu $(deps)
	$(cc) -c $< -o $@ $(flags)

$(exe):$(objs)
	$(cc) $(objs) -o $(exe) $(flags)

clean:
	rm -rf $(exe) $(objs) 