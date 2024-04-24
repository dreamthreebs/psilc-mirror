objects = CovMat.so

CovMat.so: CovMat.c
	gcc -shared -fpic -std=c99 -fopenmp -o CovMat.so CovMat.c
	#gcc -shared -fpic -o CovMat.so CovMat.c

.PHONY:clean
clean:$(objects)
	rm $(objects)
