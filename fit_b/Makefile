objects = CovMat.so

CovMat.so: CovMat_long.c
	gcc -shared -fpic -std=c99 -fopenmp -o CovMat.so CovMat_long.c
	#gcc -shared -fpic -o CovMat.so CovMat.c

.PHONY:clean
clean:$(objects)
	rm $(objects)

