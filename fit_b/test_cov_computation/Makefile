objects = CovMat.so

CovMat.so: CovMat.c
	gcc -g -shared -fpic -std=c99 -fopenmp -o CovMat.so CovMat.c
	#gcc -shared -fpic -o CovMat.so CovMat.c
	# gcc -O3 -fopenmp -o CovMat CovMat.c -lm


.PHONY:clean
clean:$(objects)
	rm $(objects)

