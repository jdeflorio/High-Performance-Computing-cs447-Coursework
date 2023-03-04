all: ball_samp-seq ball_samp-par
CPPFLAGS = -O -mavx2 -mfma

ball_samp-seq:
	g++ sphere.cpp -o ball_samp-seq -DSEQ $(CPPFLAGS)

ball_samp-par:
	g++ sphere.cpp -o ball_samp-par -DPAR -fopenmp $(CPPFLAGS)

clean:
	rm -f *.o ball_samp-seq ball_samp-par