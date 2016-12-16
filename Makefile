all:
	@echo 'Run "make regression", for compiling the ridge regression document classifier.'
	@echo 'Run "make centroid", for compiling the Rocchio/centroid-based document classifier.'

regression:
	@echo 'Compiling ridge regression document classifier:'
	g++ -Wall -O3 -std=c++11 -DRIDGE -I lib/Eigen/ main.cpp -o regression

centroid:
	@echo 'Compiling centroid-based document classifier:'
	g++ -Wall -O3 -std=c++11 -DROCCHIO -I lib/Eigen/ main.cpp -o centroid
