vlfeat:
	python setup.py build_ext -i

clean:
	rm -rf build vlfeat.so vlfeat.cpp

