CFLAGS = `pkg-config --cflags opencv`
LIBS = `pkg-config --libs opencv`

# For compiling
export PKG_CONFIG_PATH=~/trash/opencvout/lib/pkgconfig/
# For running
export LD_LIBRARY_PATH=/home/chase/trash/opencvout/lib/

CFLAGS=$(shell PKG_CONFIG_PATH=$(PKG_CONFIG_PATH) pkg-config opencv --cflags)
LIBS=$(shell PKG_CONFIG_PATH=$(PKG_CONFIG_PATH) pkg-config opencv --libs)

all: main run

main : main.cpp header.h
	g++ -g -Wall -o main main.cpp $(CFLAGS) $(LIBS) -std=c++0x

run: 
	LD_LIBRARY_PATH=/home/chase/trash/opencvout/lib/ && ./main longBrush2.png --novid

runvid: main
	LD_LIBRARY_PATH=/home/chase/trash/opencvout/lib/ && ./main longBrush2.png

.PHONY: clean
clean:
	rm -f main

.PHONY: run clean runvid
