FLAGS	:= -std=c++20 \
				 -Wall -Wextra -pedantic -D_GLIBCXX_DEBUG -O3 -g \
				 -Isrc \
				 -march=native -msse4.2 -msse4.1 -msse3 -mavx -mavx2 -mfma -mfpmath=sse \
				 -funroll-loops

LIBS	:= -lopenblas
FILES	:= $(wildcard src/*.cpp)
OBJS	:= $(patsubst src/%.cpp, obj/%.o, $(FILES))
HDRS	:= $(wildcard src/*.hpp)

main: $(OBJS)
	g++ $(FLAGS) $(OBJS) -o $@ $(LIBS)

obj/%.o: src/%.cpp $(HDRS)
	@mkdir -p obj
	g++ $(FLAGS) -c $< -o $@

