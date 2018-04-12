override HOME := ./..
include $(HOME)/Makefile 

%.exe : %.cpp $(OBJS)
	$(CXX) $(CFLAGS) $(OBJS) $(LIBS) $(UPC) $< -o $@ 
