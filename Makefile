HOME = .

include $(HOME)/make.inc

CFLAGS += -DDIM=$(DIM) 

SRC = $(HOME)/src
UTILS = $(HOME)/utils
WRITER = $(UTILS)/VisitWriter

# FFTW setup 
FFTW_INC = -I$(FFTW_HOME)/include 
FFTW_LIB = -L$(FFTW_HOME)/lib -lfftw3 
ifdef FFTW_HOME
CFLAGS += -DFFTW $(FFTW_INC)
LIBS += $(FFTW_LIB) 
endif 

# silo setup 
SILO_INC = -I$(SILO_HOME)/include 
SILO_LIB = -L$(SILO_HOME)/lib -lsilo -lm 
ifdef SILO_HOME 
CFLAGS += -DSILO $(SILO_INC)
LIBS += $(SILO_LIB)
endif 

# upcxx stuff 
UPC = $(shell upcxx-meta PPFLAGS) $(shell upcxx-meta LDFLAGS) \
	$(shell upcxx-meta LIBFLAGS) 

# look for source files in
VPATH = $(SRC) $(WRITER) 
# look for includes in 
CFLAGS += -I$(SRC) -I$(WRITER) 

OBJDIR = $(HOME)/obj
DEPDIR = $(HOME)/dep

SRCFILES = $(notdir $(wildcard $(SRC)/*.cpp $(WRITER)/*.cpp)) 
OBJS = $(patsubst %.cpp, $(OBJDIR)/%.o, $(SRCFILES))
DEPS = $(patsubst $(OBJDIR)/%.o, $(DEPDIR)/%.d, $(OBJS))

# don't delete intermediate files (.o's) 
.SECONDARY :

$(OBJDIR)/%.o : %.cpp $(HOME)/Makefile $(HOME)/make.inc
	mkdir -p $(OBJDIR); $(CXX) -c $(CFLAGS) $(LIBS) $(UPC) $< -o $@
	mkdir -p $(DEPDIR) 
	$(CXX) -MM $(CFLAGS) $(LIBS) $(UPC) $< | sed -e '1s@^@$(OBJDIR)\/@' > $*.d
	mv $*.d $(DEPDIR)

cleantree :
	rm -rf $(DEPDIR) $(OBJDIR) $(HOME)/test/*.vtk $(HOME)/test/*.visit \
		$(HOME)/test/*.exe

-include $(DEPS)

objs :
	@echo $(OBJS)
flags : 
	@echo $(CFLAGS)
upc :
	@echo $(UPC)