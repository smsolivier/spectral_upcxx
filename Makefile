HOME = .

include $(HOME)/make.inc

SRC = $(HOME)/src
UTILS = $(HOME)/utils
WRITER = $(UTILS)/VisitWriter
TIMER = $(UTILS)/timer
CHTIMER = $(UTILS)/chtimer

# import settings from make.inc 
CFLAGS += -DDIM=$(DIM) $(OPT)

# use openmp threading 
ifdef OMP
CFLAGS += -fopenmp -DOMP
endif

# print progress statements 
ifdef VERBOSE
CFLAGS += -DVERBOSE
endif

# add debugging checks 
ifdef DEBUG
CFLAGS += -DDEBUG
endif

# use FFTW_MEASURE to optimize 1D FFT calls 
ifdef MEASURE
CFLAGS += -DMEASURE 
endif

# write VTK files in text 
ifdef VISIT_ASCII
CFLAGS += -DVISIT_ASCII=0 
else 
CFLAGS += -DVISIT_ASCII=1
endif 

# zero highest mode out 
ifdef ZERO 
CFLAGS += -DZERO
endif

# use pencil global transpose 
ifdef PENCILS
CFLAGS += -DPENCILS
endif 

# use slabs in global transpose 
ifdef SLABS 
CFLAGS += -DSLABS
endif 

# transpose to contiguous in z before doing z transforms 
ifdef TRANSPOSE 
CFLAGS += -DTRANSPOSE 
endif

# enable CH_Timer 
ifndef TIMER 
CFLAGS += -DCH_NTIMER
endif

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
VPATH = $(SRC) $(WRITER) $(TIMER) $(CHTIMER) 
# look for includes in 
CFLAGS += -I$(SRC) -I$(WRITER) -I$(TIMER) -I$(CHTIMER)

OBJDIR = $(HOME)/obj
DEPDIR = $(HOME)/dep

SRCFILES = $(notdir $(wildcard $(SRC)/*.cpp $(WRITER)/*.cpp $(TIMER)/*.cpp $(CHTIMER)/*.cpp)) 
OBJS = $(patsubst %.cpp, $(OBJDIR)/%.o, $(SRCFILES))
DEPS = $(patsubst $(OBJDIR)/%.o, $(DEPDIR)/%.d, $(OBJS))

# don't delete intermediate files (.o's) 
.SECONDARY :

$(OBJDIR)/%.o : %.cpp $(HOME)/Makefile $(HOME)/make.inc
	mkdir -p $(OBJDIR); $(CXX) -c $(CFLAGS) $(LIBS) $(UPC) $< -o $@
	mkdir -p $(DEPDIR) 
	$(CXX) -MM $(CFLAGS) $(LIBS) $(UPC) $< | sed -e '1s@^@$(OBJDIR)\/@' > $*.d
	mv $*.d $(DEPDIR)

clean :
	rm -f *.exe *.vtk *.visit *.time
cleantree :
	rm -rf $(DEPDIR) $(OBJDIR) $(HOME)/test/*.vtk $(HOME)/test/*.visit \
		$(HOME)/test/*.exe $(HOME)/exec/*.vtk $(HOME)/exec/*.exe $(HOME)/exec/*.visit

-include $(DEPS)

.PHONY : docs
docs :
	cd $(HOME)/docs; doxygen Doxyfile 
objs :
	@echo $(OBJS)
flags : 
	@echo $(CFLAGS)
upc :
	@echo $(UPC)
