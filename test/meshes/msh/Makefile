%-2.2.msh: %.geo
	gmsh -2 -format msh2 -o $@ $<
%-4.1.msh: %.geo
	gmsh -2 -o $@ $<
all: insulated-2.2.msh insulated-4.1.msh
