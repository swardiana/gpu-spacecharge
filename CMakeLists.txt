# For example the framework can run without GEANT4, but ROOT is
# mandatory


# **************************************************************************
# * Copyright(c) 1998-2014, ALICE Experiment at CERN, All rights reserved. *
# *                                                                        *
# * Author: The ALICE Off-line Project.                                    *
# * Contributors are mentioned in the code where appropriate.              *
# *                                                                        *
# * Permission to use, copy, modify and distribute this software and its   *
# * documentation strictly for non-commercial purposes is hereby granted   *
# * without fee, provided that the above copyright notice appears in all   *
# * copies and that both the copyright notice and this permission notice   *
# * appear in the supporting documentation. The authors make no claims     *
# * about the suitability of this software for any purpose. It is          *
# * provided "as is" without express or implied warranty.                  *
# **************************************************************************

# Module
set(MODULE AliTPCSpaceCharge3DDriftLineGPU)

find_package(CUDA)
if(NOT CUDA_FOUND)
    message( FATAL_ERROR "NVIDIA CUDA package not found. Please install or set SC_CUDA=OFF" )
else()
    find_library(LIBCUDA_SO_PATH libcuda.so)
    string(FIND ${LIBCUDA_SO_PATH} "-NOTFOUND" LIBCUDA_SO_PATH_NOTFOUND )
endif(NOT CUDA_FOUND)
message( STATUS "Building SpaceCharge distortion framework with CUDA support" )

if(LIBCUDA_SO_PATH_NOTFOUND GREATER -1)
  message( FATAL_ERROR "NVIDIA CUDA libcuda.so not found. Please install or set SC_CUDA=OFF" )
endif(LIBCUDA_SO_PATH_NOTFOUND GREATER -1)

#set nvcc flags
set(CUDA_NVCC_FLAGS -Wno-deprecated-gpu-targets --use_fast_math --maxrregcount 64 -O4 -Xcompiler -fPIC -Xptxas -O4 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_61,code=sm_61)

#for convenience
set(SCGPUDIR ${AliRoot_SOURCE_DIR}/TPC/TPCbase/spacecharge-gpu)




# Module include folder
include_directories(${AliRoot_SOURCE_DIR}/TPC/TPCbase/spacecharge-gpu
                    ${AliRoot_SOURCE_DIR}/TPC/TPCbase/spacecharge-gpu/interface
                    ${AliRoot_SOURCE_DIR}/TPC/TPCbase/spacecharge-gpu/nvcckernel
                   )

# Additional include folders in alphabetical order except ROOT
include_directories(SYSTEM ${ROOT_INCLUDE_DIR})
include_directories(${AliRoot_SOURCE_DIR}/TPC/TPCbase
                   )

#compile CUDA object file
cuda_compile(SCGPU_O nvcckernel/PoissonSolver3DGPU.cu)
cuda_compile(SCGPU_1 nvcckernel/DifferentialGPU.cu)
cuda_compile(SCGPU_2 nvcckernel/IntegrateEzGPU.cu)



#set it back
if(STDCXX11FOUND GREATER -1)
  string ( REPLACE "-std=c++98" "-std=c++11" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}" )
endif()
#end of CUDA specific stuff; from here on build usual library


# bit of a hack, but otherwise dictionary compilation fails:
# defines cudaError_t to make it known
# defines __ROOT__; triggers "ifdef" in source code to circumvent broken cint behaviour with "volatile" keyword
# not sure why __ROOT__ is used here, could just as well be something else
set_source_files_properties(${CMAKE_CURRENT_BINARY_DIR}/G__AliTPCSpaceCharge3DDriftLineGPU.cxx PROPERTIES COMPILE_DEFINITIONS "cudaError_t=int;__ROOT__")

# Sources in alphabetical order
set(SRCS
    ${SCGPUDIR}/interface/AliTPCPoissonSolverCuda.cxx
    ${SCGPUDIR}/interface/AliTPCSpaceCharge3DDriftLineCuda.cxx
    ${SCGPU_O}
    ${SCGPU_1}
    ${SCGPU_2}
   )

# Headers from sources
set(CINTHDRS
    ${SCGPUDIR}/interface/AliTPCPoissonSolverCuda.h
    ${SCGPUDIR}/interface/AliTPCSpaceCharge3DDriftLineCuda.h
    )
  
set(HDRS
    ${CINTHDRS}
    ${SCGPUDIR}/nvcckernel/PoissonSolver3DGPU.h
    ${SCGPUDIR}/nvcckernel/DifferentialGPU.h
    ${SCGPUDIR}/nvcckernel/IntegrateEzGPU.h
  )

# Generate the dictionary
# It will create G_ARG1.cxx and G_ARG1.h / ARG1 = function first argument
get_directory_property(incdirs INCLUDE_DIRECTORIES)

# generate_dictionary("${MODULE}" "${MODULE}LinkDef.h" "${HDRS}" "${incdirs}")

# set(ROOT_DEPENDENCIES Core EG GenVector Geom Gpad Graf Gui Hist MathCore Matrix Minuit Physics Proof RIO Spectrum Tree)
# set(ALIROOT_DEPENDENCIES STEERBase STAT STEER CDB ESD RAWDatabase RAWDatarec TPCbase)

#set(LIBDEPS ${ALIROOT_DEPENDENCIES} ${ROOT_DEPENDENCIES})
#generate_rootmap("${MODULE}" "${LIBDEPS}" "${CMAKE_CURRENT_SOURCE_DIR}/${MODULE}LinkDef.h")



generate_dictionary("${MODULE}" "${SCGPUDIR}/interface/${MODULE}LinkDef.h" "${HDRS}" "${incdirs}")

# Generate the ROOT map
# Dependecies
set(ROOT_DEPENDENCIES Core EG GenVector Geom Gpad Graf Gui Hist MathCore Matrix Minuit Physics Proof RIO Spectrum Tree)
set(ALIROOT_DEPENDENCIES STEERBase STAT STEER CDB ESD RAWDatabase RAWDatarec  TPCbase)

set(LIBDEPS ${ALIROOT_DEPENDENCIES} ${ROOT_DEPENDENCIES} ${CUDA_LIBRARY})
generate_rootmap("${MODULE}" "${LIBDEPS}" "${SCGPUDIR}/interface/${MODULE}LinkDef.h")

# Add a library to the project using the specified source files
add_library(${MODULE} SHARED ${SRCS} G__${MODULE}.cxx)
#CUDA run-time and driver
target_link_libraries(${MODULE} ${CUDA_LIBRARIES} ${LIBCUDA_SO_PATH} TPCbase)

# Additional compilation flags
set_target_properties(${MODULE} PROPERTIES COMPILE_FLAGS "")

# Installation
install(TARGETS ${MODULE}
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib)

install(FILES ${HDRS} DESTINATION include)
#add_subdirectory(nvcckernel)
#add_subdirectory(interface)
