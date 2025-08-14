# Written By: Raffaele Montella <raffaele.montella@uniparthenope.it>,
#              Department of Science and Technologies
#
# Edited By: Theodoros Aslanidis <theodoros.aslanidis@uniparthenope.it>,
#              School of Computer Science, University College Dublin

project(gvirtus-plugin-cusparse)
find_package(CUDAToolkit REQUIRED)

include_directories(${CUDAToolkit_INCLUDE_DIRS})

resolve_cuda_library_version(cusparse CUSPARSE_VERSION)

gvirtus_add_backend(cusparse ${CUSPARSE_VERSION}
	backend/CusparseHandler.cpp)

target_link_libraries(${PROJECT_NAME}
	CUDA::cusparse
)

gvirtus_add_frontend(cusparse ${CUSPARSE_VERSION}
	frontend/Cusparse.cpp
	frontend/Cusparse_helper.cpp
	frontend/CusparseFrontend.cpp
)
