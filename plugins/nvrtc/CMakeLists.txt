# Written By: Theodoros Aslanidis <theodoros.aslanidis@ucdconnect.ie>,
#             School of Computer Science, University College Dublin

project(gvirtus-plugin-nvrtc)
find_package(CUDAToolkit REQUIRED)

include_directories(${CUDAToolkit_INCLUDE_DIRS})

resolve_cuda_library_version(nvrtc NVRTC_VERSION)

gvirtus_add_backend(nvrtc ${NVRTC_VERSION}
	backend/NvrtcHandler_compilation.cpp
	backend/NvrtcHandler_error.cpp
	backend/NvrtcHandler_giq.cpp
	backend/NvrtcHandler_host.cpp
	backend/NvrtcHandler_pch.cpp
	backend/NvrtcHandler.cpp
)

target_link_libraries(${PROJECT_NAME}
	CUDA::nvrtc
)

gvirtus_add_frontend(nvrtc ${NVRTC_VERSION}
	frontend/Nvrtc_compilation.cpp
	frontend/Nvrtc_error.cpp
	frontend/Nvrtc_giq.cpp
	frontend/Nvrtc_host.cpp
	frontend/Nvrtc_pch.cpp
	frontend/NvrtcFrontend.cpp
)

target_compile_definitions(nvrtc
	PUBLIC NVRTC_VERSION=${NVRTC_VERSION}
)