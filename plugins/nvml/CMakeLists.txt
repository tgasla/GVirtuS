project(gvirtus-plugin-nvml)
find_package(CUDAToolkit REQUIRED)

include_directories(${CUDAToolkit_INCLUDE_DIRS})

resolve_cuda_library_version(nvml NVML_VERSION)

gvirtus_add_backend(nvml ${NVML_VERSION}
	backend/NvmlHandler_init.cpp
	backend/NvmlHandler_systemquery.cpp
	backend/NvmlHandler_devicequery.cpp
	backend/NvmlHandler_error.cpp
	backend/NvmlHandler_event.cpp
	backend/NvmlHandler_internal.cpp
	backend/NvmlHandler.cpp
)

target_link_libraries(${PROJECT_NAME}
	CUDA::nvml
)

gvirtus_add_frontend(nvml_frontend ${NVML_VERSION}
    frontend/Nvml_init.cpp
	frontend/Nvml_systemquery.cpp
	frontend/Nvml_devicequery.cpp
    frontend/Nvml_error.cpp
	frontend/Nvml_event.cpp
	frontend/Nvml_internal.cpp
    frontend/NvmlFrontend.cpp
)

# nvml shared library does not follow usual naming convention
set_target_properties(nvml_frontend PROPERTIES OUTPUT_NAME "nvidia-ml")