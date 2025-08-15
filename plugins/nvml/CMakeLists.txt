project(gvirtus-plugin-nvml)
find_package(CUDAToolkit REQUIRED)

include_directories(${CUDAToolkit_INCLUDE_DIRS})

get_target_property(lib_path CUDA::nvml IMPORTED_LOCATION)
if(lib_path)
    message(STATUS "CUDA library nvml found at ${lib_path}")
else()
    message(WARNING "nvml library not found, skipping plugins/nvml")
    return()
endif()

execute_process(
    COMMAND readelf -d ${lib_path}
    OUTPUT_VARIABLE readelf_output
    ERROR_QUIET
)

 # Match the SONAME line and capture the SONAME string inside []
set(NVML_VERSION "UNKNOWN")
string(REGEX MATCH "nvidia-ml\\.so[^]]*" soname_match "${readelf_output}")
if(soname_match)
    # message(STATUS "Detected SONAME: ${soname_match}")
     # Extract the version number after ".so."
    string(REGEX REPLACE ".*\\.so\\.([0-9]+(\\.[0-9]+)*)" "\\1" NVML_VERSION "${soname_match}")
else()
    message(WARNING "nvml library version not found")
endif()

message(STATUS "Resolved version for nvml: ${NVML_VERSION}")

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