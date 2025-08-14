# Written By: Giuseppe Coviello <giuseppe.coviello@uniparthenope.it>,
#             Department of Applied Science
#
# Edited By: Theodoros Aslanidis <theodoros.aslanidis@ucdconnect.ie>,
#             School of Computer Science, University College Dublin

project(gvirtus-plugin-cudart)
find_package(CUDA REQUIRED)

include_directories(${CUDART_INCLUDE_DIRECTORY} ${CUDA_INCLUDE_DIRS} util cuda_internals)

resolve_cuda_library_version(cudart CUDA_VERSION)

gvirtus_add_backend(cudart ${CUDA_VERSION}
    backend/CudaRtHandler_api.cpp
    backend/CudaRtHandler_device.cpp
    backend/CudaRtHandler_driver_entry_point.cpp
    backend/CudaRtHandler_error.cpp
    backend/CudaRtHandler_event.cpp
    backend/CudaRtHandler_execution.cpp
    backend/CudaRtHandler_graph.cpp
    backend/CudaRtHandler_internal.cpp
    backend/CudaRtHandler_memory.cpp
    backend/CudaRtHandler_occupancy.cpp
    backend/CudaRtHandler_opengl.cpp
    backend/CudaRtHandler_profiler.cpp
    backend/CudaRtHandler_stream_memory.cpp
    backend/CudaRtHandler_stream.cpp
    backend/CudaRtHandler_surface.cpp
    backend/CudaRtHandler_texture.cpp
    backend/CudaRtHandler_thread.cpp
    backend/CudaRtHandler_version.cpp
    backend/CudaRtHandler.cpp
    util/CudaUtil.cpp
)

target_link_libraries(${PROJECT_NAME} ${CUDA_CUDART_LIBRARY} lz4)

gvirtus_add_frontend(cudart ${CUDA_VERSION}
        frontend/CudaRt_api.cpp
        frontend/CudaRt_device.cpp
        frontend/CudaRt_driver_entry_point.cpp
        frontend/CudaRt_event.cpp
        frontend/CudaRt_error.cpp
        frontend/CudaRt_execution.cpp
        frontend/CudaRt_graph.cpp
        frontend/CudaRtFrontend.cpp
        frontend/CudaRt_internal.cpp
        frontend/CudaRt_memory.cpp
        frontend/CudaRt_occupancy.cpp
        frontend/CudaRt_opengl.cpp
        frontend/CudaRt_profiler.cpp
        frontend/CudaRt_stream_memory.cpp
        frontend/CudaRt_stream.cpp
        frontend/CudaRt_surface.cpp
        frontend/CudaRt_texture.cpp
        frontend/CudaRt_thread.cpp
        frontend/CudaRt_version.cpp
        util/CudaUtil.cpp
)

add_subdirectory(demo)