#LD_PRELOAD="${GVIRTUS_HOME}/lib/frontend/libcudart.so:${GVIRTUS_HOME}/lib/frontend/libcuda.so:${GVIRTUS_HOME}/lib/frontend/libcublas.so:${GVIRTUS_HOME}/lib/frontend/libcudnn.so:${GVIRTUS_HOME}/lib/frontend/libcufft.so:${GVIRTUS_HOME}/lib/frontend/libcurand.so:${GVIRTUS_HOME}/lib/frontend/libcusparse.so:${GVIRTUS_HOME}/lib/frontend/libcusolver.so:${GVIRTUS_HOME}/lib/frontend/libnvrtc.so" \
#LD_LIBRARY_PATH=${GVIRTUS_HOME}/lib/frontend:${GVIRTUS_HOME}/lib \
#PYTORCH_NVML_BASED_CUDA_CHECK=1 \
#TORCHINDUCTOR_FORCE_DISABLE_CACHES=1 \
#python3 distilbert.py

LD_PRELOAD="${GVIRTUS_HOME}/lib/frontend/libcudart.so:${GVIRTUS_HOME}/lib/frontend/libcuda.so:${GVIRTUS_HOME}/lib/frontend/libcublas.so:${GVIRTUS_HOME}/lib/frontend/libcublasLt.so:${GVIRTUS_HOME}/lib/frontend/libcudnn.so:${GVIRTUS_HOME}/lib/frontend/libcufft.so:${GVIRTUS_HOME}/lib/frontend/libcurand.so:${GVIRTUS_HOME}/lib/frontend/libcusparse.so:${GVIRTUS_HOME}/lib/frontend/libcusolver.so:${GVIRTUS_HOME}/lib/frontend/libnvrtc.so" \
LD_LIBRARY_PATH=${GVIRTUS_HOME}/lib/frontend:${GVIRTUS_HOME}/lib \
PYTORCH_NVML_BASED_CUDA_CHECK=1 \
TORCHINDUCTOR_FORCE_DISABLE_CACHES=1 \
CUDA_LAUNCH_BLOCKING=1 \
TORCH_USE_CUDA_DSA=1 \
TORCH_SHOW_CPP_STACKTRACES=1  \
TORCH_SHOW_MEMORY_USAGE=1 \
PYTORCH_CUDA_FUSER_DISABLE=1 \
TORCH_CUDA_DEBUG=1 \
TOKENIZERS_PARALLELISM=false \
TORCH_DISABLE_ADDR2LINE=1 \
python3 distilbert.py

# if you do gdb --args python3 test.py
# then do a `catch throw``, then `run` and then `bt` (when the crash happens)

# NVIDIA_CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=100 \
# CUPTI_LOG_LEVEL=INFO \
# CUPTI_LOG_FILE=cuda_trace.log \
# PYTORCH_CUDA_ALLOC_CONF=debug_info \
# LD_DEBUG=libs \
# python3 distilbert.py

# to debug CUDA issues, you can use strace
#strace -f -e trace=openat,mmap,mprotect,read python3 distilbert.py -o strace.log

# or ltrace
#ltrace -f -e '__cudaRegisterFatBinary+__cudaRegisterVar+__cudaRegisterFunction+cudaMalloc+cudaMemcpy+cudaLaunchKernel' -o ltrace.log python3 distilbert.py
#ltrace -f python3 distilbert.py

# or you can use LD_AUDIT to trace the CUDA calls
# first create the audit.c file with the following content:
# #define _GNU_SOURCE
# #include <stdio.h>
# #include <link.h>

# unsigned int la_version(unsigned int version) {
#     return version;
# }

# uintptr_t la_symbind64(Elf64_Sym *sym, unsigned int ndx,
#                        uintptr_t *refcook, uintptr_t *defcook,
#                        unsigned int *flags, const char *symname) {
#     printf("[AUDIT] Binding symbol: %s\n", symname);
#     return sym->st_value;  // Correct: return as uintptr_t
# }

# unsigned int la_objopen(struct link_map *map, Lmid_t lmid, uintptr_t *cookie) {
#     printf("[AUDIT] Loaded object: %s\n", map->l_name);
#     return LA_FLG_BINDTO | LA_FLG_BINDFROM;
# }

# then produce the shared object file:
#gcc -fPIC -shared -o audit.so audit.c -ldl

# and run the script with:
#LD_AUDIT=./audit.so python3 distilbert.py > audit.log

# or you can use LD_DEBUG to trace the library loading
#LD_DEBUG=libs python3 distilbert.py

# or you can use objdump to find if a symbol is present in the library
#objdump -T /usr/local/gvirtus/lib/frontend/libcudart.so | grep __cudaRegisterFatBinary

# or nm similarly
#nm -D /usr/local/gvirtus/lib/frontend/libcudart.so | grep __cudaRegisterFatBinary
