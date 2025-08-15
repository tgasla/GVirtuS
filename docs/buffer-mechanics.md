# Buffer Handling Patterns and Utilities in GVirtuS

## Overview

This document describes the three available methods for interfacing host pointer-based CUDNN routines between the frontend and backend, focusing on how arguments are serialized/deserialized using a `Buffer`. It also explains the helper methods: `Get`, `Assign`, and `Delegate`. The examples presented below are taken from cuDNN, but the same serialization logic and conventions apply throughout the entire GVirtuS codebase.

---

## Methods of Handling Host Pointers

### 🧩 Method B1 (Compatible with F1)

F1/B1 (stack‐allocated outputs, frontend reads “values”)
Pros: simple, no heap allocations for the outputs
Cons: uses VLA‐style stack arrays (int dimA[nbDimsRequested];) which is not standard C++ (only a compiler extension), and can overflow the stack if nbDimsRequested is large.

**Frontend**

```cpp
// F1: frontend does NOT pass any output pointers;
extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetTensorNdDescriptor(
    const cudnnTensorDescriptor_t tensorDesc,
    int nbDimsRequested,
    cudnnDataType_t *dataType,
    int *nbDims,
    int dimA[],
    int strideA[]) {
    
    CudnnFrontend::Prepare();
    CudnnFrontend::AddDevicePointerForArguments(tensorDesc);
    CudnnFrontend::AddVariableForArguments<int>(nbDimsRequested);
    CudnnFrontend::Execute("cudnnGetTensorNdDescriptor");

    if (CudnnFrontend::Success()) {
        *dataType = CudnnFrontend::GetOutputVariable<cudnnDataType_t>();
        *nbDims   = CudnnFrontend::GetOutputVariable<int>();
        int *dimA_backend    = CudnnFrontend::GetOutputHostPointer<int>(nbDimsRequested);
        std::memcpy(dimA, dimA_backend, nbDimsRequested * sizeof(int));
        int *strideA_backend = CudnnFrontend::GetOutputHostPointer<int>(nbDimsRequested);
        std::memcpy(strideA, strideA_backend, nbDimsRequested * sizeof(int));
    }
    return CudnnFrontend::GetExitCode();
}
```

**Backend**

```cpp
CUDNN_ROUTINE_HANDLER(GetTensorNdDescriptor) {
  const cudnnTensorDescriptor_t tensorDesc = in->Get<cudnnTensorDescriptor_t>();
  int nbDimsRequested = in->Get<int>();

  // STACK-ALLOCATED variables:
  cudnnDataType_t dataType;
  int            nbDims;
  int dimA[nbDimsRequested];       // <— VLA (non‐standard in C++)
  int strideA[nbDimsRequested];    // <— VLA again

  cudnnStatus_t cs = cudnnGetTensorNdDescriptor(
      tensorDesc, nbDimsRequested,
      &dataType, &nbDims, dimA, strideA);

  std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
  out->Add<cudnnDataType_t>(dataType);
  out->Add<int>(nbDims);
  out->Add<int>(dimA, nbDimsRequested);
  out->Add<int>(strideA, nbDimsRequested);

  return std::make_shared<Result>(cs, out);
}
```

**Frontend Responsibilities:**
- **DO NOT pass uninitialized pointers.**
- Only simple types (`int`, `cudnnDataType_t`) are passed by value.
- The output buffer contains all the final values (including arrays).

**Backend Responsibilities:**
- Allocates **static arrays** on the stack.
- Reads and writes directly using known sizes (`nbDimsRequested`).
- Returns a fully constructed output buffer to the frontend.

**Pros:**
- Simple.
- Stack-allocated buffers are efficient.

**Cons:**
- Not dynamic; `nbDimsRequested` must be known and fixed on both sides.
- Cannot handle unbounded dimensions or dynamic input shapes efficiently.

---

Why this is problematic:
VLA on the stack (int dimA[nbDimsRequested];) is not valid C++ (it’s a compiler extension).

If nbDimsRequested is large (say 8, 16, 32), you risk a stack overflow.

You have to copy each piece (scalar + array) into the Buffer manually with out->Add(…).

In modern C++, it’s better not to rely on variable‐length arrays on the stack. B1 “works” when nbDimsRequested is small (e.g. ≤4), but it’s brittle if you ever ask for more dimensions.

### 🧩 Method B2 (Compatible with F2)

F2/B2 (backend uses Delegate<T>() to carve out output space in a heap‐backed buffer; frontend reads pointers from that buffer)
Pros: safe C++ (no VLAs), dynamic heap storage for exactly the right size, no accidental stack overflow, clear ownership (the backend “places” all outputs into its shared Buffer).
Cons: frontend must remember to call GetOutputHostPointer<T>() (and dereference) for scalars as well as arrays.

**Frontend**

```cpp
extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetTensorNdDescriptor(
    const cudnnTensorDescriptor_t tensorDesc,
    int nbDimsRequested,
    cudnnDataType_t *dataType,
    int *nbDims,
    int dimA[],
    int strideA[]) 
{
  CudnnFrontend::Prepare();
  CudnnFrontend::AddDevicePointerForArguments(tensorDesc);
  CudnnFrontend::AddVariableForArguments<int>(nbDimsRequested);
  CudnnFrontend::Execute("cudnnGetTensorNdDescriptor");
  if (CudnnFrontend::Success()) {
    // Because B2 used Delegate<…>(), the output buffer now contains:
    //   [cudnnDataType_t][int][ int[nbDimsRequested] ][ int[nbDimsRequested] ]
    // So to read them:
    *dataType = *CudnnFrontend::GetOutputHostPointer<cudnnDataType_t>();
    *nbDims   = *CudnnFrontend::GetOutputHostPointer<int>();
    int *dimA_backend    = CudnnFrontend::GetOutputHostPointer<int>(nbDimsRequested);
    std::memcpy(dimA, dimA_backend, nbDimsRequested * sizeof(int));
    int *strideA_backend = CudnnFrontend::GetOutputHostPointer<int>(nbDimsRequested);
    std::memcpy(strideA, strideA_backend, nbDimsRequested * sizeof(int));
  }
  return CudnnFrontend::GetExitCode();
}
```

**Backend**

```cpp
CUDNN_ROUTINE_HANDLER(GetTensorNdDescriptor) {
  const cudnnTensorDescriptor_t tensorDesc = in->Get<cudnnTensorDescriptor_t>();
  int nbDimsRequested = in->Get<int>();

  auto out = std::make_shared<Buffer>();
  // Delegate<…>() allocates exactly the right amount of *heap* space in the Buffer:
  cudnnDataType_t *dataType = out->Delegate<cudnnDataType_t>();
  int             *nbDims   = out->Delegate<int>();
  int             *dimA     = out->Delegate<int>(nbDimsRequested);
  int             *strideA  = out->Delegate<int>(nbDimsRequested);

  // Call the real cuDNN:
  cudnnStatus_t cs = cudnnGetTensorNdDescriptor(
      tensorDesc, nbDimsRequested,
      dataType, nbDims, dimA, strideA);

  return std::make_shared<Result>(cs, out);
}
```

**Frontend Responsibilities:**
- Reads returned **pointer values from buffer memory**.
- Must interpret the delegated memory properly on return.

**Backend Responsibilities:**
- Uses `Delegate<T>(n)` to reserve memory *inside the output buffer*.
- Uses the delegated memory as output pointers in the cuDNN call.

**Pros:**
- No stack allocations.
- Dynamic: adapts to variable sizes at runtime.

**Cons:**
- Frontend must dereference pointers manually from output buffer.
- Slightly more complex to implement correctly.

---

### 🧩 Method B3 (Compatible with F3)

F3/B3 (frontend “pretends” to pass uninitialized output pointers so that the backend will “fill in” those pointers from the same Buffer)
Pros: none that outweigh the complexity.
Cons: very confusing: you’re effectively using in->Get<T>() on pointers that were never initialized (just to force the Buffer to reserve space). It makes the data‐flow hard to follow, and it’s easy to misuse or get UB if someone forgets that “these pointers are only dummy placeholders.”

**Frontend**

```cpp
extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetTensorNdDescriptor(
    const cudnnTensorDescriptor_t tensorDesc,
    int nbDimsRequested,
    cudnnDataType_t *dataType,  // UNINITIALIZED pointer!
    int *nbDims,                 // UNINITIALIZED pointer!
    int dimA[],                  // UNINITIALIZED array!
    int strideA[])               // UNINITIALIZED array!
{
  CudnnFrontend::Prepare();
  CudnnFrontend::AddDevicePointerForArguments(tensorDesc);
  CudnnFrontend::AddVariableForArguments<int>(nbDimsRequested);

  // The “trick” here: we actually pass host pointers FOR uninitialized data,
  // just so the backend can see how many bytes to allocate for them.
  CudnnFrontend::AddHostPointerForArguments<cudnnDataType_t>(dataType);
  CudnnFrontend::AddHostPointerForArguments<int>(nbDims);
  CudnnFrontend::AddHostPointerForArguments<int>(dimA, nbDimsRequested);
  CudnnFrontend::AddHostPointerForArguments<int>(strideA, nbDimsRequested);

  CudnnFrontend::Execute("cudnnGetTensorNdDescriptor");
  if (CudnnFrontend::Success()) {
    *dataType = CudnnFrontend::GetOutputVariable<cudnnDataType_t>();
    *nbDims   = CudnnFrontend::GetOutputVariable<int>();
    int *dimA_backend    = CudnnFrontend::GetOutputHostPointer<int>(nbDimsRequested);
    std::memcpy(dimA, dimA_backend, nbDimsRequested * sizeof(int));
    int *strideA_backend = CudnnFrontend::GetOutputHostPointer<int>(nbDimsRequested);
    std::memcpy(strideA, strideA_backend, nbDimsRequested * sizeof(int));
  }
  return CudnnFrontend::GetExitCode();
}
```

**Backend**

```cpp
CUDNN_ROUTINE_HANDLER(GetTensorNdDescriptor) {
  const cudnnTensorDescriptor_t tensorDesc = in->Get<cudnnTensorDescriptor_t>();
  int nbDimsRequested = in->Get<int>();

  // These “in->Get<T>()” calls actually allocate heap space inside the Buffer
  // (using Get<T>(n) to carve out n slots).
  cudnnDataType_t dataType = in->Get<cudnnDataType_t>();
  int            nbDims   = in->Get<int>();         // "dummy" int
  int *dimA    = in->Get<int>(nbDimsRequested);     // “dummy” memory
  int *strideA = in->Get<int>(nbDimsRequested);     // “dummy” memory

  // Now call the real cuDNN, writing into those dummy‐space pointers:
  cudnnStatus_t cs = cudnnGetTensorNdDescriptor(
      tensorDesc, nbDimsRequested,
      &dataType, &nbDims, dimA, strideA);

  // Finally copy everything into the output Buffer:
  auto out = std::make_shared<Buffer>();
  out->Add<cudnnDataType_t>(dataType);
  out->Add<int>(nbDims);
  out->Add<int>(dimA, nbDimsRequested);
  out->Add<int>(strideA, nbDimsRequested);
  return std::make_shared<Result>(cs, out);
}
```

**Backend Responsibilities:**
- Uses `Assign<T>(n)` or `Get<T>(n)` to access memory regions *allocated and passed from the frontend*.
- Backend assumes pointers are *not initialized* but size-allocated.

**Frontend Responsibilities:**
- Passes memory locations (empty but properly sized).
- Backend fills in the memory, then sends scalar and array values back using `Add`.

**Pros:**
- Flexible: frontend handles pointer memory logic.
- Efficient when used with `Assign`.

**Cons:**
- Easy to misuse if pointer memory is not correctly allocated on the frontend.
- Requires discipline in memory management semantics.

---

Overall, the preferred method is F2/B2.

---

## Buffer Helper Methods

### ✅ `Get<T>()`

```cpp
T result = buffer->Get<T>();
```

- Deserializes a value from the buffer.
- Moves the internal `offset` pointer.
- If used with a pointer (`Get<T>(n)`), allocates `new T[n]` and copies data from buffer.

**Use When:**
- You want to **copy** a value or array into new memory.
- You **take ownership** of the returned memory.

**Avoid When:**
- You don't want to manage memory (requires manual delete).
- You're working with memory already in the buffer.

---

### ✅ `Assign<T>(n)`

```cpp
T* ptr = buffer->Assign<T>(n);
```

- Returns a pointer into the internal buffer memory.
- No allocation, just a direct view.
- In-place, non-owning.

**Use When:**
- Reading **non-owning** views of data from the buffer.
- Need direct pointer access (e.g., calling cudnn routines).

**Avoid When:**
- You need to keep the memory beyond buffer lifespan.
- You plan to free or write to memory permanently.

---

### ✅ `Delegate<T>(n)`

```cpp
T* ptr = buffer->Delegate<T>(n);
```

- Allocates `n * sizeof(T)` bytes **within the output buffer**.
- Returns a pointer where backend functions can write.
- Safe and dynamic.

**Use When:**
- You need to provide **writeable memory** for output arguments.
- You're preparing output buffer data for frontend to later consume.

**Avoid When:**
- You want to keep output on stack (not in buffer).
- You plan to allocate large temporary buffers better placed elsewhere.

---

## Efficiency Tips

- 🔄 **Use `Assign<T>(n)` instead of `Get<T>(n)`** when you're just reading from the buffer. It's faster and avoids heap allocation.
- 🧠 **Use `Delegate<T>(n)`** when backend functions require memory to write into and you want to **retain results in the output buffer.**
- ⚠️ **Avoid mixing `Delegate` and `Assign` on the same field** in the same handler unless you're fully in control of pointer ownership.
- 🧹 **Always pair `Get<T>(n)` with smart pointers** or `delete[]` to avoid memory leaks if you allocate.

---

## Summary

- ✅ For in-place access: `Assign`
- ✅ For dynamic output writing: `Delegate`
- ✅ For full ownership/copy: `Get`
