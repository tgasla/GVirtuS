# 🧠 Pointer Passing Patterns and Best Practices

## ✅ Overview

This guide documents how to pass various types of values and pointers between the frontend and backend using GVirtuS-style marshalling. It focuses on `Add`, `Assign`, `Marshal`, and `Pointer` semantics.

---

## 🟩 Native Simple Types

Use `AddVariableForArguments` and `GetOutputVariable<T>()` for:

- `int`, `float`, `double`, `bool`, `long long`, etc.
- CUDA typedefs that resolve to native types (e.g., enums or integers).

```cpp
// Frontend
AddVariableForArguments<int>(someFlag);

// Backend
int someFlag = in->Get<int>();
```

---

## 🟨 Device Pointers (`void*`, `opaque*`)

Use `AddDevicePointerForArguments()` when:

- Passing a GPU memory address (device pointer).
- The pointer may be to an **opaque struct** (not serializable).
- For host pointers do not pass a memory address as the other side cannot dereference it.

```cpp
// Frontend
AddDevicePointerForArguments(y);

// Backend
void* y = in->Get<void*>(); // or in->GetFromMarshal<void*>()
```

⚠️ Avoid casting to `long long`. Use `uint64_t` instead for safety and clarity.

---

## 🟦 Marshalling Pointers (`Marshal<T>()`)

Use `AddMarshal<T>()` when:

- Passing a GPU pointe that is `void*` with semantic clarity.
- Backend must retrieve it as a raw address.

```cpp
// Frontend
AddMarshal<void*>(y);

// Backend
void* y = in->GetFromMarshal<void*>();

// Return path
out->AddMarshal<void*>(y);
void* y = GetOutputDevicePointer(); // OR GetOutputVariable<void*>()
```

✅ `AddMarshal` converts the pointer to `uint64_t` internally, making it more standardized and safer.

---

## 🟫 Host Pointers (Arrays of Native Types)

Use `AddHostPointerForArguments<T>()` and `Assign<T>()` / `Get<T>()` when:

- Passing a host pointer to an array of `int`, `float`, etc.
- ⚠️ Always specify the number of elements!

```cpp
// Frontend
CudnnFrontend::AddHostPointerForArguments<int>(dimA, nbDimsRequested);

// Backend
int* dimA = in->Assign<int>(nbDimsRequested); // OR in->Get<int>(nbDimsRequested);
```

To return arrays:

```cpp
// Backend
out->Add<int>(dimA, nbDimsRequested);

// Frontend
int *dimA_backend = CudnnFrontend::GetOutputHostPointer<int>(nbDimsRequested);
std::memcpy(dimA, dimA_backend, nbDimsRequested * sizeof(int)); // ✅ Required!
```

❗️`memcpy` is essential when receiving host pointer arrays from backend — never skip it.

---

## 🔁 Interchangeability

You can **mix** `Add<void*>()` and `AddMarshal<void*>()`, `Get<void*>()` and `GetFromMarshal<void*>()`, `GetOutputDevicePointer()` and `GetOutputVariable<void*>()`, and `AddDevicePointerForArguments(var)` and `AddVariableForArguments<uint64_t>((uint_64_t)var)` as long as the underlying format is a memory address (i.e., `uint64_t` on most modern systems).

---

## 📝 Summary of Usage

| Use Case | Function | Notes |
|----------|----------|-------|
| Native types (int, float) | `AddVariableForArguments` / `Get<T>()` | Simple data |
| Host pointer to array | `AddHostPointerForArguments<T>` / `Assign<T>(n)` | Specify `n` |
| GPU memory address | `AddDevicePointerForArguments` | For `void*`, `opaque*`, etc |
| Void pointer w/ semantic clarity | `AddMarshal<void*>` / `GetFromMarshal<void*>()` | Standardized |
| Returning host array | `Add<T>(data, n)` / `GetOutputHostPointer<T>(n)` + `memcpy` | ❗Always use `memcpy` |

---

## ⚠️ Warnings

- Avoid using `long long` for memory addresses — prefer `uint64_t`.
- Never try to pass opaque structs by value — use pointers.
- Always `memcpy` host arrays when receiving them.