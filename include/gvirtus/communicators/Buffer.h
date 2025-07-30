/*
 * gVirtuS -- A GPGPU transparent virtualization component.
 *
 * Copyright (C) 2009-2010  The University of Napoli Parthenope at Naples.
 *
 * This file is part of gVirtuS.
 *
 * gVirtuS is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * gVirtuS is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with gVirtuS; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
 *
 * Written by: Giuseppe Coviello <giuseppe.coviello@uniparthenope.it>,
 *             Department of Applied Science
 */

/**
 * @file   Buffer.h
 * @author Giuseppe Coviello <giuseppe.coviello@uniparthenope.it>
 * @date   Sun Oct 18 13:16:46 2009
 *
 * @brief
 *
 *
 */

#pragma once

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <typeinfo>
#include <type_traits>
#include <cstddef>
#include <cxxabi.h>
#include <execinfo.h>

#include <gvirtus/common/gvirtus-type.h>

#include "Communicator.h"

#define BLOCK_SIZE 4096

static void printStacktrace() {
    void *callstack[128];
    int frames = backtrace(callstack, 128);
    char **symbols = backtrace_symbols(callstack, frames);
    std::cerr << "------ STACK TRACE ------" << std::endl;
    for (int i = 0; i < frames; ++i) {
        std::cerr << symbols[i] << std::endl;
    }
    std::cerr << "-------------------------" << std::endl;
    free(symbols);
}

namespace gvirtus::communicators {
/**
 * Buffer is a general purpose for marshalling and unmarshalling data. It's used
 * for exchanging data beetwen Frontend and Backend. It has the functionality to
 * be created starting from an input stream and to be sent over an output
 * stream.
 */

    template <typename T>
    std::string demangled_type_name() {
        int status = 0;
        std::unique_ptr<char, void(*)(void*)> demangled(
            abi::__cxa_demangle(typeid(T).name(), nullptr, nullptr, &status),
            std::free
        );
        return (status == 0) ? demangled.get() : typeid(T).name();
    }

    template <typename T>
    constexpr std::size_t safe_sizeof() {
        if constexpr (std::is_void_v<T>) {
            return 1;
        } else if constexpr (std::is_function_v<T>) {
            throw std::runtime_error("safe_sizeof<T> cannot be used with function types");
        } else {
            return sizeof(T);
        }
    }

    class Buffer {
    public:
        Buffer(size_t initial_size = 0, size_t block_size = BLOCK_SIZE);
        Buffer(const Buffer &orig);
        Buffer(std::istream &in);
        Buffer(char *buffer, size_t buffer_size, size_t block_size = BLOCK_SIZE);
        virtual ~Buffer();

        template <class T>
        void Add(T item) {
            if ((mLength + safe_sizeof<T>()) >= mSize) {
                mSize = ((mLength + safe_sizeof<T>()) / mBlockSize + 1) * mBlockSize;
                if ((mpBuffer = (char *)realloc(mpBuffer, mSize)) == NULL)
                    throw std::runtime_error("Buffer::Add(item): Can't reallocate memory.");
            }
            memmove(mpBuffer + mLength, (char *)&item, safe_sizeof<T>());
            mLength += safe_sizeof<T>();
            mBackOffset = mLength;
        }

        template <class T>
        void Add(T *item, size_t n = 1) {
            if (item == NULL) {
                Add((size_t)0);
                return;
            }
            size_t size = safe_sizeof<T>() * n;
            Add(size);
            if ((mLength + size) >= mSize) {
                mSize = ((mLength + size) / mBlockSize + 1) * mBlockSize;
                if ((mpBuffer = (char *)realloc(mpBuffer, mSize)) == NULL)
                    throw std::runtime_error("Buffer::Add(item, n): Can't reallocate memory.");
            }
            memmove(mpBuffer + mLength, (char *)item, size);
            mLength += size;
            mBackOffset = mLength;
        }

        template <class T>
        void AddConst(const T item) {
            if ((mLength + safe_sizeof<T>()) >= mSize) {
                mSize = ((mLength + safe_sizeof<T>()) / mBlockSize + 1) * mBlockSize;
                if ((mpBuffer = (char *)realloc(mpBuffer, mSize)) == NULL)
                    throw std::runtime_error("Buffer::AddConst(item): Can't reallocate memory.");
            }
            memmove(mpBuffer + mLength, (char *)&item, safe_sizeof<T>());
            mLength += safe_sizeof<T>();
            mBackOffset = mLength;
        }

        template <class T>
        void AddConst(const T *item, size_t n = 1) {
            if (item == NULL) {
                Add((size_t)0);
                return;
            }
            size_t size = safe_sizeof<T>() * n;
            Add(size);
            if ((mLength + size) >= mSize) {
                mSize = ((mLength + size) / mBlockSize + 1) * mBlockSize;
                if ((mpBuffer = (char *)realloc(mpBuffer, mSize)) == NULL)
                    throw std::runtime_error("Buffer::AddConst(item, n): Can't reallocate memory.");
            }
            memmove(mpBuffer + mLength, (char *)item, size);
            mLength += size;
            mBackOffset = mLength;
        }

        void AddString(const char *s) {
            size_t size = strlen(s) + 1;
            Add(size);
            Add(s, size);
        }

        template <class T>
        void AddMarshal(T item) {
            Add((gvirtus::common::pointer_t)item);
        }

        template <class T>
        void Read(Communicator *c) {
            auto required_size = mLength + safe_sizeof<T>();
            if (required_size >= mSize) {
                mSize = (required_size / mBlockSize + 1) * mBlockSize;
                if ((mpBuffer = (char *)realloc(mpBuffer, mSize)) == NULL)
                    throw std::runtime_error("Buffer::Read(*c) Can't reallocate memory.");
            }
            c->Read(mpBuffer + mLength, safe_sizeof<T>());
            mLength += safe_sizeof<T>();
            mBackOffset = mLength;
        }

        template <class T>
        void Read(Communicator *c, size_t n = 1) {
            auto required_size = mLength + safe_sizeof<T>() * n;
            if (required_size >= mSize) {
                mSize = (required_size / mBlockSize + 1) * mBlockSize;
                if ((mpBuffer = (char *)realloc(mpBuffer, mSize)) == NULL)
                    throw std::runtime_error("Buffer::Read(*c, n): Can't reallocate memory.");
            }
            c->Read(mpBuffer + mLength, safe_sizeof<T>() * n);
            mLength += safe_sizeof<T>() * n;
            mBackOffset = mLength;
        }

        template <class T>
        T Get() {
            if (mOffset + safe_sizeof<T>() > mLength) {
                printStacktrace();
                throw std::runtime_error(std::string("Buffer::Get(): Can't read any ") + demangled_type_name<T>());
            }
            T result = *((T *)(mpBuffer + mOffset));
            mOffset += safe_sizeof<T>();
            return result;
        }

        template <class T>
        T BackGet() {
            if (mBackOffset - safe_sizeof<T>() > mLength)
                throw std::runtime_error(std::string("Buffer::BackGet(): Can't read ") + demangled_type_name<T>());
            T result = *((T *)(mpBuffer + mBackOffset - safe_sizeof<T>()));
            mBackOffset -= safe_sizeof<T>();
            return result;
        }

        template <class T>
        T *Get(size_t n) {
            if (Get<size_t>() == 0) return NULL;
            if (mOffset + safe_sizeof<T>() * n > mLength)
                throw std::runtime_error(std::string("Buffer::Get(n): Can't read  ") + demangled_type_name<T>());
            T *result = new T[n];
            memmove((char *)result, mpBuffer + mOffset, safe_sizeof<T>() * n);
            mOffset += safe_sizeof<T>() * n;
            return result;
        }

        template <class T>
        T *Delegate(size_t n = 1) {
            size_t size = safe_sizeof<T>() * n;
            Add(size);
            if ((mLength + size) >= mSize) {
                mSize = ((mLength + size) / mBlockSize + 1) * mBlockSize;
                if ((mpBuffer = (char *)realloc(mpBuffer, mSize)) == NULL)
                    throw std::runtime_error("Buffer::Delegate(n): Can't reallocate memory.");
            }
            T *dst = (T *)(mpBuffer + mLength);
            mLength += size;
            mBackOffset = mLength;
            return dst;
        }

        template <class T>
        T *Assign(size_t n = 1) {
            if (Get<size_t>() == 0) return NULL;
            if (mOffset + safe_sizeof<T>() * n > mLength) {
                throw std::runtime_error(std::string("Buffer::Assign(n): Can't read  ") + demangled_type_name<T>());
            }
            T *result = (T *)(mpBuffer + mOffset);
            mOffset += safe_sizeof<T>() * n;
            return result;
        }

        template <class T>
        T *AssignAll() {
            size_t size = Get<size_t>();
            if (size == 0) return NULL;
            size_t n = size / safe_sizeof<T>();
            if (mOffset + safe_sizeof<T>() * n > mLength)
                throw std::runtime_error(std::string("Buffer::AssignAll(): Can't read ") + demangled_type_name<T>());
            T *result = (T *)(mpBuffer + mOffset);
            mOffset += safe_sizeof<T>() * n;
            return result;
        }

        char *AssignString() {
            size_t size = Get<size_t>();
            return Assign<char>(size);
        }

        template <class T>
        T *BackAssign(size_t n = 1) {
            if (mBackOffset - safe_sizeof<T>() * n > mLength)
                throw std::runtime_error(std::string("Buffer::BackAssign(n): Can't read ") + demangled_type_name<T>());
            T *result = (T *)(mpBuffer + mBackOffset - safe_sizeof<T>() * n);
            mBackOffset -= safe_sizeof<T>() * n + sizeof(size_t);
            return result;
        }

        template <class T>
        T GetFromMarshal() {
            return (T)Get<gvirtus::common::pointer_t>();
        }

        inline bool Empty() { return mOffset == mLength; }

        void Reset();
        void Reset(Communicator *c);
        const char *const GetBuffer() const;
        size_t GetBufferSize() const;
        void Dump(Communicator *c) const;

    private:
        size_t mBlockSize;
        size_t mSize;
        size_t mLength;
        size_t mOffset;
        size_t mBackOffset;
        char *mpBuffer;
        bool mOwnBuffer;
    };
}  // namespace gvirtus::communicators