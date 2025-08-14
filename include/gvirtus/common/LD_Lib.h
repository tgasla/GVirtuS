#pragma once

#include <dlfcn.h>

#include <memory>
#include <string>

// #define ldDEBUG

namespace gvirtus::common {

template <class T, typename... Args>
class LD_Lib {
   private:
    using create_t = std::shared_ptr<T>(Args...);

    /* DLOPEN */
    void dlopen(const std::string &path, int flag = RTLD_LAZY) {
#ifdef ldDEBUG
        std::cout << "LD_Lib::dlopen(path, flag) called" << std::endl;
#endif

        m_module = ::dlopen(path.c_str(), flag);

        if (m_module == nullptr) {
            throw std::runtime_error(std::string("Error loading: ") + dlerror());
        }

#ifdef ldDEBUG
        std::cout << "LD_Lib::dlopen(path, flag) returned" << std::endl;
#endif
    }

    /* SET_CREATE_SYMBOL */
    void set_create_symbol(const std::string &function) {
#ifdef ldDEBUG
        std::cout << "LD_Lib::set_create_symbol(&function) called" << std::endl;
        std::cout << "function: " << function.c_str() << std::endl;
        if (m_module == nullptr) {
            std::cout << "m_module is null!!! " << std::endl;
        }
#endif

        // Since the value of the symbol could actually be NULL (so that a NULL return from dlsym()
        // need not indicate an error) the correct way to test for an error is to
        // call dlerror() to clear any old error conditions, then call dlsym(), and then
        // call dlerror() again, saving its return value, and check whether this saved value is not
        // NULL.
        dlerror();
        sym = (create_t *)dlsym(m_module, function.c_str());
        auto dl_error = dlerror();

        if (dl_error != nullptr) {
#ifdef ldDEBUG
            std::cout << "LD_Lib::set_create_symbol(&function) exception!" << std::endl;
#endif
            std::string error(dl_error);
            ::dlclose(m_module);
            throw std::runtime_error(std::string("Cannot load symbol create: ") + error);
        }

#ifdef ldDEBUG
        std::cout << "LD_Lib::set_create_symbol(&function) returned" << std::endl;
#endif
    }

   public:
    /* LD_LIB CONSTRUCTOR */
    LD_Lib(const std::string path, std::string fcreate_name = "create_t") {
#ifdef ldDEBUG
        std::cout << "LD_Lib::LD_Lib(path, fcreate_name) called (it's the constructor)"
                  << std::endl;
        std::cout << "path: " << path << std::endl;
        std::cout << "fcreate_name: " << fcreate_name << std::endl;
#endif

        _obj_ptr = nullptr;
        dlopen(path);
        set_create_symbol(fcreate_name);

#ifdef ldDEBUG
        std::cout << "LD_Lib::LD_Lib(path, fcreate_name) returned" << std::endl;
#endif
    }

    /* LD_LIB DESTRUCTOR */
    ~LD_Lib() {
        if (m_module != nullptr) {
            sym = nullptr;
            _obj_ptr.reset();
            ::dlclose(m_module);
        }
    }

    /* BUILD_OBJ */
    void build_obj(Args... args) {
#ifdef ldDEBUG
        std::cout << "LD_Lib::build_obj(Args... args) called" << std::endl;
#endif
        _obj_ptr = this->sym(args...);
#ifdef ldDEBUG
        std::cout << "LD_Lib::build_obj(Args... args) returned" << std::endl;
#endif
    }

    /* OBJ_PTR */
    std::shared_ptr<T> obj_ptr() { return _obj_ptr; }

   protected:
    create_t *sym;
    void *m_module;
    std::shared_ptr<T> _obj_ptr;
};

}  // namespace gvirtus::common
