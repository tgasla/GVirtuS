#include "gvirtus/backend/Process.h"

#include <gvirtus/common/JSON.h>
#include <gvirtus/common/SignalException.h>
#include <gvirtus/common/SignalState.h>
#include "communicators/hybrid/HybridCommunicator.h"
#include <signal.h>
#include <gvirtus/backend/Process.h>
#include <pthread.h>
#include <signal.h>
#include <unistd.h>
#include <functional>
#include <thread>
#include <iostream>

#define DEBUG

using gvirtus::backend::Process;
using gvirtus::common::LD_Lib;
using gvirtus::communicators::Buffer;
using gvirtus::communicators::Communicator;
using gvirtus::communicators::Endpoint;

using std::chrono::steady_clock;

using namespace std;

Process::Process(std::shared_ptr<LD_Lib<Communicator, std::shared_ptr<Endpoint>>> communicator, vector <string> &plugins) : Observable() {
    logger = log4cplus::Logger::getInstance(LOG4CPLUS_TEXT("Process"));

    // Set the logging level
    log4cplus::LogLevel logLevel = log4cplus::INFO_LOG_LEVEL;
    char *val = getenv("GVIRTUS_LOGLEVEL");

    std::string logLevelString = (val == NULL ? std::string("") : std::string(val));

    if (!logLevelString.empty()) {
        logLevel = std::stoi(logLevelString);
    }
    logger.setLogLevel(logLevel);

    signal(SIGCHLD, SIG_IGN);
    _communicator = communicator;
    mPlugins = plugins;
}

bool getstring(Communicator *c, string &s) {
    printf("[getstring] c=%p to_string=%s\n", (void*)c, c?c->to_string().c_str():"<null>");

#ifdef DEBUG
    // print all to_string info
    const char* rtti = "<no-rtti>";
    try { rtti = typeid(*c).name(); } catch (...) {}
    std::string name;
    try { name = c->to_string(); } catch (...) { name = "<no to_string()>"; }
    printf("[getstring] c=%p rtti=%s to_string()=%s\n", (void*)c, rtti, name.c_str());
#endif

    // TODO: FIX LISKOV SUBSTITUTION AND DIPENDENCE INVERSION!!!!!
    if (c->to_string() == "tcpcommunicator") {
        s = "";
        char ch = 0;
        while (c->Read(&ch, 1) == 1) {
            // If reading is ended, return true
            if (ch == 0) {
                return true;
            }
            s += ch;
        }
        return false;
    }
    else if (c->to_string() == "rdmacommunicator") {
        try {
            s = "";
            size_t size = 30;
            char *buf = (char *) malloc(size);
            size = c->Read(buf, size);

            // if read, return true
            if (size > 0) {
                s += std::string(buf);
                printf("append\n");
                //free(buf);
                printf("free\n");
                return true;
            }
        }
        catch (std::string & exc) {
            cerr << exc;
        }
        catch (const char * exc) {
            cerr << std::string(exc);
        }
        return false;
    }
else if (c->to_string() == "hybridcommunicator") {
    s.clear();
    char ch = 0;
    // same as tcp/ip, and stop until read /0
    while (c->Read(&ch, 1) == 1) {
        if (ch == 0) {
            return true;            // take the complete routine name
        }
        s += ch;
    }
    return false;
}

    throw "Communicator getstring read error... Unknown communicator type...";
}

extern std::string getEnvVar(std::string const &key);

std::string getGVirtuSHome() {
    std::string gvirtus_home = getEnvVar("GVIRTUS_HOME");
    return gvirtus_home;
}

void Process::Start() {
    LOG4CPLUS_DEBUG(logger, "✓ - [Process " << getpid() << "] Process::Start() called.");

    for_each(mPlugins.begin(), mPlugins.end(), [this](const std::string &plug) {
                 std::string gvirtus_home = getGVirtuSHome();

                 std::string to_append = "libgvirtus-plugin-" + plug + ".so";
                 LOG4CPLUS_DEBUG(logger, "✓ - [Process " << getpid() << "] appending " << to_append << ".");

                 auto ld_path = fs::path(gvirtus_home + "/lib").append(to_append);

                 try {
                     auto dl = std::make_shared<LD_Lib<Handler>>(ld_path, "create_t");
                     dl->build_obj();
                     _handlers.push_back(dl);
                 }
                 catch (const std::string &e) {
                     LOG4CPLUS_ERROR(logger, e);
                 }
             }
    );

    // inserisci i sym dei plugin in h
    std::function<void(Communicator *)> execute = [=](Communicator *client_comm) {
        LOG4CPLUS_DEBUG(logger, "✓ - [Process " << getpid() << "]" << "Process::Start()'s \"execute\" lambda called");
        // carica i puntatori ai simboli dei moduli in mHandlers

        string routine;
        std::shared_ptr<Buffer> input_buffer = std::make_shared<Buffer>();

        while (getstring(client_comm, routine)) {
    LOG4CPLUS_DEBUG(logger, "✓ - Received routine " << routine);

    // === before reading buffer, chose the protocol of this round by rountine ===
    gvirtus::communicators::HybridCommunicator* hybrid = nullptr;
    if (client_comm && client_comm->to_string() == "hybridcommunicator") {
        hybrid = dynamic_cast<gvirtus::communicators::HybridCommunicator*>(client_comm);
    }
    if (hybrid) {
        // all those function payload will transfer by RDMA
        const bool use_rdma =
            routine.rfind("cudaRegisterFatBinary", 0)== 0 ||
            routine.rfind("cudaRegisterFatBinaryEnd", 0)== 0 ||
            routine.rfind("cudaMemcpyAsync", 0) == 0 ||   
            routine.rfind("cudaMemcpy",      0) == 0;     

        if (use_rdma) {
            // bytes_hint if >0 ,then trigger the first 8B under TCP moniter.
            // real payload size after 8B head.
            hybrid->begin_call(routine, gvirtus::communicators::Transport::RDMA, /*bytes_hint*/ 1);
        } else {
            hybrid->begin_call(routine, gvirtus::communicators::Transport::TCP, 0);
        }
    }

    // now reading buffer：8B from TCP, payload will transfer by the selected protocol
    input_buffer->Reset(client_comm);

    std::shared_ptr<Handler> h = nullptr;
    for (auto &ptr_el : _handlers) {
        if (ptr_el->obj_ptr()->CanExecute(routine)) {
            h = ptr_el->obj_ptr();
            break;
        }
    }

    std::shared_ptr<communicators::Result> result;
    if (h == nullptr) {
        LOG4CPLUS_ERROR(logger, "✖ - [Process " << getpid() << "]: Requested unknown routine " << routine << ".");
        result = std::make_shared<communicators::Result>(-1, std::make_shared<Buffer>());
    } else {
        auto start = steady_clock::now();
        result = h->Execute(routine, input_buffer);
        result->TimeTaken(std::chrono::duration_cast<std::chrono::milliseconds>(steady_clock::now() - start).count() / 1000.0);
    }

    // return info：control the head transfer by TCP，then payload RDMA
    result->Dump(client_comm);

    // stop this round, and clean all context
    if (hybrid) {
        hybrid->end_call();
    }

    if (result->GetExitCode() != 0 && routine.compare("cudaLaunch")) {
        LOG4CPLUS_DEBUG(logger, "✓ - [Process " << getpid() << "]: Requested '" << routine << "' routine.");
        LOG4CPLUS_DEBUG(logger, "✓ - - [Process " << getpid() << "]: Exit Code '" << result->GetExitCode() << "'.");
    }
}

        Notify("process-ended");
    };

    /*
    common::SignalState sig_hand;
    sig_hand.setup_signal_state(SIGINT);
*/

    try {
        _communicator->obj_ptr()->Serve();

        int pid = 0;
        while (true) {
            Communicator *client = const_cast<Communicator *>(_communicator->obj_ptr()->Accept());
            printf("[Process] Accept client=%p, comm=%s\n", (void*)client, client?client->to_string().c_str():"<null>");

            if (client != nullptr) {
                //      if ((pid = fork()) == 0) {
                std::thread(execute, client).detach();
                //        exit(0);
                //      }

            } else
                _communicator->obj_ptr()->run();

            // check if process received SIGINT

            if (common::SignalState::get_signal_state(SIGINT)) {
                LOG4CPLUS_DEBUG(logger, "✓ - SIGINT received, killing server on [Process " << getpid() << "]...");
                break;
            }

        }
    }
    catch (std::string &exc) {
        LOG4CPLUS_ERROR(logger, "✖ - [Process " << getpid() << "]: " << exc);
    }

    LOG4CPLUS_DEBUG(logger, "✓ - Process::Start() returned [Process " << getpid() << "].");
    //exit(EXIT_SUCCESS);
}

Process::~Process() {
    _communicator.reset();
    _handlers.clear();
    mPlugins.clear();
}
