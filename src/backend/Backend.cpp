#include "gvirtus/backend/Backend.h"

#include <gvirtus/communicators/CommunicatorFactory.h>
#include <gvirtus/communicators/EndpointFactory.h>

#include <sys/wait.h>
#include <unistd.h>
#include <cerrno>

using gvirtus::backend::Backend;

Backend::Backend(const fs::path &path) {
    // logger setup
    this->logger = log4cplus::Logger::getInstance(LOG4CPLUS_TEXT("Backend"));

    char *logLevel_envVar = getenv("GVIRTUS_LOGLEVEL");
    std::string logLevelString = (logLevel_envVar == nullptr ? std::string("") : std::string(logLevel_envVar));

    log4cplus::LogLevel logLevel = logLevelString.empty() ? log4cplus::INFO_LOG_LEVEL : std::stoi(logLevelString);
    this->logger.setLogLevel(logLevel);

    // json setup
    if (not (fs::exists(path) and fs::is_regular_file(path) and path.extension() == ".json")) {
        LOG4CPLUS_ERROR(logger, "✖ - " << fs::path(__FILE__).filename() << ":" << __LINE__ << ":" << " json path error: no such file.");
        exit(EXIT_FAILURE);
    }

    LOG4CPLUS_DEBUG(logger, "✓ - " << fs::path(__FILE__).filename() << ":" << __LINE__ << ":" << " Json file has been loaded.");

    // endpoints setup
    LOG4CPLUS_TRACE(logger, "🛈  - Initializing endpoints setup");

    _properties = common::JSON<Property>(path).parser();
    _children.reserve(_properties.endpoints());

    LOG4CPLUS_TRACE(logger, "🛈  - Got properties and reserved children array");

    if (_properties.endpoints() > 1) LOG4CPLUS_INFO(logger, "🛈  - Application serves on " << _properties.endpoints() << " several endpoint");

    try {
        for (int i = 0; i < _properties.endpoints(); i++) {
            _children.push_back(
                    std::make_unique<Process>(
                            communicators::CommunicatorFactory::get_communicator(
                                    communicators::EndpointFactory::get_endpoint(path),
                                    _properties.secure()
                            ),
                            _properties.plugins().at(i)
                    )
            );
        }
        /*
        for (int i = 0; i < _properties.endpoints(); i++) {
            LOG4CPLUS_TRACE(logger, "🛈  - Setting up process " << i << ":");

            auto secure = _properties.secure();
            LOG4CPLUS_TRACE(logger, "🛈  - Secure:  " << secure);

            auto endpoint = communicators::EndpointFactory::get_endpoint(path);
            LOG4CPLUS_TRACE(logger, "🛈  - Endpoint: ok!");

            auto communicator = communicators::CommunicatorFactory::get_communicator(endpoint, secure);
            LOG4CPLUS_TRACE(logger, "🛈  - Communicator: ok!");

            auto plugins = _properties.plugins().at(i);
            LOG4CPLUS_TRACE(logger, "🛈  - Properties: ok!");

            auto child = std::make_unique<Process>(communicator, plugins);
            LOG4CPLUS_TRACE(logger, "🛈  - Process: ok!");

            _children.push_back(child);
        }
         */
    }
    catch (const char * exc) {
        LOG4CPLUS_ERROR(logger, "🛈  - Exception during process setup: " << exc);
    }
    catch (std::string & exc) {
        LOG4CPLUS_ERROR(logger, "🛈  - Exception during process setup: " << exc);
    }
    catch (std::logic_error & exc) {
        LOG4CPLUS_ERROR(logger, "🛈  - Exception during process setup: " << exc.what());
    }


    LOG4CPLUS_INFO(logger, "🛈  - Backend Initialization is complete!");
}

void Backend::Start() {
    // std::function<void(std::unique_ptr<gvirtus::Thread> & children)> task =
    // [this](std::unique_ptr<gvirtus::Thread> &children) {
    //   LOG4CPLUS_DEBUG(logger, "✓ - [Thread " << std::this_thread::get_id() <<
    //   "]: Started."); children->Start(); LOG4CPLUS_DEBUG(logger, "✓ - [Thread "
    //   << std::this_thread::get_id() << "]: Finished.");
    // };
    LOG4CPLUS_DEBUG(logger, "✓ - [Process " << getpid() << "] " << "Backend::Start() called.");

    int pid = 0;

    // _children definition: "std::vector<std::unique_ptr<Process>> _children"
    for (int i = 0; i < _children.size(); i++) {
        activeChilds++;
        if ((pid = fork()) == 0) {
            _children[i]->Start();
            LOG4CPLUS_TRACE(logger, "Child exited.");
            break;
        }
    }

    /* PARENT */
    pid_t pid_wait = 0;
    int stat_loc;
    if (pid != 0) {
        signal(SIGINT, SIG_IGN);
        signal(SIGHUP, SIG_IGN);

        LOG4CPLUS_TRACE(logger, "Active childs: " << activeChilds);

        int status;
        do {
            LOG4CPLUS_DEBUG(logger, "✓ - [Process " << getpid() << "] " << "Waiting for childs to terminate. Current active childs: " << activeChilds);
            int waitres = wait(&status);
            activeChilds--;

            LOG4CPLUS_TRACE(logger, "Active childs: %d" << activeChilds);

            if (waitres < 0) {
                LOG4CPLUS_TRACE(logger, "Error " << strerror(errno) << " on wait.");
            }
            else {
                LOG4CPLUS_TRACE(logger, "Process " << waitres << " returned successfully.");
                break;
            }
        } while (not WIFEXITED(status) and not WIFSIGNALED(status));

        LOG4CPLUS_INFO(logger, "✓ - No child processes are currently running. Use CTRL + C to terminate the backend.");

        signal(SIGINT, sigint_handler);
        pause();
    }

    LOG4CPLUS_DEBUG(logger, "✓ - [Process " << getpid() << "] " << "Backend::Start() returned.");
}

void Backend::EventOccurred(std::string &event, void *object) {
    LOG4CPLUS_DEBUG(logger, "✓ - EventOccurred: " << event);
}

