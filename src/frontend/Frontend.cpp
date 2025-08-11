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
 * Written By: Carlo Palmieri <carlo.palmieri@uniparthenope.it>,
 *             Department of Applied Science
 *             Giuseppe Coviello <giuseppe.coviello@uniparthenope.it>,
 *             Department of Applied Science
 *             Raffaele Montella <raffaele.montella@uniparthenope.it>,
 *             Department of Science and Technologies
 *             Antonio Mentone <antonio.mentone@uniparthenope.it>,
 *             Department of Science and Technologies
 * Edited By: Mariano Aponte <aponte2001@gmail.com>,
 *            Department of Science and Technologies, University of Naples Parthenope
 *            Theodoros Aslanidis <theodoros.aslanidis@ucdconnect.ie>,
 *            Department of Computer Science, University College Dublin
 */

#include <gvirtus/communicators/CommunicatorFactory.h>
#include <gvirtus/communicators/EndpointFactory.h>
#include <gvirtus/frontend/Frontend.h>
#include <pthread.h>
#include <stdlib.h> /* getenv */
#include <sys/syscall.h>
#include <sys/types.h>
#include <unistd.h>

#include <chrono>
#include <iostream>
#include <mutex>

#include "log4cplus/configurator.h"
#include "log4cplus/logger.h"
#include "log4cplus/loggingmacros.h"

using namespace std;
using namespace log4cplus;

using gvirtus::communicators::Buffer;
using gvirtus::communicators::Communicator;
using gvirtus::communicators::CommunicatorFactory;
using gvirtus::communicators::EndpointFactory;
using gvirtus::frontend::Frontend;

using std::chrono::steady_clock;

static Frontend msFrontend;
std::mutex gFrontendMutex;
map<pthread_t, Frontend *> *Frontend::mpFrontends = NULL;
static bool initialized = false;

Logger logger;

std::string getEnvVar(std::string const &key) {
    char *env_var = getenv(key.c_str());
    return (env_var == nullptr) ? std::string("") : std::string(env_var);
}

void Frontend::Init(Communicator *c) {
    // Logger configuration
    BasicConfigurator basicConfigurator;
    basicConfigurator.configure();

    // Set the logging level
    std::string logLevelString = getEnvVar("GVIRTUS_LOGLEVEL");
    LogLevel logLevel = INFO_LOG_LEVEL;
    if (!logLevelString.empty()) {
        try {
            logLevel = static_cast<LogLevel>(std::stoi(logLevelString));
        } catch (const std::exception &e) {
            std::cerr << "[GVIRTUS WARNING] Invalid GVIRTUS_LOGLEVEL value: '" << logLevelString
                      << "'. Using default INFO_LOG_LEVEL. (" << e.what() << ")\n";
            logLevel = INFO_LOG_LEVEL;
        }
    }

    Logger root = Logger::getRoot();
    root.setLogLevel(logLevel);

    logger = Logger::getInstance(LOG4CPLUS_TEXT("Frontend"));

    pid_t tid = syscall(SYS_gettid);

    // Get the GVIRTUS_CONFIG environment varibale
    std::string config_path = getEnvVar("GVIRTUS_CONFIG");

    // Check if the configuration file is defined
    if (config_path.empty()) {
        // Check if the configuration file is in the GVIRTUS_HOME directory
        config_path = getEnvVar("GVIRTUS_HOME") + "/etc/properties.json";
        if (config_path.empty()) {
            // Finally consider the current directory
            config_path = "./properties.json";
        }
    }

    std::unique_ptr<char> default_endpoint;

    // no frontend found
    {
        std::lock_guard<std::mutex> lock(gFrontendMutex);
        if (mpFrontends->find(tid) == mpFrontends->end()) {
            Frontend *f = new Frontend();
            mpFrontends->insert(make_pair(tid, f));
        }
    }

    LOG4CPLUS_INFO(logger, "Using properties file: " + config_path);

    try {
        auto endpoint = EndpointFactory::get_endpoint(config_path);

        mpFrontends->find(tid)->second->_communicator =
            CommunicatorFactory::get_communicator(endpoint);
        mpFrontends->find(tid)->second->_communicator->obj_ptr()->Connect();
    } catch (const std::exception &e) {
        LOG4CPLUS_FATAL(logger, fs::path(__FILE__).filename()
                                    << ":" << __LINE__ << ":"
                                    << " Exception occurred: " << e.what());
        exit(EXIT_FAILURE);
    }

    mpFrontends->find(tid)->second->mpInputBuffer = std::make_shared<Buffer>();
    mpFrontends->find(tid)->second->mpOutputBuffer = std::make_shared<Buffer>();
    mpFrontends->find(tid)->second->mpLaunchBuffer = std::make_shared<Buffer>();
    mpFrontends->find(tid)->second->mExitCode = -1;
    mpFrontends->find(tid)->second->mpInitialized = true;
}

Frontend::~Frontend() {
    static bool destroying = false;
    if (destroying || mpFrontends == nullptr) return;
    destroying = true;

    std::lock_guard<std::mutex> lock(gFrontendMutex);
    {
        pid_t tid = syscall(SYS_gettid);

        auto env = getenv("GVIRTUS_DUMP_STATS");
        bool dump_stats = env && (strcasecmp(env, "on") == 0 || strcasecmp(env, "true") == 0 ||
                                  strcmp(env, "1") == 0);

        // Safe iteration while erasing entries
        for (auto it = mpFrontends->begin(); it != mpFrontends->end(); /* no increment here */) {
            if (it->second == this) {
                it = mpFrontends->erase(it);
                continue;
            }

            if (dump_stats) {
                std::cerr << "[GVIRTUS_STATS] Executed " << it->second->mRoutinesExecuted
                          << " routine(s) in " << it->second->mRoutineExecutionTime
                          << " second(s)\n"
                          << "[GVIRTUS_STATS] Sent " << it->second->mDataSent / (1024 * 1024.0)
                          << " Mb(s) in " << it->second->mSendingTime << " second(s)\n"
                          << "[GVIRTUS_STATS] Received "
                          << it->second->mDataReceived / (1024 * 1024.0) << " Mb(s) in "
                          << it->second->mReceivingTime << " second(s)\n";
            }

            delete it->second;
            it = mpFrontends->erase(it);
        }

        // Delete the map itself and set pointer to nullptr
        delete mpFrontends;
        mpFrontends = nullptr;
    }
}

Frontend *Frontend::GetFrontend(Communicator *c) {
    {
        std::lock_guard<std::mutex> lock(gFrontendMutex);
        if (mpFrontends == nullptr) mpFrontends = new map<pthread_t, Frontend *>();
    }

    pid_t tid = syscall(SYS_gettid);  // getting frontend's tid

    {
        std::lock_guard<std::mutex> lock(gFrontendMutex);
        auto it = mpFrontends->find(tid);
        if (it != mpFrontends->end()) return it->second;
    }

    Frontend *f = new Frontend();
    try {
        f->Init(c);
        {
            std::lock_guard<std::mutex> lock(gFrontendMutex);
            mpFrontends->insert(make_pair(tid, f));
        }
    } catch (const std::exception &e) {
        LOG4CPLUS_ERROR(logger, "Error initializing Frontend: " << e.what());
        delete f;  // Clean up on failure
        return nullptr;
    }

    return f;
}

void Frontend::Execute(const char *routine, const Buffer *input_buffer) {
    if (input_buffer == nullptr) input_buffer = mpInputBuffer.get();
    // if (!strcmp(routine, "cudaLaunchKernel")) {
    //     cerr << "cudaLaunchKernel called" << endl;
    // }

    pid_t tid = syscall(SYS_gettid);

    Frontend *frontend = nullptr;
    {
        std::lock_guard<std::mutex> lock(gFrontendMutex);
        auto it = mpFrontends->find(tid);
        if (it == mpFrontends->end()) {
            LOG4CPLUS_ERROR(logger, "Cannot send any job request");
            return;
        }
        frontend = it->second;
    }

    frontend->mRoutinesExecuted++;
    auto start = steady_clock::now();
    frontend->_communicator->obj_ptr()->Write(routine, strlen(routine) + 1);
    frontend->mDataSent += input_buffer->GetBufferSize();
    input_buffer->Dump(frontend->_communicator->obj_ptr().get());
    frontend->_communicator->obj_ptr()->Sync();
    frontend->mSendingTime +=
        std::chrono::duration_cast<std::chrono::milliseconds>(steady_clock::now() - start).count() /
        1000.0;
    frontend->mpOutputBuffer->Reset();

    frontend->_communicator->obj_ptr()->Read((char *)&frontend->mExitCode, sizeof(int));
    LOG4CPLUS_DEBUG(logger, "Routine '" << routine << "' returned " << frontend->mExitCode);
    // if (frontend->mExitCode != 0
    //     && strcmp(routine, "cudnnGetVersion") != 0
    //     && strcmp(routine, "cudnnGetErrorString") != 0
    //     && strcmp(routine, "cusolverDnGetVersion") != 0
    //     && strcmp(routine, "cudaGetErrorString") != 0
    //     && strcmp(routine, "cudaGetErrorName") != 0
    //     && strcmp(routine, "cusparseGetErrorString") != 0
    //     && strcmp(routine, "nvrtcGetErrorString") != 0
    // ) {
    //     LOG4CPLUS_ERROR(logger, "Error executing routine '" << routine << "': exit code " <<
    //     frontend->mExitCode); return;
    // }
    double time_taken;
    frontend->_communicator->obj_ptr()->Read(reinterpret_cast<char *>(&time_taken),
                                             sizeof(time_taken));
    frontend->mRoutineExecutionTime += time_taken;

    start = steady_clock::now();
    size_t out_buffer_size;
    frontend->_communicator->obj_ptr()->Read((char *)&out_buffer_size, sizeof(size_t));
    LOG4CPLUS_DEBUG(logger, "Output buffer size: " << out_buffer_size);
    frontend->mDataReceived += out_buffer_size;
    if (out_buffer_size > 0) {
        LOG4CPLUS_DEBUG(logger, "Output buffer size is greater than 0, reading...");
        frontend->mpOutputBuffer->Read<char>(frontend->_communicator->obj_ptr().get(),
                                             out_buffer_size);
        LOG4CPLUS_DEBUG(logger, "Output buffer read successfully.");
    }
    frontend->mReceivingTime +=
        std::chrono::duration_cast<std::chrono::milliseconds>(steady_clock::now() - start).count() /
        1000.0;
}

void Frontend::Prepare() {
    pid_t tid = syscall(SYS_gettid);
    {
        if (this->mpFrontends->find(tid) != mpFrontends->end())
            mpFrontends->find(tid)->second->mpInputBuffer->Reset();
    }
}
