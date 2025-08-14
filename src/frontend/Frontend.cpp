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
 * @file   Frontend.cpp
 * @author Giuseppe Coviello <giuseppe.coviello@uniparthenope.it>
 * @date   Wed Sep 30 12:57:11 2009
 *
 * @brief
 *
 *
 */

#include <gvirtus/communicators/CommunicatorFactory.h>
#include <gvirtus/communicators/EndpointFactory.h>
#include <gvirtus/frontend/Frontend.h>
#include <filesystem>
#include "communicators/hybrid/HybridCommunicator.h"

#include <pthread.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <unistd.h>
#include <iostream>
#include <mutex>
#include <chrono>
#include <stdlib.h> /* getenv */

#include "log4cplus/configurator.h"
#include "log4cplus/logger.h"
#include "log4cplus/loggingmacros.h"
using std::chrono::steady_clock;
using std::chrono::duration_cast;
using std::chrono::milliseconds;
using namespace std;

using gvirtus::communicators::Buffer;
using gvirtus::communicators::Communicator;
using gvirtus::communicators::CommunicatorFactory;
using gvirtus::communicators::EndpointFactory;
using gvirtus::frontend::Frontend;


static Frontend msFrontend;
std::mutex gFrontendMutex;
map<pthread_t, Frontend *> *Frontend::mpFrontends = NULL;
static bool initialized = false;

log4cplus::Logger logger;

std::string getEnvVar(std::string const &key) {
    char *env_var = getenv(key.c_str());
    return (env_var == nullptr) ? std::string("") : std::string(env_var);
}

void Frontend::Init(Communicator *c) {
    // Logger configuration
    log4cplus::BasicConfigurator basicConfigurator;
    basicConfigurator.configure();
    logger = log4cplus::Logger::getInstance(LOG4CPLUS_TEXT("GVirtuS Frontend"));

    // Set the logging level
    std::string logLevelString = getEnvVar("GVIRTUS_LOGLEVEL");
    log4cplus::LogLevel logLevel = log4cplus::INFO_LOG_LEVEL;
    if (!logLevelString.empty()) {
        try {
            logLevel = static_cast<log4cplus::LogLevel>(std::stoi(logLevelString));
        } catch (const std::exception& e) {
            LOG4CPLUS_ERROR(logger, fs::path(__FILE__).filename() << ":" << __LINE__ << ": Exception occurred: " << e.what());
            logLevel = log4cplus::INFO_LOG_LEVEL;
        }
    }

    logger.setLogLevel(logLevel);

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

    LOG4CPLUS_INFO(logger, "GVirtuS frontend version " + config_path);

    try {
        auto endpoint = EndpointFactory::get_endpoint(config_path);

        mpFrontends->find(tid)->second->_communicator = CommunicatorFactory::get_communicator(endpoint);
        mpFrontends->find(tid)->second->_communicator->obj_ptr()->Connect();
    }
    catch (const std::exception& e) {
        LOG4CPLUS_ERROR(logger, fs::path(__FILE__).filename() << ":" << __LINE__ << ":" << " Exception occurred: " << e.what());
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
        bool dump_stats = env && (strcasecmp(env, "on") == 0 || strcasecmp(env, "true") == 0 || strcmp(env, "1") == 0);

        // Safe iteration while erasing entries
        for (auto it = mpFrontends->begin(); it != mpFrontends->end(); /* no increment here */) {
            if (it->second == this) {
                it = mpFrontends->erase(it);
                continue;
            }

            if (dump_stats) {
                std::cerr << "[GVIRTUS_STATS] Executed " << it->second->mRoutinesExecuted << " routine(s) in "
                        << it->second->mRoutineExecutionTime << " second(s)\n"
                        << "[GVIRTUS_STATS] Sent " << it->second->mDataSent / (1024 * 1024.0) << " Mb(s) in "
                        << it->second->mSendingTime
                        << " second(s)\n"
                        << "[GVIRTUS_STATS] Received " << it->second->mDataReceived / (1024 * 1024.0) << " Mb(s) in "
                        << it->second->mReceivingTime
                        << " second(s)\n";
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
        if (mpFrontends == nullptr)
            mpFrontends = new map<pthread_t, Frontend *>();
    }

    pid_t tid = syscall(SYS_gettid);  // getting frontend's tid

    {
        std::lock_guard<std::mutex> lock(gFrontendMutex);
        auto it = mpFrontends->find(tid);
        if (it != mpFrontends->end())
            return it->second;
    }

    Frontend *f = new Frontend();
    try {
        f->Init(c);
        {
            std::lock_guard<std::mutex> lock(gFrontendMutex);
            mpFrontends->insert(make_pair(tid, f));
        }
    }
    catch (const std::exception& e) {
        LOG4CPLUS_ERROR(logger, "Error initializing Frontend: " << e.what());
        delete f;  // Clean up on failure
        return nullptr;
    }

    return f;
}

void Frontend::Execute(const char *routine, const Buffer *input_buffer) {
    if (input_buffer == nullptr) input_buffer = mpInputBuffer.get();

    pid_t tid = syscall(SYS_gettid);
    pid_t pid = getpid();
    size_t in_size = input_buffer->GetBufferSize();
    int exit_code = 0;
    double server_exec_sec = 0.0;
    double send_sec = 0.0;
    double recv_sec = 0.0;

    Frontend* frontend = nullptr;
    {
        std::lock_guard<std::mutex> lock(gFrontendMutex);
        auto it = mpFrontends->find(tid);
        if (it == mpFrontends->end()) {
            LOG4CPLUS_ERROR(logger, "Cannot send any job request");
            return;
        }
        frontend = it->second;
    }

    LOG4CPLUS_DEBUG(logger, "DEBUG - Received routine " << routine
                      << " [pid=" << pid << ", tid=" << tid << "]");

    frontend->mRoutinesExecuted++;

    // ===== send routine info first（under TCP）=====
    auto start_send = steady_clock::now();
    frontend->_communicator->obj_ptr()->Write(routine, strlen(routine) + 1);

    // ===== chose protocol by different routine =====
    if (frontend->_communicator->obj_ptr()->to_string() == "hybridcommunicator") {
        auto* hybrid = dynamic_cast<gvirtus::communicators::HybridCommunicator*>(
            frontend->_communicator->obj_ptr().get());
        if (hybrid) {
            if (std::string(routine).find("cudaMemcpy") != std::string::npos ||
                std::string(routine).find("cudaRegisterFatBinary")   != std::string::npos ||
                std::string(routine).find("cudaRegisterFatBinaryEnd")   != std::string::npos ||
                std::string(routine).find("cudaMemcpyAsync")   != std::string::npos) {
                hybrid->begin_call(routine, gvirtus::communicators::Transport::RDMA, in_size);
            } else {
                hybrid->begin_call(routine, gvirtus::communicators::Transport::TCP, in_size);
            }
        }
    }

    // ===== send paramemter data =====
    frontend->mDataSent += in_size;
    LOG4CPLUS_DEBUG(logger, "Write " << in_size << " bytes to the buffer");
    input_buffer->Dump(frontend->_communicator->obj_ptr().get());

    // ===== sync by chosen channle =====
    frontend->_communicator->obj_ptr()->Sync();

    send_sec = duration_cast<milliseconds>(
                   steady_clock::now() - start_send).count() / 1000.0;

    frontend->mpOutputBuffer->Reset();

    // ===== receive exit_code =====
    auto start_recv = steady_clock::now();
    frontend->_communicator->obj_ptr()->Read((char *)&exit_code, sizeof(int));
    frontend->mExitCode = exit_code;

    // ===== receive backend time cost =====
    frontend->_communicator->obj_ptr()->Read(
        reinterpret_cast<char *>(&server_exec_sec), sizeof(server_exec_sec));

    // ===== receive output buffer =====
    size_t out_buffer_size = 0;
    frontend->_communicator->obj_ptr()->Read((char *)&out_buffer_size, sizeof(size_t));
    frontend->mDataReceived += out_buffer_size;
    LOG4CPLUS_DEBUG(logger, "Read " << out_buffer_size << " bytes from the buffer");
    if (out_buffer_size > 0) {
        frontend->mpOutputBuffer->Read<char>(
            frontend->_communicator->obj_ptr().get(), out_buffer_size);
    }
    recv_sec = duration_cast<milliseconds>(
                   steady_clock::now() - start_recv).count() / 1000.0;

    // ===== update info =====
    frontend->mRoutineExecutionTime += server_exec_sec;
    frontend->mSendingTime += send_sec;
    frontend->mReceivingTime += recv_sec;

    // ===== print log =====
    LOG4CPLUS_DEBUG(logger,
        "Routine '" << routine << "' returned " << exit_code
        << " | server_exec=" << server_exec_sec << "s"
        << " | send=" << send_sec << "s"
        << " | recv=" << recv_sec << "s"
        << " | in=" << in_size << "B"
        << " | out=" << out_buffer_size << "B"
        << " | pid=" << pid << " tid=" << tid);

    LOG4CPLUS_DEBUG(logger, "DEBUG - Called: " << routine);

    // ===== stop this call，clean HybridCommunicator status =====
    if (frontend->_communicator->obj_ptr()->to_string() == "hybridcommunicator") {
        auto hybrid = std::dynamic_pointer_cast<gvirtus::communicators::HybridCommunicator>(
            frontend->_communicator->obj_ptr());
        if (hybrid) {
            hybrid->end_call();
        }
    }
}

void Frontend::Prepare() {
    pid_t tid = syscall(SYS_gettid);
    {
        if (this->mpFrontends->find(tid) != mpFrontends->end())
            mpFrontends->find(tid)->second->mpInputBuffer->Reset();
    }
}
