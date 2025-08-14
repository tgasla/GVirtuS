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
 * Written By: Antonio Pilato <antonio.pilato001@studenti.uniparthenope.it>,
 *             Department of Science and Technologies
 *
 * Edited By: Theodoros Aslanidis <theodoros.aslanidis@ucdconnect.ie>,
 *            School of Computer Science, University College Dublin
 */

#include "CusolverHandler.h"

using namespace std;
using namespace log4cplus;

using gvirtus::communicators::Buffer;
using gvirtus::communicators::Result;

std::map<string, CusolverHandler::CusolverRoutineHandler>* CusolverHandler::mspHandlers = NULL;

extern "C" std::shared_ptr<CusolverHandler> create_t() {
    return std::make_shared<CusolverHandler>();
}

extern "C" int HandlerInit() { return 0; }

CusolverHandler::CusolverHandler() {
    logger = Logger::getInstance(LOG4CPLUS_TEXT("CusolverHandler"));
    Initialize();
}

CusolverHandler::~CusolverHandler() {}

bool CusolverHandler::CanExecute(std::string routine) {
    return mspHandlers->find(routine) != mspHandlers->end();
}

std::shared_ptr<Result> CusolverHandler::Execute(std::string routine,
                                                 std::shared_ptr<Buffer> input_buffer) {
    LOG4CPLUS_DEBUG(logger, "Called " << routine);
    map<string, CusolverHandler::CusolverRoutineHandler>::iterator it;
    it = mspHandlers->find(routine);
    if (it == mspHandlers->end()) throw runtime_error("No handler for '" + routine + "' found!");
    try {
        return it->second(this, input_buffer);
    } catch (const std::exception& e) {
        LOG4CPLUS_DEBUG(logger, LOG4CPLUS_TEXT("Exception: ") << e.what());
    }
    return NULL;
}

CUSOLVER_ROUTINE_HANDLER(DnCreate) {
    cusolverDnHandle_t handle;
    cusolverStatus_t cs = cusolverDnCreate(&handle);
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try {
        out->Add<cusolverDnHandle_t>(handle);
    } catch (const std::exception& e) {
        LOG4CPLUS_DEBUG(pThis->GetLogger(), LOG4CPLUS_TEXT("Exception: ") << e.what());
        return std::make_shared<Result>(CUSOLVER_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(pThis->GetLogger(), "cusolverDnCreate Executed");
    return std::make_shared<Result>(cs, out);
}

CUSOLVER_ROUTINE_HANDLER(DnDestroy) {
    cusolverDnHandle_t handle = in->Get<cusolverDnHandle_t>();
    cusolverStatus_t cs = cusolverDnDestroy(handle);
    LOG4CPLUS_DEBUG(pThis->GetLogger(), "cusolverDnDestroy Executed");
    return std::make_shared<Result>(cs);
}

CUSOLVER_ROUTINE_HANDLER(DnSetStream) {
    cusolverDnHandle_t handle = in->Get<cusolverDnHandle_t>();
    cudaStream_t streamId = in->Get<cudaStream_t>();

    cusolverStatus_t cs = cusolverDnSetStream(handle, streamId);
    LOG4CPLUS_DEBUG(pThis->GetLogger(), "cusolverDnSetStream Executed");
    return std::make_shared<Result>(cs);
}

CUSOLVER_ROUTINE_HANDLER(DnGetStream) {
    cusolverDnHandle_t handle = in->Get<cusolverDnHandle_t>();
    cudaStream_t streamId;
    cusolverStatus_t cs = cusolverDnGetStream(handle, &streamId);
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try {
        out->Add<cudaStream_t>(streamId);
    } catch (const std::exception& e) {
        LOG4CPLUS_DEBUG(pThis->GetLogger(), LOG4CPLUS_TEXT("Exception: ") << e.what());
        return std::make_shared<Result>(cs);
    }
    LOG4CPLUS_DEBUG(pThis->GetLogger(), "cusolverDnGetStream Executed");
    return std::make_shared<Result>(cs, out);
}

void CusolverHandler::Initialize() {
    if (mspHandlers != NULL) return;
    mspHandlers = new map<string, CusolverHandler::CusolverRoutineHandler>();

    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnCreate));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnDestroy));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnSetStream));
    mspHandlers->insert(CUSOLVER_ROUTINE_HANDLER_PAIR(DnGetStream));
}