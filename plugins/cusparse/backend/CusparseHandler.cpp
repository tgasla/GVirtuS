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
 * Written By: Antonio Pilato <antonio.pilato001@studenti.uniparthenope.it>
 *             Department of Science and Technologies
 *
 * Edited By: Theodoros Aslanidis <theodoros.aslanidis@uniparthenope.it>,
 *              School of Computer Science, University College Dublin
 */

#include "CusparseHandler.h"

using namespace std;
using namespace log4cplus;

using gvirtus::communicators::Buffer;
using gvirtus::communicators::Result;

std::map<string, CusparseHandler::CusparseRoutineHandler>* CusparseHandler::mspHandlers = NULL;

extern "C" std::shared_ptr<CusparseHandler> create_t() {
    return std::make_shared<CusparseHandler>();
}

extern "C" int HandlerInit() { return 0; }

CusparseHandler::CusparseHandler() {
    logger = Logger::getInstance(LOG4CPLUS_TEXT("CusparseHandler"));
    Initialize();
}

CusparseHandler::~CusparseHandler() {}

bool CusparseHandler::CanExecute(std::string routine) {
    return mspHandlers->find(routine) != mspHandlers->end();
}

std::shared_ptr<Result> CusparseHandler::Execute(std::string routine,
                                                 std::shared_ptr<Buffer> input_buffer) {
    LOG4CPLUS_DEBUG(logger, "Called " << routine);
    map<string, CusparseHandler::CusparseRoutineHandler>::iterator it;
    it = mspHandlers->find(routine);
    if (it == mspHandlers->end()) throw runtime_error("No handler for '" + routine + "' found!");
    try {
        return it->second(this, input_buffer);
    } catch (const std::exception& e) {
        LOG4CPLUS_DEBUG(logger, LOG4CPLUS_TEXT("Exception: ") << e.what());
    }
    return NULL;
}

CUSPARSE_ROUTINE_HANDLER(GetVersion) {
    cusparseHandle_t handle = in->Get<cusparseHandle_t>();
    int version;
    cusparseStatus_t cs = cusparseGetVersion(handle, &version);
    LOG4CPLUS_DEBUG(pThis->GetLogger(), "cusparseGetVersion Executed");
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    out->Add<int>(version);
    return std::make_shared<Result>(cs, out);
}

CUSPARSE_ROUTINE_HANDLER(GetErrorString) {
    cusparseStatus_t cs = in->Get<cusparseStatus_t>();
    const char* s = cusparseGetErrorString(cs);
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try {
        out->AddString(s);
    } catch (const std::exception& e) {
        LOG4CPLUS_DEBUG(pThis->GetLogger(), LOG4CPLUS_TEXT("Exception: ") << e.what());
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(pThis->GetLogger(), "cusparseGetErrorString Executed");
    return std::make_shared<Result>(CUSPARSE_STATUS_SUCCESS, out);
}

CUSPARSE_ROUTINE_HANDLER(Create) {
    cusparseHandle_t handle;
    cusparseStatus_t cs = cusparseCreate(&handle);
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try {
        out->Add<cusparseHandle_t>(handle);
    } catch (const std::exception& e) {
        LOG4CPLUS_DEBUG(pThis->GetLogger(), LOG4CPLUS_TEXT("Exception: ") << e.what());
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(pThis->GetLogger(), "cusparseCreate Executed");
    return std::make_shared<Result>(cs, out);
}

CUSPARSE_ROUTINE_HANDLER(Destroy) {
    cusparseHandle_t handle = in->Get<cusparseHandle_t>();
    cusparseStatus_t cs = cusparseDestroy(handle);
    LOG4CPLUS_DEBUG(pThis->GetLogger(), "cusparseDestroy Executed");
    return std::make_shared<Result>(cs);
}

CUSPARSE_ROUTINE_HANDLER(SetStream) {
    cusparseHandle_t handle = in->Get<cusparseHandle_t>();
    cudaStream_t streamId = in->Get<cudaStream_t>();

    cusparseStatus_t cs = cusparseSetStream(handle, streamId);
    LOG4CPLUS_DEBUG(pThis->GetLogger(), "cusparseSetStream Executed");
    return std::make_shared<Result>(cs);
}

CUSPARSE_ROUTINE_HANDLER(GetStream) {
    cusparseHandle_t handle = in->Get<cusparseHandle_t>();
    cudaStream_t streamId;
    cusparseStatus_t cs = cusparseGetStream(handle, &streamId);
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try {
        out->Add<cudaStream_t>(streamId);
    } catch (const std::exception& e) {
        LOG4CPLUS_DEBUG(pThis->GetLogger(), LOG4CPLUS_TEXT("Exception: ") << e.what());
        return std::make_shared<Result>(cs);
    }
    LOG4CPLUS_DEBUG(pThis->GetLogger(), "cusparseGetStream Executed");
    return std::make_shared<Result>(cs, out);
}

void CusparseHandler::Initialize() {
    if (mspHandlers != NULL) return;
    mspHandlers = new map<string, CusparseHandler::CusparseRoutineHandler>();

    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(GetVersion));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(GetErrorString));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Create));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Destroy));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(SetStream));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(GetStream));
}