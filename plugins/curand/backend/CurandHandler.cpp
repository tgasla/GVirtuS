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
 * Written By: Vincenzo Santopietro <vincenzo.santopietro@uniparthenope.it>,
 *             Department of Science and Technologies
 *
 * Edited By: Theodoros Aslanidis <theodoros.aslanidis@ucdconnect.ie>,
 *            School of Computer Science, University College Dublin
 */

#include "CurandHandler.h"

using namespace std;
using namespace log4cplus;

using gvirtus::communicators::Buffer;
using gvirtus::communicators::Result;

std::map<string, CurandHandler::CurandRoutineHandler>* CurandHandler::mspHandlers = NULL;

extern "C" std::shared_ptr<CurandHandler> create_t() { return std::make_shared<CurandHandler>(); }

CurandHandler::CurandHandler() {
    logger = Logger::getInstance(LOG4CPLUS_TEXT("CurandHandler"));
    Initialize();
}

CurandHandler::~CurandHandler() {}

bool CurandHandler::CanExecute(std::string routine) {
    return mspHandlers->find(routine) != mspHandlers->end();
}

std::shared_ptr<Result> CurandHandler::Execute(std::string routine, std::shared_ptr<Buffer> in) {
    LOG4CPLUS_DEBUG(logger, "Called " << routine);
    map<string, CurandHandler::CurandRoutineHandler>::iterator it;
    it = mspHandlers->find(routine);
    if (it == mspHandlers->end())
        throw runtime_error(std::string("No handler for '") + routine + std::string("' found!"));
    try {
        return it->second(this, in);
    } catch (const std::exception& e) {
        LOG4CPLUS_DEBUG(logger, LOG4CPLUS_TEXT("Exception: ") << e.what());
    }
    return NULL;
}

void CurandHandler::Initialize() {
    if (mspHandlers != NULL) return;
    mspHandlers = new map<string, CurandHandler::CurandRoutineHandler>();

    /* CurandHandler Query Platform Info */
    mspHandlers->insert(CURAND_ROUTINE_HANDLER_PAIR(CreateGenerator));
    mspHandlers->insert(CURAND_ROUTINE_HANDLER_PAIR(CreateGeneratorHost));
    mspHandlers->insert(CURAND_ROUTINE_HANDLER_PAIR(SetPseudoRandomGeneratorSeed));
    mspHandlers->insert(CURAND_ROUTINE_HANDLER_PAIR(SetGeneratorOffset));
    mspHandlers->insert(CURAND_ROUTINE_HANDLER_PAIR(SetQuasiRandomGeneratorDimensions));
    mspHandlers->insert(CURAND_ROUTINE_HANDLER_PAIR(Generate));
    mspHandlers->insert(CURAND_ROUTINE_HANDLER_PAIR(GenerateLongLong));
    mspHandlers->insert(CURAND_ROUTINE_HANDLER_PAIR(GenerateUniform));
    mspHandlers->insert(CURAND_ROUTINE_HANDLER_PAIR(GenerateNormal));
    mspHandlers->insert(CURAND_ROUTINE_HANDLER_PAIR(GenerateLogNormal));
    mspHandlers->insert(CURAND_ROUTINE_HANDLER_PAIR(GeneratePoisson));
    mspHandlers->insert(CURAND_ROUTINE_HANDLER_PAIR(GenerateUniformDouble));
    mspHandlers->insert(CURAND_ROUTINE_HANDLER_PAIR(GenerateNormalDouble));
    mspHandlers->insert(CURAND_ROUTINE_HANDLER_PAIR(GenerateLogNormalDouble));
    mspHandlers->insert(CURAND_ROUTINE_HANDLER_PAIR(DestroyGenerator));
}