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
 * Written by: Theodoros Aslanidis <theodoros.aslanidis@ucdconnect.ie>,
 *             Department of Computer Science, University College Dublin
 *
 */

#include "NvmlHandler.h"

using namespace std;
using namespace log4cplus;

using gvirtus::communicators::Buffer;
using gvirtus::communicators::Result;

NVML_ROUTINE_HANDLER(InitWithFlags) {
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("InitWithFlags"));
    unsigned int flags = in->Get<unsigned int>();
    nvmlReturn_t cs = nvmlInitWithFlags(flags);
    LOG4CPLUS_DEBUG(logger, "nvmlInitWithFlags Executed");
    return std::make_shared<Result>(cs);
}

NVML_ROUTINE_HANDLER(Init) {
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Init"));
    nvmlReturn_t cs = nvmlInit();
    LOG4CPLUS_DEBUG(logger, "nvmlInit Executed");
    return std::make_shared<Result>(cs);
}

NVML_ROUTINE_HANDLER(Shutdown) {
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Shutdown"));
    nvmlReturn_t cs = nvmlShutdown();
    LOG4CPLUS_DEBUG(logger, "nvmlShutdown Executed");
    return std::make_shared<Result>(cs);
}