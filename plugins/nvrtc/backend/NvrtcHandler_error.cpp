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

#include "NvrtcHandler.h"

using namespace std;
using namespace log4cplus;

using gvirtus::communicators::Buffer;
using gvirtus::communicators::Result;

NVRTC_ROUTINE_HANDLER(GetErrorString) {
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GetErrorString"));
    nvrtcResult cs = in->Get<nvrtcResult>();
    const char *s = nvrtcGetErrorString(cs);
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try {
        out->AddString(s);
    } catch (const std::exception& e) {
        LOG4CPLUS_DEBUG(logger, LOG4CPLUS_TEXT("Exception: ") << e.what());
        return std::make_shared<Result>(NVRTC_ERROR_INTERNAL_ERROR);
    }
    LOG4CPLUS_DEBUG(logger, "nvrtcGetErrorString Executed");
    return std::make_shared<Result>(NVRTC_SUCCESS, out);
}