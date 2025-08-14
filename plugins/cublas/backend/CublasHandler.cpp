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
 * Written by: Raffaele Montella <raffaele.montella@uniparthenope.it>,
 *             Department of Science and Technologies
 * Edited by: Theodoros Aslanidis <theodoros.aslanidis@ucdconnect.ie>,
 *             School of Computer Science, University College Dublin
 */

#include "CublasHandler.h"

using gvirtus::communicators::Buffer;
using gvirtus::communicators::Result;

std::map<string, CublasHandler::CublasRoutineHandler>* CublasHandler::mspHandlers = NULL;

extern "C" std::shared_ptr<CublasHandler> create_t() { return std::make_shared<CublasHandler>(); }

CublasHandler::CublasHandler() {
    logger = Logger::getInstance(LOG4CPLUS_TEXT("CublasHandler"));
    Initialize();
}

CublasHandler::~CublasHandler() {}

bool CublasHandler::CanExecute(std::string routine) {
    return mspHandlers->find(routine) != mspHandlers->end();
}

std::shared_ptr<Result> CublasHandler::Execute(std::string routine,
                                               std::shared_ptr<Buffer> input_buffer) {
    LOG4CPLUS_DEBUG(logger, "Called " << routine);
    map<string, CublasHandler::CublasRoutineHandler>::iterator it;
    it = mspHandlers->find(routine);
    if (it == mspHandlers->end()) throw runtime_error("No handler for '" + routine + "' found!");
    try {
        return it->second(this, input_buffer);
    } catch (const std::exception& e) {
        LOG4CPLUS_DEBUG(logger, LOG4CPLUS_TEXT("Exception: ") << e.what());
    }
    return NULL;
}

void CublasHandler::Initialize() {
    if (mspHandlers != NULL) return;
    mspHandlers = new map<string, CublasHandler::CublasRoutineHandler>();

    /* CublasHandler Helper functions */
    mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(GetVersion_v2));
    mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Create_v2));
    mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Destroy_v2));
    mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(SetVector));
    mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(GetVector));
    mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(SetMatrix));
    mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(GetMatrix));
    mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(SetMathMode));
    mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(GetMathMode));
    mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(SetStream_v2));
    mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(GetPointerMode_v2));
    mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(SetPointerMode_v2));
    mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(SetWorkspace_v2));

    /* CublasHandler Level1 functions */
    mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Sdot_v2));
    mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Ddot_v2));
    mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Cdotu_v2));
    mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Cdotc_v2));
    mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Zdotu_v2));
    mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Zdotc_v2));

    mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Sscal_v2));
    mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Dscal_v2));
    mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Cscal_v2));
    mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Csscal_v2));
    mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Zscal_v2));
    mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Zdscal_v2));

    mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Saxpy_v2));
    mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Daxpy_v2));
    mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Caxpy_v2));
    mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Zaxpy_v2));

    mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Scopy_v2));
    mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Ccopy_v2));
    mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Dcopy_v2));
    mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Zcopy_v2));

    mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Sswap_v2));
    mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Dswap_v2));
    mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Cswap_v2));
    mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Zswap_v2));

    mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Isamax_v2));
    mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Idamax_v2));
    mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Icamax_v2));
    mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Izamax_v2));

    mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Sasum_v2));
    mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Dasum_v2));
    mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Scasum_v2));
    mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Dzasum_v2));

    mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Srot_v2));
    mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Drot_v2));
    mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Crot_v2));
    mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Csrot_v2));
    mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Zrot_v2));
    mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Zdrot_v2));

    mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Srotg_v2));
    mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Drotg_v2));
    mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Crotg_v2));
    mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Zrotg_v2));

    mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Srotm_v2));
    mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Drotm_v2));

    mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Srotmg_v2));
    mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Drotmg_v2));

    /* CublasHandler Level2 functions */
    mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Sgemv_v2));
    mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Dgemv_v2));
    mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Cgemv_v2));
    mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Zgemv_v2));

    mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Sgbmv_v2));
    mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Dgbmv_v2));
    mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Sgbmv_v2));
    mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Sgbmv_v2));

    mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Strmv_v2));
    mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Dtrmv_v2));
    mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Ctrmv_v2));
    mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Ztrmv_v2));

    mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Stbmv_v2));
    mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Dtbmv_v2));
    mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Ctbmv_v2));
    mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Ztbmv_v2));

    mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Stpmv_v2));
    mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Dtpmv_v2));
    mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Ctpmv_v2));
    mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Ztpmv_v2));

    mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Stpsv_v2));
    mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Dtpsv_v2));
    mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Ctpsv_v2));
    mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Ztpsv_v2));

    mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Stbsv_v2));
    mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Dtbsv_v2));
    mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Ctbsv_v2));
    mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Ztbsv_v2));

    mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Ssymv_v2));
    mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Dsymv_v2));
    mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Csymv_v2));
    mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Zsymv_v2));
    mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Chemv_v2));
    mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Zhemv_v2));

    mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Ssbmv_v2));
    mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Dsbmv_v2));
    mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Chbmv_v2));
    mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Zhbmv_v2));

    mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Sspmv_v2));
    mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Dspmv_v2));
    mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Chpmv_v2));
    mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Zhpmv_v2));

    mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Sger_v2));
    mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Dger_v2));
    mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Cgeru_v2));
    mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Cgerc_v2));
    mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Zgeru_v2));
    mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Zgerc_v2));

    mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Ssyr_v2));
    mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Dsyr_v2));
    mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Csyr_v2));
    mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Zsyr_v2));
    mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Cher_v2));
    mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Zher_v2));

    mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Sspr_v2));
    mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Dspr_v2));
    mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Chpr_v2));
    mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Zhpr_v2));

    mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Ssyr2_v2));
    mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Dsyr2_v2));
    mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Csyr2_v2));
    mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Zsyr2_v2));
    mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Cher2_v2));
    mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Zher2_v2));

    mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Sspr2_v2));
    mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Dspr2_v2));
    mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Chpr2_v2));
    mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Zhpr2_v2));

    /* CublasHandler Level3 functions */
    mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Sgemm_v2));
    mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Dgemm_v2));
    mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Cgemm_v2));
    mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Zgemm_v2));

    mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(SgemmBatched_v2));
    mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(DgemmBatched_v2));
    mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(CgemmBatched_v2));
    mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(ZgemmBatched_v2));

    mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Snrm2_v2));
    mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Dnrm2_v2));
    mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Scnrm2_v2));
    mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Dznrm2_v2));

    mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Ssyrk_v2));
    mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Dsyrk_v2));
    mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Csyrk_v2));
    mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Zsyrk_v2));
    mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Cherk_v2));
    mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Zherk_v2));

    mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Ssyr2k_v2));
    mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Dsyr2k_v2));
    mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Csyr2k_v2));
    mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Zsyr2k_v2));
    mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Cher2k_v2));
    mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Zher2k_v2));

    mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Ssymm_v2));
    mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Dsymm_v2));
    mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Csymm_v2));
    mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Zsymm_v2));
    mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Chemm_v2));
    mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Zhemm_v2));

    mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Strsm_v2));
    mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Dtrsm_v2));
    mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Ctrsm_v2));
    mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Ztrsm_v2));

    mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Strmm_v2));
    mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Dtrmm_v2));
    mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Ctrmm_v2));
    mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Ztrmm_v2));

    mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(SgemmStridedBatched));

    /* CublasHandler Lt functions */
    mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(LtMatmulAlgoGetHeuristic));
    // mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(LtMatrixLayoutSetAttribute));
    mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(LtMatmulDescSetAttribute));
    mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(LtMatrixLayoutCreate));
    mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(LtMatrixLayoutDestroy));
    mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(LtMatmulDescCreate));
    mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(LtMatmulDescDestroy));
    mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(LtMatmulPreferenceCreate));
    mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(LtMatmulPreferenceSetAttribute));
    mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(LtMatmulPreferenceDestroy));
    mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(LtMatmul));

    /* CublasHandler Extension functions */
    mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(GemmEx));
    mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(GemmStridedBatchedEx));
}
