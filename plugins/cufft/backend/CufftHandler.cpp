/*
 * gVirtuS -- A GPGPU transparent virtualization component.
 *
 * Copyright (C) 2009-2011  The University of Napoli Parthenope at Naples.
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
 *             Vincenzo Santopietro <vincenzo.santopietro@uniparthenope.it>,
 *             Department of Applied Science
 *
 * Edited By: Theodoros Aslanidis <theodoros.aslanidis@ucdconnect.ie>,
 *             School of Computer Science, University College Dublin
 */

#include "CufftHandler.h"

using namespace std;
using namespace log4cplus;

using gvirtus::communicators::Buffer;
using gvirtus::communicators::Result;

map<string, CufftHandler::CufftRoutineHandler>* CufftHandler::mspHandlers = NULL;

extern "C" std::shared_ptr<CufftHandler> create_t() { return std::make_shared<CufftHandler>(); }

CufftHandler::CufftHandler() {
    logger = Logger::getInstance(LOG4CPLUS_TEXT("CufftHandler"));
    Initialize();
}

CufftHandler::~CufftHandler() {}

bool CufftHandler::CanExecute(std::string routine) {
    return mspHandlers->find(routine) != mspHandlers->end();
}

std::shared_ptr<Result> CufftHandler::Execute(std::string routine,
                                              std::shared_ptr<Buffer> input_buffer) {
    LOG4CPLUS_DEBUG(logger, "Called " << routine);
    map<string, CufftHandler::CufftRoutineHandler>::iterator it;
    it = mspHandlers->find(routine);
    if (it == mspHandlers->end())
        throw std::runtime_error(std::string("No handler for '") + routine +
                                 std::string("' found!"));
    try {
        return it->second(this, input_buffer);
    } catch (const std::exception& e) {
        LOG4CPLUS_DEBUG(logger, LOG4CPLUS_TEXT("Exception: ") << e.what());
    }
    return NULL;
}

/*
 * cufftResult cufftPlan1d(cufftHandle *plan, int nx, cufftType type, int batch);
 * Creates a 1D FFT plan configuration for a specified signal size and data type.
 * The batch input parameter tells cuFFT how many 1D transforms to configure.
 */
CUFFT_ROUTINE_HANDLER(Plan1d) {
    cufftHandle* plan_adv = in->Assign<cufftHandle>();
    int nx = in->Get<int>();
    cufftType type = in->Get<cufftType>();
    int batch = in->Get<int>();

    cufftResult exit_code = cufftPlan1d(plan_adv, nx, type, batch);
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();

    try {
        out->Add(plan_adv);
    } catch (const std::exception& e) {
        LOG4CPLUS_DEBUG(pThis->GetLogger(), LOG4CPLUS_TEXT("Exception: ") << e.what());
        return std::make_shared<Result>(cudaErrorMemoryAllocation);  //???
    }
    LOG4CPLUS_DEBUG(pThis->GetLogger(), "Plan: " << *plan_adv);
    LOG4CPLUS_DEBUG(pThis->GetLogger(), "cufftPlan1d Executed");
    return std::make_shared<Result>(exit_code, out);
}

/*
 * cufftResult cufftPlan2d(cufftHandle *plan, int nx, int ny, cufftType type);
 * Creates a 2D FFT plan configuration according to specified signal sizes and data type.
 */
CUFFT_ROUTINE_HANDLER(Plan2d) {
    cufftHandle plan;  // = in->Assign<cufftHandle>();
    int nx = in->Get<int>();
    int ny = in->Get<int>();
    cufftType type = in->Get<cufftType>();

    cufftResult exit_code = cufftPlan2d(&plan, nx, ny, type);
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try {
        out->Add(&plan);
    } catch (const std::exception& e) {
        LOG4CPLUS_DEBUG(pThis->GetLogger(), LOG4CPLUS_TEXT("Exception: ") << e.what());
        return std::make_shared<Result>(cudaErrorMemoryAllocation);  //???
    }
    LOG4CPLUS_DEBUG(pThis->GetLogger(), "Plan: " << plan);
    LOG4CPLUS_DEBUG(pThis->GetLogger(), "cufftPlan2d Executed");
    return std::make_shared<Result>(exit_code, out);
}

/*
 * cufftResult cufftPlan3d(cufftHandle *plan, int nx, int ny, int nz, cufftType type);
 * Creates a 3D FFT plan configuration according to specified signal sizes and data type.
 * This function is the same as cufftPlan2d() except that it takes a third size parameter nz.
 */
CUFFT_ROUTINE_HANDLER(Plan3d) {
    try {
        cufftHandle* plan = in->Assign<cufftHandle>();
        int nx = in->Get<int>();
        int ny = in->Get<int>();
        int nz = in->Get<int>();
        cufftType type = in->Get<cufftType>();
        cufftResult ec = cufftPlan3d(plan, nx, ny, nz, type);
        std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
        out->Add(plan);
        return std::make_shared<Result>(ec, out);
    } catch (const std::exception& e) {
        LOG4CPLUS_DEBUG(pThis->GetLogger(), LOG4CPLUS_TEXT("Exception: ") << e.what());
        return std::make_shared<Result>(cudaErrorMemoryAllocation);  //???
    }
}

/*
 * Creates a FFT plan configuration of dimension rank, with sizes specified in the array n.
 * The batch input parameter tells cuFFT how many transforms to configure. With this function,
 * batched plans of 1, 2, or 3 dimensions may be created.
 */
CUFFT_ROUTINE_HANDLER(PlanMany) {
    cufftHandle* plan = in->Assign<cufftHandle>();
    int rank = in->Get<int>();
    int* n = in->Assign<int>();
    int* inembed = in->Assign<int>();
    int istride = in->Get<int>();
    int idist = in->Get<int>();

    int* onembed = in->Assign<int>();
    int ostride = in->Get<int>();
    int odist = in->Get<int>();

    cufftType type = in->Get<cufftType>();
    int batch = in->Get<int>();
    LOG4CPLUS_DEBUG(pThis->GetLogger(), "rank: " << rank << " n: " << *n << " idist: " << idist
                                                 << " ostride: " << ostride << " odist: " << odist
                                                 << " type:" << type << " batch: " << batch
                                                 << endl);
    try {
        cufftResult exit_code = cufftPlanMany(plan, rank, n, inembed, istride, idist, onembed,
                                              ostride, odist, type, batch);
        LOG4CPLUS_DEBUG(pThis->GetLogger(), "cufftPlanMany Executed");
        LOG4CPLUS_DEBUG(pThis->GetLogger(), "Plan: " << *plan);
        std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
        out->Add<cufftHandle>(plan);
        return std::make_shared<Result>(exit_code, out);
    } catch (const std::exception& e) {
        LOG4CPLUS_DEBUG(pThis->GetLogger(), LOG4CPLUS_TEXT("Exception: ") << e.what());
        return std::make_shared<Result>(cudaErrorMemoryAllocation);  //???
    }
}

CUFFT_ROUTINE_HANDLER(ExecC2R) {
    cufftHandle plan = in->Get<cufftHandle>();
    cufftReal* odata;
    cufftComplex* idata;

    idata = (in->GetFromMarshal<cufftComplex*>());

    try {
        odata = (in->GetFromMarshal<cufftReal*>());
    } catch (const std::exception& e) {
        LOG4CPLUS_DEBUG(pThis->GetLogger(), LOG4CPLUS_TEXT("Exception:") << e.what());
        odata = (cufftReal*)idata;
    }

    cufftResult exit_code = cufftExecC2R(plan, idata, odata);

    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try {
        out->AddMarshal(odata);
    } catch (const std::exception& e) {
        LOG4CPLUS_DEBUG(pThis->GetLogger(), LOG4CPLUS_TEXT("Exception: ") << e.what());
        return std::make_shared<Result>(cudaErrorMemoryAllocation);
    }
    LOG4CPLUS_DEBUG(pThis->GetLogger(), "cufftExecC2R Executed");
    return std::make_shared<Result>(exit_code, out);
}

CUFFT_ROUTINE_HANDLER(Create) {
    cufftHandle plan;
    cufftResult exit_code = cufftCreate(&plan);
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try {
        out->Add<cufftHandle>(&plan);
    } catch (const std::exception& e) {
        LOG4CPLUS_DEBUG(pThis->GetLogger(), LOG4CPLUS_TEXT("Exception: ") << e.what());
        return std::make_shared<Result>(cudaErrorMemoryAllocation);
    }
    LOG4CPLUS_DEBUG(pThis->GetLogger(), "cufftCreate Executed. Plan: " << plan);
    return std::make_shared<Result>(exit_code, out);
}

/*
 * cufftResult cufftDestroy(cufftHandle plan);
 * Frees all GPU resources associated with a cuFFT plan and destroys the internal plan data
 * structure. This function should be called once a plan is no longer needed, to avoid wasting GPU
 * memory.
 */
CUFFT_ROUTINE_HANDLER(Destroy) {
    cufftHandle plan = in->Get<cufftHandle>();
    cufftResult exit_code = cufftDestroy(plan);

    LOG4CPLUS_DEBUG(pThis->GetLogger(), "cufftDestroy Executed");
    return std::make_shared<Result>(exit_code);
}

CUFFT_ROUTINE_HANDLER(SetWorkArea) {
    cufftHandle plan = in->Get<cufftHandle>();
    void* workArea = in->GetFromMarshal<void*>();
    cufftResult exit_code = cufftSetWorkArea(plan, workArea);
    LOG4CPLUS_DEBUG(pThis->GetLogger(), "cufftSetWorkArea Executed");
    return std::make_shared<Result>(exit_code);
}

CUFFT_ROUTINE_HANDLER(SetAutoAllocation) {
    cufftHandle plan = in->Get<cufftHandle>();
    int autoAllocate = in->Get<int>();
    cufftResult exit_code = cufftSetAutoAllocation(plan, autoAllocate);
    LOG4CPLUS_DEBUG(pThis->GetLogger(), "cufftSetAutoAllocation Executed");
    return std::make_shared<Result>(exit_code);
}

CUFFT_ROUTINE_HANDLER(XtMakePlanMany) {
    cufftHandle plan = in->Get<cufftHandle>();
    int rank = in->Get<int>();
    long long int* n = in->Assign<long long int>();
    long long int* inembed = in->Assign<long long int>();
    long long int istride = in->Get<long long int>();
    long long int idist = in->Get<long long int>();
    cudaDataType inputtype = in->Get<cudaDataType>();

    long long int* onembed = in->Assign<long long int>();
    long long int ostride = in->Get<long long int>();
    long long int odist = in->Get<long long int>();
    cudaDataType outputtype = in->Get<cudaDataType>();

    long long int batch = in->Get<long long int>();
    size_t* workSize = in->Assign<size_t>();
    cudaDataType executiontype = in->Get<cudaDataType>();

    cufftResult exit_code =
        cufftXtMakePlanMany(plan, rank, n, inembed, istride, idist, inputtype, onembed, ostride,
                            odist, outputtype, batch, workSize, executiontype);
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try {
        // out->Add(n);
        // out->Add(inembed);
        // out->Add(onembed);
        out->Add(workSize);
    } catch (const std::exception& e) {
        LOG4CPLUS_DEBUG(pThis->GetLogger(), LOG4CPLUS_TEXT("Exception: ") << e.what());
        return std::make_shared<Result>(cudaErrorMemoryAllocation);
    }
    LOG4CPLUS_DEBUG(pThis->GetLogger(), "cufftXtMakePlanMany Executed");
    return std::make_shared<Result>(exit_code, out);
}

/*
 *cufftResult cufftExecC2C(cufftHandle plan, cufftComplex *idata, cufftComplex *odata, int
 *direction); cufftExecC2C() (cufftExecZ2Z()) executes a single-precision (double-precision)
 *complex-to-complex transform plan in the transform direction as specified by direction parameter.
 *cuFFT uses the GPU memory pointed to by the idata parameter as input data.
 *This function stores the Fourier coefficients in the odata array. If idata and odata are the same,
 *this method does an in-place transform.
 */
CUFFT_ROUTINE_HANDLER(ExecC2C) {
    cufftHandle plan = in->Get<cufftHandle>();
    cufftComplex *idata, *odata;

    idata = (in->GetFromMarshal<cufftComplex*>());

    try {
        odata = (in->GetFromMarshal<cufftComplex*>());
    } catch (const std::exception& e) {
        LOG4CPLUS_DEBUG(pThis->GetLogger(), LOG4CPLUS_TEXT("Exception:") << e.what());
        odata = idata;
    }

    int direction = in->Get<int>();

    cufftResult exit_code = cufftExecC2C(plan, idata, odata, direction);

    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try {
        out->AddMarshal(odata);
    } catch (const std::exception& e) {
        LOG4CPLUS_DEBUG(pThis->GetLogger(), LOG4CPLUS_TEXT("Exception:") << e.what());
        return std::make_shared<Result>(cudaErrorMemoryAllocation);
    }
    LOG4CPLUS_DEBUG(pThis->GetLogger(), "cufftExecC2C Executed");
    return std::make_shared<Result>(exit_code, out);
}

CUFFT_ROUTINE_HANDLER(ExecR2C) {
    cufftHandle plan = in->Get<cufftHandle>();
    cufftReal* idata;
    cufftComplex* odata;
    idata = (in->GetFromMarshal<cufftReal*>());
    try {
        odata = (in->GetFromMarshal<cufftComplex*>());
    } catch (const std::exception& e) {
        LOG4CPLUS_DEBUG(pThis->GetLogger(), LOG4CPLUS_TEXT("Exception:") << e.what());
    }

    cufftResult exit_code = cufftExecR2C(plan, idata, odata);

    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try {
        out->AddMarshal(odata);
    } catch (const std::exception& e) {
        LOG4CPLUS_DEBUG(pThis->GetLogger(), LOG4CPLUS_TEXT("Exception: ") << e.what());
        return std::make_shared<Result>(cudaErrorMemoryAllocation);
    }
    LOG4CPLUS_DEBUG(pThis->GetLogger(), "cufftExecR2C Executed");
    return std::make_shared<Result>(exit_code, out);
}

CUFFT_ROUTINE_HANDLER(ExecZ2Z) {
    cufftHandle plan = in->Get<cufftHandle>();
    cufftDoubleComplex *idata, *odata;
    idata = (in->GetFromMarshal<cufftDoubleComplex*>());

    try {
        odata = (in->GetFromMarshal<cufftDoubleComplex*>());
    } catch (const std::exception& e) {
        LOG4CPLUS_DEBUG(pThis->GetLogger(), LOG4CPLUS_TEXT("Exception:") << e.what());
        odata = idata;
    }
    int direction = in->Get<int>();
    cufftResult exit_code = cufftExecZ2Z(plan, idata, odata, direction);

    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try {
        out->AddMarshal(odata);
    } catch (const std::exception& e) {
        LOG4CPLUS_DEBUG(pThis->GetLogger(), LOG4CPLUS_TEXT("Exception: ") << e.what());
        return std::make_shared<Result>(cudaErrorMemoryAllocation);
    }
    LOG4CPLUS_DEBUG(pThis->GetLogger(), "cufftExecZ2Z Executed");
    return std::make_shared<Result>(exit_code, out);
}

CUFFT_ROUTINE_HANDLER(ExecD2Z) {
    cufftHandle plan = in->Get<cufftHandle>();
    cufftDoubleReal* idata;
    cufftDoubleComplex* odata;
    idata = (in->GetFromMarshal<cufftDoubleReal*>());

    try {
        odata = (in->GetFromMarshal<cufftDoubleComplex*>());
    } catch (const std::exception& e) {
        LOG4CPLUS_DEBUG(pThis->GetLogger(), LOG4CPLUS_TEXT("Exception:") << e.what());
        odata = (cufftDoubleComplex*)idata;
    }

    int direction = in->BackGet<int>();
    cufftResult exit_code = cufftExecD2Z(plan, idata, odata);

    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try {
        out->AddMarshal(odata);
    } catch (const std::exception& e) {
        LOG4CPLUS_DEBUG(pThis->GetLogger(), LOG4CPLUS_TEXT("Exception: ") << e.what());
        return std::make_shared<Result>(cudaErrorMemoryAllocation);
    }
    LOG4CPLUS_DEBUG(pThis->GetLogger(), "cufftExecD2Z Executed");
    return std::make_shared<Result>(exit_code, out);
}

CUFFT_ROUTINE_HANDLER(ExecZ2D) {
    cufftHandle plan = in->Get<cufftHandle>();
    cufftDoubleComplex* idata;
    cufftDoubleReal* odata;
    idata = (in->GetFromMarshal<cufftDoubleComplex*>());

    try {
        odata = (in->GetFromMarshal<cufftDoubleReal*>());
    } catch (const std::exception& e) {
        LOG4CPLUS_DEBUG(pThis->GetLogger(), LOG4CPLUS_TEXT("Exception:") << e.what());
        odata = (cufftDoubleReal*)idata;
    }
    cufftResult exit_code = cufftExecZ2D(plan, idata, odata);

    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try {
        out->AddMarshal(odata);
    } catch (const std::exception& e) {
        LOG4CPLUS_DEBUG(pThis->GetLogger(), LOG4CPLUS_TEXT("Exception: ") << e.what());
        return std::make_shared<Result>(cudaErrorMemoryAllocation);
    }
    LOG4CPLUS_DEBUG(pThis->GetLogger(), "cufftExecZ2D Executed");
    return std::make_shared<Result>(exit_code, out);
}

/*
 * cufftResult
    cufftXtSetGPUs(cufftHandle plan, int nGPUs, int *whichGPUs);
 *  cufftXtSetGPUs() indentifies which GPUs are to be used with the plan.
 * @brief As in the single GPU case cufftCreate() creates a plan and cufftMakePlan*() does the plan
 generation.This call will return an error if a non-default stream has been associated with the
 plan.
 * @param plan
 * @param nGPUs
 * @param *whichGPUs
 */
CUFFT_ROUTINE_HANDLER(XtSetGPUs) {
    cufftHandle plan = in->Get<cufftHandle>();
    int nGPUs = in->Get<int>();
    int* whichGPUs = in->Assign<int>(nGPUs);

    LOG4CPLUS_DEBUG(pThis->GetLogger(), "XtSetGPUs: nGPUs: " << nGPUs);
    LOG4CPLUS_DEBUG(pThis->GetLogger(), "XtSetGPUs: whichGPUs: " << *whichGPUs);

    cufftResult exit_code = cufftXtSetGPUs(plan, nGPUs, whichGPUs);

    LOG4CPLUS_DEBUG(pThis->GetLogger(), "cufftXtSetGPUs Executed");

    return std::make_shared<Result>(exit_code);
}

CUFFT_ROUTINE_HANDLER(Estimate1d) {
    int nx = in->Get<int>();
    cufftType type = in->Get<cufftType>();
    int batch = in->Get<int>();
    size_t* workSize = (in->Assign<size_t>());
    cufftResult exit_code = cufftEstimate1d(nx, type, batch, workSize);

    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try {
        out->Add(workSize);
    } catch (const std::exception& e) {
        LOG4CPLUS_DEBUG(pThis->GetLogger(), LOG4CPLUS_TEXT("Exception:") << e.what());
        return std::make_shared<Result>(cudaErrorMemoryAllocation);
    }
    LOG4CPLUS_DEBUG(pThis->GetLogger(), "cufftEstimate1d Executed");
    return std::make_shared<Result>(exit_code, out);
}

CUFFT_ROUTINE_HANDLER(Estimate2d) {
    int nx = in->Get<int>();
    int ny = in->Get<int>();
    cufftType type = in->Get<cufftType>();
    size_t* workSize = (in->Assign<size_t>());
    cufftResult exit_code = cufftEstimate2d(nx, ny, type, workSize);

    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try {
        out->Add(workSize);
    } catch (const std::exception& e) {
        LOG4CPLUS_DEBUG(pThis->GetLogger(), LOG4CPLUS_TEXT("Exception:") << e.what());
        return std::make_shared<Result>(cudaErrorMemoryAllocation);
    }
    LOG4CPLUS_DEBUG(pThis->GetLogger(), "cufftEstimate2d Executed");
    return std::make_shared<Result>(exit_code, out);
}

CUFFT_ROUTINE_HANDLER(Estimate3d) {
    int nx = in->Get<int>();
    int ny = in->Get<int>();
    int nz = in->Get<int>();
    cufftType type = in->Get<cufftType>();
    size_t* workSize = (in->Assign<size_t>());
    cufftResult exit_code = cufftEstimate3d(nx, ny, nz, type, workSize);

    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try {
        out->Add(workSize);
    } catch (const std::exception& e) {
        LOG4CPLUS_DEBUG(pThis->GetLogger(), LOG4CPLUS_TEXT("Exception:") << e.what());
        return std::make_shared<Result>(cudaErrorMemoryAllocation);
    }
    LOG4CPLUS_DEBUG(pThis->GetLogger(), "cufftEstimate3d Executed");
    return std::make_shared<Result>(exit_code, out);
}

CUFFT_ROUTINE_HANDLER(EstimateMany) {
    int rank = in->Get<int>();
    int* n = in->Assign<int>();
    int* inembed = in->Assign<int>();
    int istride = in->Get<int>();
    int idist = in->Get<int>();

    int* onembed = in->Assign<int>();
    int ostride = in->Get<int>();
    int odist = in->Get<int>();

    cufftType type = in->Get<cufftType>();
    int batch = in->Get<int>();
    size_t* workSize = (in->Assign<size_t>());

    cufftResult exit_code = cufftEstimateMany(rank, n, inembed, istride, idist, onembed, ostride,
                                              odist, type, batch, workSize);

    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try {
        out->Add(workSize);
    } catch (const std::exception& e) {
        LOG4CPLUS_DEBUG(pThis->GetLogger(), LOG4CPLUS_TEXT("Exception:") << e.what());
        return std::make_shared<Result>(cudaErrorMemoryAllocation);
    }
    LOG4CPLUS_DEBUG(pThis->GetLogger(), "cufftEstimateMany Executed");
    return std::make_shared<Result>(exit_code, out);
}

/*
 * cufftResult
    cufftMakePlan1d(cufftHandle plan, int nx, cufftType type, int batch,
        size_t *workSize);
 * @brief Following a call to cufftCreate() makes a 1D FFT plan configuration for a specified signal
 size and data type. The batch input parameter tells cuFFT how many 1D transforms to configure.
 * @param plan
 * @param nx
 * @param type
 * @param batch
 * @param *workSize
 */
CUFFT_ROUTINE_HANDLER(MakePlan1d) {
    cufftHandle plan = in->Get<cufftHandle>();
    int nx = in->Get<int>();
    cufftType type = in->Get<cufftType>();
    int batch = in->Get<int>();
    size_t* workSize = (in->Assign<size_t>());
    cufftResult exit_code = cufftMakePlan1d(plan, nx, type, batch, workSize);

    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try {
        out->Add(workSize);
    } catch (const std::exception& e) {
        LOG4CPLUS_DEBUG(pThis->GetLogger(), LOG4CPLUS_TEXT("Exception:") << e.what());
        return std::make_shared<Result>(cudaErrorMemoryAllocation);
    }
    LOG4CPLUS_DEBUG(pThis->GetLogger(), "cufftMakePlan1d Executed");
    return std::make_shared<Result>(exit_code, out);
}

CUFFT_ROUTINE_HANDLER(MakePlan2d) {
    cufftHandle plan = in->Get<cufftHandle>();
    int nx = in->Get<int>();
    int ny = in->Get<int>();
    cufftType type = in->Get<cufftType>();
    size_t* workSize = (in->Assign<size_t>());
    cufftResult exit_code = cufftMakePlan2d(plan, nx, ny, type, workSize);

    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try {
        out->Add(workSize);
    } catch (const std::exception& e) {
        LOG4CPLUS_DEBUG(pThis->GetLogger(), LOG4CPLUS_TEXT("Exception:") << e.what());
        return std::make_shared<Result>(cudaErrorMemoryAllocation);
    }
    LOG4CPLUS_DEBUG(pThis->GetLogger(), "cufftMakePlan2d Executed");
    return std::make_shared<Result>(exit_code, out);
}

CUFFT_ROUTINE_HANDLER(MakePlan3d) {
    cufftHandle plan = in->Get<cufftHandle>();
    int nx = in->Get<int>();
    int ny = in->Get<int>();
    int nz = in->Get<int>();
    cufftType type = in->Get<cufftType>();
    size_t* workSize = (in->Assign<size_t>());
    cufftResult exit_code = cufftMakePlan3d(plan, nx, ny, nz, type, workSize);

    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try {
        out->Add(workSize);
    } catch (const std::exception& e) {
        LOG4CPLUS_DEBUG(pThis->GetLogger(), LOG4CPLUS_TEXT("Exception:") << e.what());
        return std::make_shared<Result>(cudaErrorMemoryAllocation);
    }
    LOG4CPLUS_DEBUG(pThis->GetLogger(), "cufftMakePlan3d Executed");
    return std::make_shared<Result>(exit_code, out);
}

CUFFT_ROUTINE_HANDLER(MakePlanMany) {
    cufftHandle plan = in->Get<cufftHandle>();
    int rank = in->Get<int>();
    int* n = in->Assign<int>();
    int* inembed = in->Assign<int>();
    int istride = in->Get<int>();
    int idist = in->Get<int>();

    int* onembed = in->Assign<int>();
    int ostride = in->Get<int>();
    int odist = in->Get<int>();

    cufftType type = in->Get<cufftType>();
    int batch = in->Get<int>();
    size_t* workSize = (in->Assign<size_t>());

    cufftResult exit_code = cufftMakePlanMany(plan, rank, n, inembed, istride, idist, onembed,
                                              ostride, odist, type, batch, workSize);
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try {
        out->Add(workSize);
    } catch (const std::exception& e) {
        LOG4CPLUS_DEBUG(pThis->GetLogger(), LOG4CPLUS_TEXT("Exception:") << e.what());
        return std::make_shared<Result>(cudaErrorMemoryAllocation);
    }
    LOG4CPLUS_DEBUG(pThis->GetLogger(), "cufftMakePlanMany Executed");
    return std::make_shared<Result>(exit_code, out);
}

CUFFT_ROUTINE_HANDLER(MakePlanMany64) {
    cufftHandle plan = in->Get<cufftHandle>();
    int rank = in->Get<int>();
    long long int* n = in->Assign<long long int>();
    long long int* inembed = in->Assign<long long int>();
    long long int istride = in->Get<long long int>();
    long long int idist = in->Get<long long int>();

    long long int* onembed = in->Assign<long long int>();
    long long int ostride = in->Get<long long int>();
    long long int odist = in->Get<long long int>();

    cufftType type = in->Get<cufftType>();
    long long int batch = in->Get<long long int>();
    size_t* workSize = in->Assign<size_t>();

    cufftResult exit_code = cufftMakePlanMany64(plan, rank, n, inembed, istride, idist, onembed,
                                                ostride, odist, type, batch, workSize);
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try {
        out->Add(workSize);
    } catch (const std::exception& e) {
        LOG4CPLUS_DEBUG(pThis->GetLogger(), LOG4CPLUS_TEXT("Exception:") << e.what());
        return std::make_shared<Result>(cudaErrorMemoryAllocation);
    }
    LOG4CPLUS_DEBUG(pThis->GetLogger(), "cufftMakePlanMany64 Executed");
    return std::make_shared<Result>(exit_code, out);
}

CUFFT_ROUTINE_HANDLER(GetSize1d) {
    cufftHandle handle = in->Get<cufftHandle>();
    int nx = in->Get<int>();
    cufftType type = in->Get<cufftType>();
    int batch = in->Get<int>();
    size_t* workSize = (in->Assign<size_t>());
    cufftResult exit_code = cufftGetSize1d(handle, nx, type, batch, workSize);

    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try {
        out->Add(workSize);
    } catch (const std::exception& e) {
        LOG4CPLUS_DEBUG(pThis->GetLogger(), LOG4CPLUS_TEXT("Exception:") << e.what());
        return std::make_shared<Result>(cudaErrorMemoryAllocation);
    }
    LOG4CPLUS_DEBUG(pThis->GetLogger(), "cufftGetSize1d Executed");
    return std::make_shared<Result>(exit_code, out);
}

CUFFT_ROUTINE_HANDLER(GetSize2d) {
    cufftHandle handle = in->Get<cufftHandle>();
    int nx = in->Get<int>();
    int ny = in->Get<int>();
    cufftType type = in->Get<cufftType>();
    size_t* workSize = (in->Assign<size_t>());
    cufftResult exit_code = cufftGetSize2d(handle, nx, ny, type, workSize);

    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try {
        out->Add(workSize);
    } catch (const std::exception& e) {
        LOG4CPLUS_DEBUG(pThis->GetLogger(), LOG4CPLUS_TEXT("Exception:") << e.what());
        return std::make_shared<Result>(cudaErrorMemoryAllocation);
    }
    LOG4CPLUS_DEBUG(pThis->GetLogger(), "cufftGetSize2d Executed");
    return std::make_shared<Result>(exit_code, out);
}

CUFFT_ROUTINE_HANDLER(GetSize3d) {
    cufftHandle handle = in->Get<cufftHandle>();
    int nx = in->Get<int>();
    int ny = in->Get<int>();
    int nz = in->Get<int>();
    cufftType type = in->Get<cufftType>();
    size_t* workSize = (in->Assign<size_t>());
    cufftResult exit_code = cufftGetSize3d(handle, nx, ny, nz, type, workSize);

    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try {
        out->Add(workSize);
    } catch (const std::exception& e) {
        LOG4CPLUS_DEBUG(pThis->GetLogger(), LOG4CPLUS_TEXT("Exception:") << e.what());
        return std::make_shared<Result>(cudaErrorMemoryAllocation);
    }
    LOG4CPLUS_DEBUG(pThis->GetLogger(), "cufftGetSize3d executed");
    return std::make_shared<Result>(exit_code, out);
}

CUFFT_ROUTINE_HANDLER(GetSizeMany) {
    cufftHandle handle = in->Get<cufftHandle>();
    int rank = in->Get<int>();
    int* n = in->Assign<int>();
    int* inembed = in->Assign<int>();
    int istride = in->Get<int>();
    int idist = in->Get<int>();

    int* onembed = in->Assign<int>();
    int ostride = in->Get<int>();
    int odist = in->Get<int>();

    cufftType type = in->Get<cufftType>();
    int batch = in->Get<int>();
    size_t* workSize = (in->Assign<size_t>());

    cufftResult exit_code = cufftGetSizeMany(handle, rank, n, inembed, istride, idist, onembed,
                                             ostride, odist, type, batch, workSize);
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try {
        out->Add(workSize);
    } catch (const std::exception& e) {
        LOG4CPLUS_DEBUG(pThis->GetLogger(), LOG4CPLUS_TEXT("Exception: ") << e.what());
        return std::make_shared<Result>(cudaErrorMemoryAllocation);
    }
    LOG4CPLUS_DEBUG(pThis->GetLogger(), "cufftGetSizeMany executed");
    return std::make_shared<Result>(exit_code, out);
}

CUFFT_ROUTINE_HANDLER(GetSizeMany64) {
    cufftHandle plan = in->Get<cufftHandle>();
    int rank = in->Get<int>();
    long long int* n = in->Assign<long long int>();
    long long int* inembed = in->Assign<long long int>();
    long long int istride = in->Get<long long int>();
    long long int idist = in->Get<long long int>();

    long long int* onembed = in->Assign<long long int>();
    long long int ostride = in->Get<long long int>();
    long long int odist = in->Get<long long int>();

    cufftType type = in->Get<cufftType>();
    long long int batch = in->Get<long long int>();
    size_t* workSize = in->Assign<size_t>();

    cufftResult exit_code = cufftGetSizeMany64(plan, rank, n, inembed, istride, idist, onembed,
                                               ostride, odist, type, batch, workSize);
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try {
        out->Add(workSize);
    } catch (const std::exception& e) {
        LOG4CPLUS_DEBUG(pThis->GetLogger(), LOG4CPLUS_TEXT("Exception: ") << e.what());
        return std::make_shared<Result>(cudaErrorMemoryAllocation);
    }
    LOG4CPLUS_DEBUG(pThis->GetLogger(), "cufftGetSizeMany64 executed");
    return std::make_shared<Result>(exit_code, out);
}

CUFFT_ROUTINE_HANDLER(GetSize) {
    cufftHandle handle = in->Get<cufftHandle>();
    size_t* workSize = in->Assign<size_t>();
    cufftResult exit_code = cufftGetSize(handle, workSize);
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try {
        out->Add(workSize);
    } catch (const std::exception& e) {
        LOG4CPLUS_DEBUG(pThis->GetLogger(), LOG4CPLUS_TEXT("Exception: ") << e.what());
        return std::make_shared<Result>(cudaErrorMemoryAllocation);
    }
    LOG4CPLUS_DEBUG(pThis->GetLogger(), "cufftGetSize executed");
    return std::make_shared<Result>(exit_code, out);
}

CUFFT_ROUTINE_HANDLER(SetStream) {
    cufftHandle plan = in->Get<cufftHandle>();
    cudaStream_t stream = in->GetFromMarshal<cudaStream_t>();

    cufftResult exit_code = cufftSetStream(plan, stream);

    LOG4CPLUS_DEBUG(pThis->GetLogger(), "cufftSetStream executed with plan");
    return std::make_shared<Result>(exit_code);
}

CUFFT_ROUTINE_HANDLER(GetProperty) {
    libraryPropertyType type = in->Get<libraryPropertyType>();
    int value;
    cufftResult exit_code = cufftGetProperty(type, &value);
    LOG4CPLUS_DEBUG(pThis->GetLogger(), "cufftGetProperty executed");
    if (exit_code != CUFFT_SUCCESS) {
        return std::make_shared<Result>(exit_code);
    }
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    out->Add(value);
    return std::make_shared<Result>(exit_code, out);
}

CUFFT_ROUTINE_HANDLER(XtMalloc) {
    cufftHandle plan = in->Get<cufftHandle>();
    cufftXtSubFormat format = in->Get<cufftXtSubFormat>();

    cudaLibXtDesc* data = nullptr;
    cufftResult exit_code = cufftXtMalloc(plan, &data, format);

    LOG4CPLUS_DEBUG(pThis->GetLogger(), "cufftXtMalloc executed with");

    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    out->AddMarshal(data);  // add the opaque descriptor pointer
    return std::make_shared<Result>(exit_code, out);
}

/*
 * cufftResult cufftXtMalloc(cufftHandle plan, cudaLibXtDesc **descriptor,
        cufftXtSubFormat format);
 * @param plan
 * @param descriptor
 * @param format
 */
// original gvirtus code
// CUFFT_ROUTINE_HANDLER(XtMalloc){
//
//     cufftHandle plan = in->Get<cufftHandle>();
//     cudaLibXtDesc ** desc = in->Assign<cudaLibXtDesc*>();
//     cufftXtSubFormat format = in->Get<cufftXtSubFormat>();

//     cufftResult exit_code = cufftXtMalloc(plan,desc,format);
//     std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
//     // Buffer * out = new Buffer();
//     try{
//         out->Add(desc);
//     } catch (const std::exception& e){
//         LOG4CPLUS_DEBUG(pThis->GetLogger(), LOG4CPLUS_TEXT("Exception: ") << e.what());
//         return std::make_shared<Result>(cudaErrorMemoryAllocation);
//     }
//     return std::make_shared<Result>(exit_code,out);
// }

/*
 * cufftResult cufftXtMemcpy(cufftHandle plan, void *dstPointer, void *srcPointer, cufftXtCopyType
 * type);
 * @param plan
 * @param *dstPointer
 * @param *srcPointer
 * @param type
 */
// original gvirtus code
CUFFT_ROUTINE_HANDLER(XtMemcpy) {
    cufftHandle plan = in->Get<cufftHandle>();
    void* dstPointer = NULL;

    void* srcPointer = NULL;
    cufftXtCopyType type = in->BackGet<cufftXtCopyType>();
    cufftResult exit_code;
    std::shared_ptr<Buffer> out;
    try {
        switch (type) {
            case CUFFT_COPY_HOST_TO_DEVICE:
                dstPointer = in->GetFromMarshal<void*>();
                srcPointer = (in->Assign<void>());
                exit_code = cufftXtMemcpy(plan, dstPointer, srcPointer, CUFFT_COPY_HOST_TO_DEVICE);
                // out->AddMarshal(dstPointer);
                break;
            // TODO: IMPLEMENT THE FOLLOWING CASES
            // case CUFFT_COPY_DEVICE_TO_DEVICE:

            //     dstPointer = in->GetFromMarshal<void*>();
            //     srcPointer = in->GetFromMarshal<void*>();
            //     exit_code = cufftXtMemcpy(plan,dstPointer,srcPointer,type);
            //     out = new Buffer();
            //     out->AddMarshal(dstPointer);

            //     break;
            // case CUFFT_COPY_DEVICE_TO_HOST:
            //     dstPointer = in->Assign<char>();
            //     srcPointer = in->GetFromMarshal<void*>();
            //     exit_code = cufftXtMemcpy(plan,dstPointer,srcPointer,type);
            //     out = new Buffer();
            //     out->Add(dstPointer);
            //     break;
            default:
                break;
        }
    } catch (const std::exception& e) {
        LOG4CPLUS_DEBUG(pThis->GetLogger(), LOG4CPLUS_TEXT("Exception: ") << e.what());
        return std::make_shared<Result>(cudaErrorMemoryAllocation);
    }
    LOG4CPLUS_DEBUG(pThis->GetLogger(), "cufftXtMemcpy Executed");
    return std::make_shared<Result>(exit_code, out);
}

// tgasla attempt to implement XtMemcpy
// CUFFT_ROUTINE_HANDLER(XtMemcpy) {
//
//     // 1) Read the plan handle
//     cufftHandle plan = in->Get<cufftHandle>();

//     // 2) Read the copy type first (important for correct unmarshaling)
//     cufftXtCopyType type = in->Get<cufftXtCopyType>();

//     void* dstPointer;
//     void* srcPointer;

//     // 3) Retrieve pointers in the correct way depending on type
//     switch (type) {
//         case CUFFT_COPY_HOST_TO_DEVICE:
//             dstPointer = in->Get<void*>();               // Device pointer
//             srcPointer = in->GetFromMarshal<void*>();    // Host pointer
//             break;
//         case CUFFT_COPY_DEVICE_TO_HOST:
//             dstPointer = in->GetFromMarshal<void*>();    // Host pointer
//             srcPointer = in->Get<void*>();               // Device pointer
//             break;
//         case CUFFT_COPY_DEVICE_TO_DEVICE:
//             dstPointer = in->Get<void*>();               // Device pointer
//             srcPointer = in->Get<void*>();               // Device pointer
//             break;
//         default:
//             return std::make_shared<Result>(CUFFT_INVALID_VALUE);
//     }

//     cufftResult exit_code;
//     try {
//         exit_code = cufftXtMemcpy(plan, dstPointer, srcPointer, type);
//     } catch (const std::exception &e) {
//         LOG4CPLUS_DEBUG(pThis->GetLogger(), LOG4CPLUS_TEXT("Exception: ") << e.what());
//         return std::make_shared<Result>(cudaErrorMemoryAllocation);
//     }

//     if (exit_code != CUFFT_SUCCESS) {
//         std::cerr << "[XtMemcpy] cufftXtMemcpy failed with code " << exit_code << std::endl;
//     }

//     return std::make_shared<Result>(exit_code);
// }

/*Da testare*/
CUFFT_ROUTINE_HANDLER(XtExecDescriptorC2C) {
    cufftHandle plan = in->Get<cufftHandle>();
    cudaLibXtDesc* input = in->GetFromMarshal<cudaLibXtDesc*>();
    cudaLibXtDesc* output = in->GetFromMarshal<cudaLibXtDesc*>();
    int direction = in->Get<int>();
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    cufftResult exit_code;
    try {
        exit_code = cufftXtExecDescriptorC2C(plan, input, output, direction);
        out->AddMarshal(output);
    } catch (const std::exception& e) {
        LOG4CPLUS_DEBUG(pThis->GetLogger(), LOG4CPLUS_TEXT("Exception: ") << e.what());
        return std::make_shared<Result>(cudaErrorMemoryAllocation);
    }
    LOG4CPLUS_DEBUG(pThis->GetLogger(), "cufftXtExecDescriptorC2C Executed");
    return std::make_shared<Result>(exit_code, out);
}

/*Da testare*/
CUFFT_ROUTINE_HANDLER(XtFree) {
    cudaLibXtDesc* descriptor = in->GetFromMarshal<cudaLibXtDesc*>();
    cufftResult exit_code = cufftXtFree(descriptor);
    LOG4CPLUS_DEBUG(pThis->GetLogger(), "cufftXtFree Executed");
    return std::make_shared<Result>(exit_code);
}

/* -- FUNCTION NOT SUPPORTED IN GVIRTUS -- */
CUFFT_ROUTINE_HANDLER(XtSetCallback) { return std::make_shared<Result>(CUFFT_NOT_IMPLEMENTED); }

CUFFT_ROUTINE_HANDLER(GetVersion) {
    int version;

    cufftResult exit_code = cufftGetVersion(&version);

    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try {
        out->Add(version);
    } catch (const std::exception& e) {
        LOG4CPLUS_DEBUG(pThis->GetLogger(), LOG4CPLUS_TEXT("Exception: ") << e.what());
        return std::make_shared<Result>(cudaErrorMemoryAllocation);
    }

    LOG4CPLUS_DEBUG(pThis->GetLogger(), "cufftGetVersion: " << version);
    LOG4CPLUS_DEBUG(pThis->GetLogger(), "cufftGetVersion Executed");

    return std::make_shared<Result>(exit_code, out);
}

void CufftHandler::Initialize() {
    if (mspHandlers != NULL) return;
    mspHandlers = new map<string, CufftHandler::CufftRoutineHandler>();
    /* - Plan - */
    mspHandlers->insert(CUFFT_ROUTINE_HANDLER_PAIR(Plan1d));
    mspHandlers->insert(CUFFT_ROUTINE_HANDLER_PAIR(Plan2d));
    mspHandlers->insert(CUFFT_ROUTINE_HANDLER_PAIR(Plan3d));
    mspHandlers->insert(CUFFT_ROUTINE_HANDLER_PAIR(PlanMany));
    /* - Estimate - */
    mspHandlers->insert(CUFFT_ROUTINE_HANDLER_PAIR(Estimate1d));
    mspHandlers->insert(CUFFT_ROUTINE_HANDLER_PAIR(Estimate2d));
    mspHandlers->insert(CUFFT_ROUTINE_HANDLER_PAIR(Estimate3d));
    mspHandlers->insert(CUFFT_ROUTINE_HANDLER_PAIR(EstimateMany));
    /* - MakePlan - */
    mspHandlers->insert(CUFFT_ROUTINE_HANDLER_PAIR(MakePlan1d));
    mspHandlers->insert(CUFFT_ROUTINE_HANDLER_PAIR(MakePlan2d));
    mspHandlers->insert(CUFFT_ROUTINE_HANDLER_PAIR(MakePlan3d));
    mspHandlers->insert(CUFFT_ROUTINE_HANDLER_PAIR(MakePlanMany));
    mspHandlers->insert(CUFFT_ROUTINE_HANDLER_PAIR(MakePlanMany64));
    /* - GetSize - */
    mspHandlers->insert(CUFFT_ROUTINE_HANDLER_PAIR(GetSize1d));
    mspHandlers->insert(CUFFT_ROUTINE_HANDLER_PAIR(GetSize2d));
    mspHandlers->insert(CUFFT_ROUTINE_HANDLER_PAIR(GetSize3d));
    mspHandlers->insert(CUFFT_ROUTINE_HANDLER_PAIR(GetSizeMany));
    mspHandlers->insert(CUFFT_ROUTINE_HANDLER_PAIR(GetSizeMany64));
    mspHandlers->insert(CUFFT_ROUTINE_HANDLER_PAIR(GetSize));
    /* - Estimate - */
    mspHandlers->insert(CUFFT_ROUTINE_HANDLER_PAIR(SetWorkArea));
    mspHandlers->insert(CUFFT_ROUTINE_HANDLER_PAIR(SetAutoAllocation));
    mspHandlers->insert(CUFFT_ROUTINE_HANDLER_PAIR(GetVersion));
    mspHandlers->insert(CUFFT_ROUTINE_HANDLER_PAIR(SetStream));
    mspHandlers->insert(CUFFT_ROUTINE_HANDLER_PAIR(GetProperty));
    /* - Create/Destroy - */
    mspHandlers->insert(CUFFT_ROUTINE_HANDLER_PAIR(Create));
    mspHandlers->insert(CUFFT_ROUTINE_HANDLER_PAIR(Destroy));
    /* - Exec - */
    mspHandlers->insert(CUFFT_ROUTINE_HANDLER_PAIR(ExecC2C));
    mspHandlers->insert(CUFFT_ROUTINE_HANDLER_PAIR(ExecR2C));
    mspHandlers->insert(CUFFT_ROUTINE_HANDLER_PAIR(ExecC2R));
    mspHandlers->insert(CUFFT_ROUTINE_HANDLER_PAIR(ExecZ2Z));
    mspHandlers->insert(CUFFT_ROUTINE_HANDLER_PAIR(ExecD2Z));
    mspHandlers->insert(CUFFT_ROUTINE_HANDLER_PAIR(ExecZ2D));
    /* -- CufftX -- */
    mspHandlers->insert(CUFFT_ROUTINE_HANDLER_PAIR(XtMakePlanMany));
    mspHandlers->insert(CUFFT_ROUTINE_HANDLER_PAIR(XtSetGPUs));
    mspHandlers->insert(CUFFT_ROUTINE_HANDLER_PAIR(XtExecDescriptorC2C));
    mspHandlers->insert(CUFFT_ROUTINE_HANDLER_PAIR(XtSetCallback));
    /* - Memory Management - */
    mspHandlers->insert(CUFFT_ROUTINE_HANDLER_PAIR(XtMalloc));
    mspHandlers->insert(CUFFT_ROUTINE_HANDLER_PAIR(XtMemcpy));
    mspHandlers->insert(CUFFT_ROUTINE_HANDLER_PAIR(XtFree));
}
