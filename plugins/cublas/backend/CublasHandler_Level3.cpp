#include "CublasHandler.h"

using gvirtus::communicators::Buffer;
using gvirtus::communicators::Result;

CUBLAS_ROUTINE_HANDLER(Sgemm_v2) {
    // TODO: For now, it only works if alpha, beta are host pointers.
    // If they are device pointers, we need to add a check in the backend to handle them correctly.
    // If the function cublasSetPointerMode is called with CUBLAS_POINTER_MODE_DEVICE,
    // then alpha and beta should be device pointers, otherwise they should be host pointers.
    // We need to keep track of the pointer mode in the backend and handle it accordingly.
    Logger logger=Logger::getInstance(LOG4CPLUS_TEXT("Sgemm"));
    
    cublasHandle_t handle = in->Get<cublasHandle_t>();
    cublasOperation_t transa = in->Get<cublasOperation_t>();
    cublasOperation_t transb = in->Get<cublasOperation_t>();
    int m  = in->Get<int>();
    int n  = in->Get<int>();
    int k  = in->Get<int>();
    const float * alpha = in->Assign<float>();
    float * A = in->GetFromMarshal<float*>();
    int lda = in->Get<int>();
    float * B = in->GetFromMarshal<float*>();
    int ldb = in->Get<int>();
    const float * beta = in->Assign<float>();
    float * C = in->GetFromMarshal<float*>();
    int ldc = in->Get<int>();
    cublasStatus_t cs = cublasSgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    LOG4CPLUS_DEBUG(logger, "cublasSgemm_v2 Executed");
    return std::make_shared<Result>(cs);
}

CUBLAS_ROUTINE_HANDLER(SgemmBatched_v2){
    Logger logger=Logger::getInstance(LOG4CPLUS_TEXT("SgemmBatched"));
    
    cublasHandle_t handle;
    handle = (cublasHandle_t) in->Get<long long int>();
    
    cublasOperation_t transa = in->Get<cublasOperation_t>();
    cublasOperation_t transb = in->Get<cublasOperation_t>();
    int m  = in->Get<int>();
    int n  = in->Get<int>();
    int k  = in->Get<int>();
    const float * alpha = in->Assign<float>();
    
    const float ** A = (const float **) in->GetFromMarshal<float**>();
    int lda = in->Get<int>();
    const float ** B = (const float **) in->GetFromMarshal<float**>();
    int ldb = in->Get<int>();
    const float * beta = in->Assign<float>();
    float ** C = in->GetFromMarshal<float**>();
    int ldc = in->Get<int>();
    int batchSize = in->Get<int>();
    
    cublasStatus_t cs;
        std::shared_ptr<Buffer> out = std::make_shared<Buffer>();

    try{
        cs = cublasSgemmBatched(handle,transa,transb,m,n,k,alpha,A,lda,B,ldb,beta,C,ldc,batchSize);
        out->AddMarshal<float **>(C);
    } catch (const std::exception& e) {
        LOG4CPLUS_DEBUG(logger, LOG4CPLUS_TEXT("Exception: ") << e.what());
        return std::make_shared<Result>(cudaErrorMemoryAllocation);
    }
    LOG4CPLUS_DEBUG(logger, "cublasSgemmBatched_v2 Executed");
    return std::make_shared<Result>(cs,out);
}

CUBLAS_ROUTINE_HANDLER(Dgemm_v2) {
    Logger logger=Logger::getInstance(LOG4CPLUS_TEXT("Dgemm"));
    
    cublasHandle_t handle;
    handle = in->Get<cublasHandle_t>();
    
    cublasOperation_t transa = in->Get<cublasOperation_t>();
    cublasOperation_t transb = in->Get<cublasOperation_t>();
    int m  = in->Get<int>();
    int n  = in->Get<int>();
    int k  = in->Get<int>();
    const double * alpha = in->Assign<double>();
    double * A = in->GetFromMarshal<double*>();
    int lda = in->Get<int>();
    double * B = in->GetFromMarshal<double*>();
    int ldb = in->Get<int>();
    const double * beta = in->Assign<double>();
    double * C = in->GetFromMarshal<double*>();
    int ldc = in->Get<int>();
    cublasStatus_t cs = cublasDgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    LOG4CPLUS_DEBUG(logger, "cublasDgemm_v2 Executed");
    return std::make_shared<Result>(cs);
}

CUBLAS_ROUTINE_HANDLER(DgemmBatched_v2){
    Logger logger=Logger::getInstance(LOG4CPLUS_TEXT("DgemmBatched"));
    
    cublasHandle_t handle;
    handle = (cublasHandle_t) in->Get<long long int>();
    
    cublasOperation_t transa = in->Get<cublasOperation_t>();
    cublasOperation_t transb = in->Get<cublasOperation_t>();
    int m  = in->Get<int>();
    int n  = in->Get<int>();
    int k  = in->Get<int>();
    const double * alpha = in->Assign<double>();
    
    const double ** A = (const double **) in->GetFromMarshal<double**>();
    int lda = in->Get<int>();
    const double ** B = (const double **) in->GetFromMarshal<double**>();
    int ldb = in->Get<int>();
    const double * beta = in->Assign<double>();
    double ** C = in->GetFromMarshal<double**>();
    int ldc = in->Get<int>();
    int batchSize = in->Get<int>();
    
    cublasStatus_t cs;
        std::shared_ptr<Buffer> out = std::make_shared<Buffer>();

    try{
        cs = cublasDgemmBatched(handle,transa,transb,m,n,k,alpha,A,lda,B,ldb,beta,C,ldc,batchSize);
        out->AddMarshal<double **>(C);
    } catch (const std::exception& e) {
        LOG4CPLUS_DEBUG(logger, LOG4CPLUS_TEXT("Exception: ") << e.what());
        return std::make_shared<Result>(cudaErrorMemoryAllocation);
    }
   LOG4CPLUS_DEBUG(logger, "cublasDgemmBatched_v2 Executed");
    return std::make_shared<Result>(cs,out);
}


CUBLAS_ROUTINE_HANDLER(Cgemm_v2) {
    Logger logger=Logger::getInstance(LOG4CPLUS_TEXT("Cgemm"));
    
    cublasHandle_t handle;
    handle = (cublasHandle_t) in->Get<long long int>();
    
    cublasOperation_t transa = in->Get<cublasOperation_t>();
    cublasOperation_t transb = in->Get<cublasOperation_t>();
    int m  = in->Get<int>();
    int n  = in->Get<int>();
    int k  = in->Get<int>();
    const cuComplex * alpha = in->Assign<cuComplex>();
    
    cuComplex * A = in->GetFromMarshal<cuComplex*>();
    int lda = in->Get<int>();
    cuComplex * B = in->GetFromMarshal<cuComplex*>();
    int ldb = in->Get<int>();
    const cuComplex * beta = in->Assign<cuComplex>();
    cuComplex * C = in->GetFromMarshal<cuComplex*>();
    int ldc = in->Get<int>();
    cublasStatus_t cs;
        std::shared_ptr<Buffer> out = std::make_shared<Buffer>();

    try{
        cs = cublasCgemm(handle,transa,transb,m,n,k,alpha,A,lda,B,ldb,beta,C,ldc);
        out->AddMarshal<cuComplex *>(C);
    } catch (const std::exception& e) {
        LOG4CPLUS_DEBUG(logger, LOG4CPLUS_TEXT("Exception: ") << e.what());
        return std::make_shared<Result>(cudaErrorMemoryAllocation);
    }
    LOG4CPLUS_DEBUG(logger, "cublasCgemm_v2 Executed");
    return std::make_shared<Result>(cs,out);
}

CUBLAS_ROUTINE_HANDLER(CgemmBatched_v2){
    Logger logger=Logger::getInstance(LOG4CPLUS_TEXT("CgemmBatched"));
    
    cublasHandle_t handle;
    handle = (cublasHandle_t) in->Get<long long int>();
    
    cublasOperation_t transa = in->Get<cublasOperation_t>();
    cublasOperation_t transb = in->Get<cublasOperation_t>();
    int m  = in->Get<int>();
    int n  = in->Get<int>();
    int k  = in->Get<int>();
    const cuComplex * alpha = in->Assign<cuComplex>();
    
    const cuComplex ** A = (const cuComplex **) in->GetFromMarshal<cuComplex**>();
    int lda = in->Get<int>();
    const cuComplex ** B = (const cuComplex **) in->GetFromMarshal<cuComplex**>();
    int ldb = in->Get<int>();
    const cuComplex * beta = in->Assign<cuComplex>();
    cuComplex ** C = in->GetFromMarshal<cuComplex**>();
    int ldc = in->Get<int>();
    int batchSize = in->Get<int>();
    
    cublasStatus_t cs;
        std::shared_ptr<Buffer> out = std::make_shared<Buffer>();

    try{
        cs = cublasCgemmBatched(handle,transa,transb,m,n,k,alpha,A,lda,B,ldb,beta,C,ldc,batchSize);
        out->AddMarshal<cuComplex **>(C);
    } catch (const std::exception& e) {
        LOG4CPLUS_DEBUG(logger, LOG4CPLUS_TEXT("Exception: ") << e.what());
        return std::make_shared<Result>(cudaErrorMemoryAllocation);
    }
    LOG4CPLUS_DEBUG(logger, "cublasCgemmBatched_v2 Executed");
    return std::make_shared<Result>(cs, out);
}


CUBLAS_ROUTINE_HANDLER(Zgemm_v2) {
    Logger logger=Logger::getInstance(LOG4CPLUS_TEXT("Zgemm"));
    
    cublasHandle_t handle;
    handle = (cublasHandle_t) in->Get<long long int>();
    
    cublasOperation_t transa = in->Get<cublasOperation_t>();
    cublasOperation_t transb = in->Get<cublasOperation_t>();
    int m  = in->Get<int>();
    int n  = in->Get<int>();
    int k  = in->Get<int>();
    const cuDoubleComplex * alpha = in->Assign<cuDoubleComplex>();
    
    cuDoubleComplex * A = in->GetFromMarshal<cuDoubleComplex*>();
    int lda = in->Get<int>();
    cuDoubleComplex * B = in->GetFromMarshal<cuDoubleComplex*>();
    int ldb = in->Get<int>();
    const cuDoubleComplex * beta = in->Assign<cuDoubleComplex>();
    cuDoubleComplex * C = in->GetFromMarshal<cuDoubleComplex*>();
    int ldc = in->Get<int>();
    cublasStatus_t cs;
        std::shared_ptr<Buffer> out = std::make_shared<Buffer>();

    try{
        cs = cublasZgemm(handle,transa,transb,m,n,k,alpha,A,lda,B,ldb,beta,C,ldc);
        out->AddMarshal<cuDoubleComplex *>(C);
    } catch (const std::exception& e) {
        LOG4CPLUS_DEBUG(logger, LOG4CPLUS_TEXT("Exception: ") << e.what());
        return std::make_shared<Result>(cudaErrorMemoryAllocation);
    }
    LOG4CPLUS_DEBUG(logger, "cublasZgemm_v2 Executed");
    return std::make_shared<Result>(cs, out);
}

CUBLAS_ROUTINE_HANDLER(ZgemmBatched_v2){
    Logger logger=Logger::getInstance(LOG4CPLUS_TEXT("ZgemmBatched"));
    
    cublasHandle_t handle;
    handle = (cublasHandle_t) in->Get<long long int>();
    
    cublasOperation_t transa = in->Get<cublasOperation_t>();
    cublasOperation_t transb = in->Get<cublasOperation_t>();
    int m  = in->Get<int>();
    int n  = in->Get<int>();
    int k  = in->Get<int>();
    const cuDoubleComplex * alpha = in->Assign<cuDoubleComplex>();
    
    const cuDoubleComplex ** A = (const cuDoubleComplex **) in->GetFromMarshal<cuDoubleComplex**>();
    int lda = in->Get<int>();
    const cuDoubleComplex ** B = (const cuDoubleComplex **) in->GetFromMarshal<cuDoubleComplex**>();
    int ldb = in->Get<int>();
    const cuDoubleComplex * beta = in->Assign<cuDoubleComplex>();
    cuDoubleComplex ** C = in->GetFromMarshal<cuDoubleComplex**>();
    int ldc = in->Get<int>();
    int batchSize = in->Get<int>();
    
    cublasStatus_t cs;
        std::shared_ptr<Buffer> out = std::make_shared<Buffer>();

    try{
        cs = cublasZgemmBatched(handle,transa,transb,m,n,k,alpha,A,lda,B,ldb,beta,C,ldc,batchSize);
        out->AddMarshal<cuDoubleComplex **>(C);
    } catch (const std::exception& e) {
        LOG4CPLUS_DEBUG(logger, LOG4CPLUS_TEXT("Exception: ") << e.what());
        return std::make_shared<Result>(cudaErrorMemoryAllocation);
    }
    LOG4CPLUS_DEBUG(logger, "cublasZgemmBatched_v2 Executed");
    return std::make_shared<Result>(cs, out);
}

CUBLAS_ROUTINE_HANDLER(Snrm2_v2){
    Logger logger=Logger::getInstance(LOG4CPLUS_TEXT("Snrm2_v2"));
    
    cublasHandle_t handle = in->Get<cublasHandle_t>();
    int n = in->Get<int>();
    const float * x = in->GetFromMarshal<float*>();
    int incx = in->Get<int>();
    float result;
    
    cublasStatus_t cs = cublasSnrm2_v2(handle, n, x, incx, &result);
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();

    try {
        out->Add<float>(result);
    } catch (const std::exception& e) {
        LOG4CPLUS_DEBUG(logger, LOG4CPLUS_TEXT("Exception: ") << e.what());
        return std::make_shared<Result>(cudaErrorMemoryAllocation);
    }
    LOG4CPLUS_DEBUG(logger, "cublasSnrm2_v2 Executed");
    return std::make_shared<Result>(cs, out);
}

CUBLAS_ROUTINE_HANDLER(Dnrm2_v2){
    Logger logger=Logger::getInstance(LOG4CPLUS_TEXT("Dnrm2_v2"));
    
    cublasHandle_t handle;
    handle = in->Get<cublasHandle_t>();
    int n = in->Get<int>();
    double * x = in->GetFromMarshal<double*>();
    int incx = in->Get<int>();
    double result;
    
    cublasStatus_t cs = cublasDnrm2_v2(handle, n, x, incx, &result);
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();

    try {
        out->Add<double>(result);
    } catch (const std::exception& e) {
        LOG4CPLUS_DEBUG(logger, LOG4CPLUS_TEXT("Exception: ") << e.what());
        return std::make_shared<Result>(cudaErrorMemoryAllocation);
    }
    LOG4CPLUS_DEBUG(logger, "cublasDnrm2_v2 Executed");
    return std::make_shared<Result>(cs, out);
}

CUBLAS_ROUTINE_HANDLER(Scnrm2_v2){
    Logger logger=Logger::getInstance(LOG4CPLUS_TEXT("Scnrm2_v2"));
    
    cublasHandle_t handle;
    handle = (cublasHandle_t) in->Get<long long int>();
    int n = in->Get<int>();
    cuComplex * x = in->GetFromMarshal<cuComplex*>();
    int incx = in->Get<int>();
    float result;
    
    cublasStatus_t cs = cublasScnrm2_v2(handle, n, x, incx, &result);
        std::shared_ptr<Buffer> out = std::make_shared<Buffer>();

    try{
        out->Add<float>(result);
    } catch (const std::exception& e) {
        LOG4CPLUS_DEBUG(logger, LOG4CPLUS_TEXT("Exception: ") << e.what());
        return std::make_shared<Result>(cudaErrorMemoryAllocation);
    }
    LOG4CPLUS_DEBUG(logger, "cublasScnrm2_v2 Executed");
    return std::make_shared<Result>(cs, out);
}

CUBLAS_ROUTINE_HANDLER(Dznrm2_v2){
    Logger logger=Logger::getInstance(LOG4CPLUS_TEXT("Dznrm2_v2"));
    
    cublasHandle_t handle;
    handle = (cublasHandle_t) in->Get<long long int>();
    int n = in->Get<int>();
    cuDoubleComplex * x = in->GetFromMarshal<cuDoubleComplex*>();
    int incx = in->Get<int>();
    double result;
    
    cublasStatus_t cs = cublasDznrm2_v2(handle, n, x, incx, &result);
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();

    try {
        out->Add<double>(result);
    } catch (const std::exception& e) {
        LOG4CPLUS_DEBUG(logger, LOG4CPLUS_TEXT("Exception: ") << e.what());
        return std::make_shared<Result>(cudaErrorMemoryAllocation);
    }
    LOG4CPLUS_DEBUG(logger, "cublasDznrm2_v2 Executed");
    return std::make_shared<Result>(cs,out);
}

CUBLAS_ROUTINE_HANDLER(Ssyrk_v2){
    Logger logger=Logger::getInstance(LOG4CPLUS_TEXT("Ssyrk_v2"));
    
    cublasHandle_t handle;
    handle = (cublasHandle_t) in->Get<long long int>();
    cublasFillMode_t uplo = in->Get<cublasFillMode_t>();
    cublasOperation_t trans = in->Get<cublasOperation_t>();
    int n = in->Get<int>();
    int k = in->Get<int>();
    const float * alpha = in->Assign<float>();
    float * A = in->GetFromMarshal<float*>();
    int lda = in->Get<int>();
    const float * beta = in->Assign<float>();
    float * C = in->GetFromMarshal<float*>();
    int ldc = in->Get<int>();
    
    cublasStatus_t cs = cublasSsyrk_v2(handle,uplo,trans,n,k,alpha,A,lda,beta,C,ldc);
    return std::make_shared<Result>(cs);
}

CUBLAS_ROUTINE_HANDLER(Dsyrk_v2){
    Logger logger=Logger::getInstance(LOG4CPLUS_TEXT("Dsyrk_v2"));
    
    cublasHandle_t handle;
    handle = (cublasHandle_t) in->Get<long long int>();
    cublasFillMode_t uplo = in->Get<cublasFillMode_t>();
    cublasOperation_t trans = in->Get<cublasOperation_t>();
    int n = in->Get<int>();
    int k = in->Get<int>();
    const double * alpha = in->Assign<double>();
    double * A = in->GetFromMarshal<double*>();
    int lda = in->Get<int>();
    const double * beta = in->Assign<double>();
    double * C = in->GetFromMarshal<double*>();
    int ldc = in->Get<int>();
    
    cublasStatus_t cs = cublasDsyrk_v2(handle,uplo,trans,n,k,alpha,A,lda,beta,C,ldc);
    return std::make_shared<Result>(cs);
}

CUBLAS_ROUTINE_HANDLER(Csyrk_v2){
    Logger logger=Logger::getInstance(LOG4CPLUS_TEXT("Csyrk_v2"));
    
    cublasHandle_t handle;
    handle = (cublasHandle_t) in->Get<long long int>();
    cublasFillMode_t uplo = in->Get<cublasFillMode_t>();
    cublasOperation_t trans = in->Get<cublasOperation_t>();
    int n = in->Get<int>();
    int k = in->Get<int>();
    const cuComplex * alpha = in->Assign<cuComplex>();
    cuComplex * A = in->GetFromMarshal<cuComplex*>();
    int lda = in->Get<int>();
    const cuComplex * beta = in->Assign<cuComplex>();
    cuComplex * C = in->GetFromMarshal<cuComplex*>();
    int ldc = in->Get<int>();
    
    cublasStatus_t cs = cublasCsyrk_v2(handle,uplo,trans,n,k,alpha,A,lda,beta,C,ldc);
    return std::make_shared<Result>(cs);
}

CUBLAS_ROUTINE_HANDLER(Zsyrk_v2){
    Logger logger=Logger::getInstance(LOG4CPLUS_TEXT("Zsyrk_v2"));
    
    cublasHandle_t handle;
    handle = (cublasHandle_t) in->Get<long long int>();
    cublasFillMode_t uplo = in->Get<cublasFillMode_t>();
    cublasOperation_t trans = in->Get<cublasOperation_t>();
    int n = in->Get<int>();
    int k = in->Get<int>();
    const cuDoubleComplex * alpha = in->Assign<cuDoubleComplex>();
    cuDoubleComplex * A = in->GetFromMarshal<cuDoubleComplex*>();
    int lda = in->Get<int>();
    const cuDoubleComplex * beta = in->Assign<cuDoubleComplex>();
    cuDoubleComplex * C = in->GetFromMarshal<cuDoubleComplex*>();
    int ldc = in->Get<int>();
    
    cublasStatus_t cs = cublasZsyrk_v2(handle,uplo,trans,n,k,alpha,A,lda,beta,C,ldc);
    return std::make_shared<Result>(cs);
}


CUBLAS_ROUTINE_HANDLER(Cherk_v2){
    Logger logger=Logger::getInstance(LOG4CPLUS_TEXT("Cherk_v2"));
    
    cublasHandle_t handle;
    handle = (cublasHandle_t) in->Get<long long int>();
    cublasFillMode_t uplo = in->Get<cublasFillMode_t>();
    cublasOperation_t trans = in->Get<cublasOperation_t>();
    int n = in->Get<int>();
    int k = in->Get<int>();
    const float * alpha = in->Assign<float>();
    cuComplex * A = in->GetFromMarshal<cuComplex*>();
    int lda = in->Get<int>();
    const float * beta = in->Assign<float>();
    cuComplex * C = in->GetFromMarshal<cuComplex*>();
    int ldc = in->Get<int>();
    
    cublasStatus_t cs = cublasCherk_v2(handle,uplo,trans,n,k,alpha,A,lda,beta,C,ldc);
    return std::make_shared<Result>(cs);
}

CUBLAS_ROUTINE_HANDLER(Zherk_v2){
    Logger logger=Logger::getInstance(LOG4CPLUS_TEXT("Zherk_v2"));
    
    cublasHandle_t handle;
    handle = (cublasHandle_t) in->Get<long long int>();
    cublasFillMode_t uplo = in->Get<cublasFillMode_t>();
    cublasOperation_t trans = in->Get<cublasOperation_t>();
    int n = in->Get<int>();
    int k = in->Get<int>();
    const double * alpha = in->Assign<double>();
    cuDoubleComplex * A = in->GetFromMarshal<cuDoubleComplex*>();
    int lda = in->Get<int>();
    const double * beta = in->Assign<double>();
    cuDoubleComplex * C = in->GetFromMarshal<cuDoubleComplex*>();
    int ldc = in->Get<int>();
    
    cublasStatus_t cs = cublasZherk_v2(handle,uplo,trans,n,k,alpha,A,lda,beta,C,ldc);
    return std::make_shared<Result>(cs);
}

CUBLAS_ROUTINE_HANDLER(Ssyr2k_v2){
    Logger logger=Logger::getInstance(LOG4CPLUS_TEXT("Ssyr2k_v2"));
    
    cublasHandle_t handle;
    handle = (cublasHandle_t) in->Get<long long int>();
    cublasFillMode_t uplo = in->Get<cublasFillMode_t>();
    cublasOperation_t trans = in->Get<cublasOperation_t>();
    int n = in->Get<int>();
    int k = in->Get<int>();
    const float * alpha = in->Assign<float>();
    float * A = in->GetFromMarshal<float*>();
    int lda = in->Get<int>();
    
    float * B = in->GetFromMarshal<float*>();
    int ldb = in->Get<int>();
    
    const float * beta = in->Assign<float>();
    
    float * C = in->GetFromMarshal<float*>();
    int ldc = in->Get<int>();
    
    cublasStatus_t cs = cublasSsyr2k_v2(handle,uplo,trans,n,k,alpha,A,lda,B,ldb,beta,C,ldc);
    return std::make_shared<Result>(cs);
}


CUBLAS_ROUTINE_HANDLER(Dsyr2k_v2){
    Logger logger=Logger::getInstance(LOG4CPLUS_TEXT("Dsyr2k_v2"));
    
    cublasHandle_t handle;
    handle = (cublasHandle_t) in->Get<long long int>();
    cublasFillMode_t uplo = in->Get<cublasFillMode_t>();
    cublasOperation_t trans = in->Get<cublasOperation_t>();
    int n = in->Get<int>();
    int k = in->Get<int>();
    const double * alpha = in->Assign<double>();
    double * A = in->GetFromMarshal<double*>();
    int lda = in->Get<int>();
    
    double * B = in->GetFromMarshal<double*>();
    int ldb = in->Get<int>();
    
    const double * beta = in->Assign<double>();
    
    double * C = in->GetFromMarshal<double*>();
    int ldc = in->Get<int>();
    
    cublasStatus_t cs = cublasDsyr2k_v2(handle,uplo,trans,n,k,alpha,A,lda,B,ldb,beta,C,ldc);
    return std::make_shared<Result>(cs);
}

CUBLAS_ROUTINE_HANDLER(Csyr2k_v2){
    Logger logger=Logger::getInstance(LOG4CPLUS_TEXT("Csyr2k_v2"));
    
    cublasHandle_t handle;
    handle = (cublasHandle_t) in->Get<long long int>();
    cublasFillMode_t uplo = in->Get<cublasFillMode_t>();
    cublasOperation_t trans = in->Get<cublasOperation_t>();
    int n = in->Get<int>();
    int k = in->Get<int>();
    const cuComplex * alpha = in->Assign<cuComplex>();
    cuComplex * A = in->GetFromMarshal<cuComplex*>();
    int lda = in->Get<int>();
    
    cuComplex * B = in->GetFromMarshal<cuComplex*>();
    int ldb = in->Get<int>();
    
    const cuComplex * beta = in->Assign<cuComplex>();
    
    cuComplex * C = in->GetFromMarshal<cuComplex*>();
    int ldc = in->Get<int>();
    
    cublasStatus_t cs = cublasCsyr2k_v2(handle,uplo,trans,n,k,alpha,A,lda,B,ldb,beta,C,ldc);
    return std::make_shared<Result>(cs);
}


CUBLAS_ROUTINE_HANDLER(Zsyr2k_v2){
    Logger logger=Logger::getInstance(LOG4CPLUS_TEXT("Zsyr2k_v2"));
    
    cublasHandle_t handle;
    handle = (cublasHandle_t) in->Get<long long int>();
    cublasFillMode_t uplo = in->Get<cublasFillMode_t>();
    cublasOperation_t trans = in->Get<cublasOperation_t>();
    int n = in->Get<int>();
    int k = in->Get<int>();
    const cuDoubleComplex * alpha = in->Assign<cuDoubleComplex>();
    cuDoubleComplex * A = in->GetFromMarshal<cuDoubleComplex*>();
    int lda = in->Get<int>();
    
    cuDoubleComplex * B = in->GetFromMarshal<cuDoubleComplex*>();
    int ldb = in->Get<int>();
    
    const cuDoubleComplex * beta = in->Assign<cuDoubleComplex>();
    
    cuDoubleComplex * C = in->GetFromMarshal<cuDoubleComplex*>();
    int ldc = in->Get<int>();
    
    cublasStatus_t cs = cublasZsyr2k_v2(handle,uplo,trans,n,k,alpha,A,lda,B,ldb,beta,C,ldc);
    return std::make_shared<Result>(cs);
}

CUBLAS_ROUTINE_HANDLER(Cher2k_v2){
    Logger logger=Logger::getInstance(LOG4CPLUS_TEXT("Cher2k_v2"));
    
    cublasHandle_t handle;
    handle = (cublasHandle_t) in->Get<long long int>();
    cublasFillMode_t uplo = in->Get<cublasFillMode_t>();
    cublasOperation_t trans = in->Get<cublasOperation_t>();
    int n = in->Get<int>();
    int k = in->Get<int>();
    const cuComplex * alpha = in->Assign<cuComplex>();
    cuComplex * A = in->GetFromMarshal<cuComplex*>();
    int lda = in->Get<int>();
    
    cuComplex * B = in->GetFromMarshal<cuComplex*>();
    int ldb = in->Get<int>();
    
    const float * beta = in->Assign<float>();
    
    cuComplex * C = in->GetFromMarshal<cuComplex*>();
    int ldc = in->Get<int>();
    
    cublasStatus_t cs = cublasCher2k_v2(handle,uplo,trans,n,k,alpha,A,lda,B,ldb,beta,C,ldc);
    return std::make_shared<Result>(cs);
}

CUBLAS_ROUTINE_HANDLER(Zher2k_v2){
    Logger logger=Logger::getInstance(LOG4CPLUS_TEXT("Zher2k_v2"));
    
    cublasHandle_t handle;
    handle = (cublasHandle_t) in->Get<long long int>();
    cublasFillMode_t uplo = in->Get<cublasFillMode_t>();
    cublasOperation_t trans = in->Get<cublasOperation_t>();
    int n = in->Get<int>();
    int k = in->Get<int>();
    const cuDoubleComplex * alpha = in->Assign<cuDoubleComplex>();
    cuDoubleComplex * A = in->GetFromMarshal<cuDoubleComplex*>();
    int lda = in->Get<int>();
    
    cuDoubleComplex * B = in->GetFromMarshal<cuDoubleComplex*>();
    int ldb = in->Get<int>();
    
    const double * beta = in->Assign<double>();
    
    cuDoubleComplex * C = in->GetFromMarshal<cuDoubleComplex*>();
    int ldc = in->Get<int>();
    
    cublasStatus_t cs = cublasZher2k_v2(handle,uplo,trans,n,k,alpha,A,lda,B,ldb,beta,C,ldc);
    return std::make_shared<Result>(cs);
}


CUBLAS_ROUTINE_HANDLER(Ssymm_v2){
    Logger logger=Logger::getInstance(LOG4CPLUS_TEXT("Ssymm_v2"));
    
    cublasHandle_t handle;
    handle = (cublasHandle_t) in->Get<long long int>();
    cublasSideMode_t side = in->Get<cublasSideMode_t>();
    cublasFillMode_t uplo = in->Get<cublasFillMode_t>();
    int n = in->Get<int>();
    int k = in->Get<int>();
   
    const float * alpha = in->Assign<float>();
    float * A = in->GetFromMarshal<float*>();
    int lda = in->Get<int>();
    
    float * B = in->GetFromMarshal<float*>();
    int ldb = in->Get<int>();
    
    const float * beta = in->Assign<float>();
    
    float * C = in->GetFromMarshal<float*>();
    int ldc = in->Get<int>();
    
    cublasStatus_t cs = cublasSsymm_v2(handle,side,uplo,n,k,alpha,A,lda,B,ldb,beta,C,ldc);
    return std::make_shared<Result>(cs);
}

CUBLAS_ROUTINE_HANDLER(Dsymm_v2){
    Logger logger=Logger::getInstance(LOG4CPLUS_TEXT("Dsymm_v2"));
    
    cublasHandle_t handle;
    handle = (cublasHandle_t) in->Get<long long int>();
    cublasSideMode_t side = in->Get<cublasSideMode_t>();
    cublasFillMode_t uplo = in->Get<cublasFillMode_t>();
    int n = in->Get<int>();
    int k = in->Get<int>();
   
    const double * alpha = in->Assign<double>();
    double * A = in->GetFromMarshal<double*>();
    int lda = in->Get<int>();
    
    double * B = in->GetFromMarshal<double*>();
    int ldb = in->Get<int>();
    
    const double * beta = in->Assign<double>();
    
    double * C = in->GetFromMarshal<double*>();
    int ldc = in->Get<int>();
    
    cublasStatus_t cs = cublasDsymm_v2(handle,side,uplo,n,k,alpha,A,lda,B,ldb,beta,C,ldc);
    return std::make_shared<Result>(cs);
}


CUBLAS_ROUTINE_HANDLER(Csymm_v2){
    Logger logger=Logger::getInstance(LOG4CPLUS_TEXT("Csymm_v2"));
    
    cublasHandle_t handle;
    handle = (cublasHandle_t) in->Get<long long int>();
    cublasSideMode_t side = in->Get<cublasSideMode_t>();
    cublasFillMode_t uplo = in->Get<cublasFillMode_t>();
    int n = in->Get<int>();
    int k = in->Get<int>();
   
    const cuComplex * alpha = in->Assign<cuComplex>();
    cuComplex * A = in->GetFromMarshal<cuComplex*>();
    int lda = in->Get<int>();
    
    cuComplex * B = in->GetFromMarshal<cuComplex*>();
    int ldb = in->Get<int>();
    
    const cuComplex * beta = in->Assign<cuComplex>();
    
    cuComplex * C = in->GetFromMarshal<cuComplex*>();
    int ldc = in->Get<int>();
    
    cublasStatus_t cs = cublasCsymm_v2(handle,side,uplo,n,k,alpha,A,lda,B,ldb,beta,C,ldc);
    return std::make_shared<Result>(cs);
}

CUBLAS_ROUTINE_HANDLER(Zsymm_v2){
    Logger logger=Logger::getInstance(LOG4CPLUS_TEXT("Zsymm_v2"));
    
    cublasHandle_t handle;
    handle = (cublasHandle_t) in->Get<long long int>();
    cublasSideMode_t side = in->Get<cublasSideMode_t>();
    cublasFillMode_t uplo = in->Get<cublasFillMode_t>();
    int n = in->Get<int>();
    int k = in->Get<int>();
   
    const cuDoubleComplex * alpha = in->Assign<cuDoubleComplex>();
    cuDoubleComplex * A = in->GetFromMarshal<cuDoubleComplex*>();
    int lda = in->Get<int>();
    
    cuDoubleComplex * B = in->GetFromMarshal<cuDoubleComplex*>();
    int ldb = in->Get<int>();
    
    const cuDoubleComplex * beta = in->Assign<cuDoubleComplex>();
    
    cuDoubleComplex * C = in->GetFromMarshal<cuDoubleComplex*>();
    int ldc = in->Get<int>();
    
    cublasStatus_t cs = cublasZsymm_v2(handle,side,uplo,n,k,alpha,A,lda,B,ldb,beta,C,ldc);
    return std::make_shared<Result>(cs);
}

CUBLAS_ROUTINE_HANDLER(Chemm_v2){
    Logger logger=Logger::getInstance(LOG4CPLUS_TEXT("Chemm_v2"));
    
    cublasHandle_t handle;
    handle = (cublasHandle_t) in->Get<long long int>();
    cublasSideMode_t side = in->Get<cublasSideMode_t>();
    cublasFillMode_t uplo = in->Get<cublasFillMode_t>();
    int n = in->Get<int>();
    int k = in->Get<int>();
   
    const cuComplex * alpha = in->Assign<cuComplex>();
    cuComplex * A = in->GetFromMarshal<cuComplex*>();
    int lda = in->Get<int>();
    
    cuComplex * B = in->GetFromMarshal<cuComplex*>();
    int ldb = in->Get<int>();
    
    const cuComplex * beta = in->Assign<cuComplex>();
    
    cuComplex * C = in->GetFromMarshal<cuComplex*>();
    int ldc = in->Get<int>();
    
    cublasStatus_t cs = cublasChemm_v2(handle,side,uplo,n,k,alpha,A,lda,B,ldb,beta,C,ldc);
    return std::make_shared<Result>(cs);
}

CUBLAS_ROUTINE_HANDLER(Zhemm_v2){
    Logger logger=Logger::getInstance(LOG4CPLUS_TEXT("Zhemm_v2"));
    
    cublasHandle_t handle;
    handle = (cublasHandle_t) in->Get<long long int>();
    cublasSideMode_t side = in->Get<cublasSideMode_t>();
    cublasFillMode_t uplo = in->Get<cublasFillMode_t>();
    int n = in->Get<int>();
    int k = in->Get<int>();
   
    const cuDoubleComplex * alpha = in->Assign<cuDoubleComplex>();
    cuDoubleComplex * A = in->GetFromMarshal<cuDoubleComplex*>();
    int lda = in->Get<int>();
    
    cuDoubleComplex * B = in->GetFromMarshal<cuDoubleComplex*>();
    int ldb = in->Get<int>();
    
    const cuDoubleComplex * beta = in->Assign<cuDoubleComplex>();
    
    cuDoubleComplex * C = in->GetFromMarshal<cuDoubleComplex*>();
    int ldc = in->Get<int>();
    
    cublasStatus_t cs = cublasZhemm_v2(handle,side,uplo,n,k,alpha,A,lda,B,ldb,beta,C,ldc);
    return std::make_shared<Result>(cs);
}


CUBLAS_ROUTINE_HANDLER(Strsm_v2){
    Logger logger=Logger::getInstance(LOG4CPLUS_TEXT("Strsm_v2"));
    
    cublasHandle_t handle;
    handle = (cublasHandle_t) in->Get<long long int>();
    cublasSideMode_t side = in->Get<cublasSideMode_t>();
    cublasFillMode_t uplo = in->Get<cublasFillMode_t>();
    cublasOperation_t trans = in->Get<cublasOperation_t>();
    cublasDiagType_t diag = in->Get<cublasDiagType_t>();
    int m  = in->Get<int>();
    int n  = in->Get<int>();
    
    const float * alpha = in->Assign<float>();
    float * A = in->GetFromMarshal<float*>();
    int lda = in->Get<int>();
    
    float * B = in->GetFromMarshal<float*>();
    int ldb = in->Get<int>();
    
    cublasStatus_t cs = cublasStrsm_v2(handle,side,uplo,trans,diag,m,n,alpha,A,lda,B,ldb);
    return std::make_shared<Result>(cs);
}

CUBLAS_ROUTINE_HANDLER(Dtrsm_v2){
    Logger logger=Logger::getInstance(LOG4CPLUS_TEXT("Dtrsm_v2"));
    
    cublasHandle_t handle;
    handle = (cublasHandle_t) in->Get<long long int>();
    cublasSideMode_t side = in->Get<cublasSideMode_t>();
    cublasFillMode_t uplo = in->Get<cublasFillMode_t>();
    cublasOperation_t trans = in->Get<cublasOperation_t>();
    cublasDiagType_t diag = in->Get<cublasDiagType_t>();
    int m  = in->Get<int>();
    int n  = in->Get<int>();
    
    const double * alpha = in->Assign<double>();
    double * A = in->GetFromMarshal<double*>();
    int lda = in->Get<int>();
    
    double * B = in->GetFromMarshal<double*>();
    int ldb = in->Get<int>();
    
    cublasStatus_t cs = cublasDtrsm_v2(handle,side,uplo,trans,diag,m,n,alpha,A,lda,B,ldb);
    return std::make_shared<Result>(cs);
}

CUBLAS_ROUTINE_HANDLER(Ctrsm_v2){
    Logger logger=Logger::getInstance(LOG4CPLUS_TEXT("Ctrsm_v2"));
    
    cublasHandle_t handle;
    handle = (cublasHandle_t) in->Get<long long int>();
    cublasSideMode_t side = in->Get<cublasSideMode_t>();
    cublasFillMode_t uplo = in->Get<cublasFillMode_t>();
    cublasOperation_t trans = in->Get<cublasOperation_t>();
    cublasDiagType_t diag = in->Get<cublasDiagType_t>();
    int m  = in->Get<int>();
    int n  = in->Get<int>();
    
    const cuComplex * alpha = in->Assign<cuComplex>();
    cuComplex * A = in->GetFromMarshal<cuComplex*>();
    int lda = in->Get<int>();
    
    cuComplex * B = in->GetFromMarshal<cuComplex*>();
    int ldb = in->Get<int>();
    
    cublasStatus_t cs = cublasCtrsm_v2(handle,side,uplo,trans,diag,m,n,alpha,A,lda,B,ldb);
    return std::make_shared<Result>(cs);
}

CUBLAS_ROUTINE_HANDLER(Ztrsm_v2){
    Logger logger=Logger::getInstance(LOG4CPLUS_TEXT("Ztrsm_v2"));
    
    cublasHandle_t handle;
    handle = (cublasHandle_t) in->Get<long long int>();
    cublasSideMode_t side = in->Get<cublasSideMode_t>();
    cublasFillMode_t uplo = in->Get<cublasFillMode_t>();
    cublasOperation_t trans = in->Get<cublasOperation_t>();
    cublasDiagType_t diag = in->Get<cublasDiagType_t>();
    int m  = in->Get<int>();
    int n  = in->Get<int>();
    
    const cuDoubleComplex * alpha = in->Assign<cuDoubleComplex>();
    cuDoubleComplex * A = in->GetFromMarshal<cuDoubleComplex*>();
    int lda = in->Get<int>();
    
    cuDoubleComplex * B = in->GetFromMarshal<cuDoubleComplex*>();
    int ldb = in->Get<int>();
    
    cublasStatus_t cs = cublasZtrsm_v2(handle,side,uplo,trans,diag,m,n,alpha,A,lda,B,ldb);
    return std::make_shared<Result>(cs);
}

CUBLAS_ROUTINE_HANDLER(Strmm_v2){
    Logger logger=Logger::getInstance(LOG4CPLUS_TEXT("Strmm_v2"));
    
    cublasHandle_t handle;
    handle = (cublasHandle_t) in->Get<long long int>();
    cublasSideMode_t side = in->Get<cublasSideMode_t>();
    cublasFillMode_t uplo = in->Get<cublasFillMode_t>();
    cublasOperation_t trans = in->Get<cublasOperation_t>();
    cublasDiagType_t diag = in->Get<cublasDiagType_t>();
    int m  = in->Get<int>();
    int n  = in->Get<int>();
    
    const float * alpha = in->Assign<float>();
    float * A = in->GetFromMarshal<float*>();
    int lda = in->Get<int>();
    
    float * B = in->GetFromMarshal<float*>();
    int ldb = in->Get<int>();
    
    float * C = in->GetFromMarshal<float*>();
    int ldc = in->Get<int>();
    
    cublasStatus_t cs = cublasStrmm_v2(handle,side,uplo,trans,diag,m,n,alpha,A,lda,B,ldb,C,ldc);
    return std::make_shared<Result>(cs);
}

CUBLAS_ROUTINE_HANDLER(Dtrmm_v2){
    Logger logger=Logger::getInstance(LOG4CPLUS_TEXT("Dtrmm_v2"));
    
    cublasHandle_t handle;
    handle = (cublasHandle_t) in->Get<long long int>();
    cublasSideMode_t side = in->Get<cublasSideMode_t>();
    cublasFillMode_t uplo = in->Get<cublasFillMode_t>();
    cublasOperation_t trans = in->Get<cublasOperation_t>();
    cublasDiagType_t diag = in->Get<cublasDiagType_t>();
    int m  = in->Get<int>();
    int n  = in->Get<int>();
    
    const double * alpha = in->Assign<double>();
    double * A = in->GetFromMarshal<double*>();
    int lda = in->Get<int>();
    
    double * B = in->GetFromMarshal<double*>();
    int ldb = in->Get<int>();
    
    double * C = in->GetFromMarshal<double*>();
    int ldc = in->Get<int>();
    
    cublasStatus_t cs = cublasDtrmm_v2(handle,side,uplo,trans,diag,m,n,alpha,A,lda,B,ldb,C,ldc);
    return std::make_shared<Result>(cs);
}

CUBLAS_ROUTINE_HANDLER(Ctrmm_v2){
    Logger logger=Logger::getInstance(LOG4CPLUS_TEXT("Ctrmm_v2"));
    
    cublasHandle_t handle;
    handle = (cublasHandle_t) in->Get<long long int>();
    cublasSideMode_t side = in->Get<cublasSideMode_t>();
    cublasFillMode_t uplo = in->Get<cublasFillMode_t>();
    cublasOperation_t trans = in->Get<cublasOperation_t>();
    cublasDiagType_t diag = in->Get<cublasDiagType_t>();
    int m  = in->Get<int>();
    int n  = in->Get<int>();
    
    const cuComplex * alpha = in->Assign<cuComplex>();
    cuComplex * A = in->GetFromMarshal<cuComplex*>();
    int lda = in->Get<int>();
    
    cuComplex * B = in->GetFromMarshal<cuComplex*>();
    int ldb = in->Get<int>();
    
    cuComplex * C = in->GetFromMarshal<cuComplex*>();
    int ldc = in->Get<int>();
    
    cublasStatus_t cs = cublasCtrmm_v2(handle,side,uplo,trans,diag,m,n,alpha,A,lda,B,ldb,C,ldc);
    return std::make_shared<Result>(cs);
}

CUBLAS_ROUTINE_HANDLER(Ztrmm_v2){
    Logger logger=Logger::getInstance(LOG4CPLUS_TEXT("Ztrsm_v2"));
    
    cublasHandle_t handle;
    handle = (cublasHandle_t) in->Get<long long int>();
    cublasSideMode_t side = in->Get<cublasSideMode_t>();
    cublasFillMode_t uplo = in->Get<cublasFillMode_t>();
    cublasOperation_t trans = in->Get<cublasOperation_t>();
    cublasDiagType_t diag = in->Get<cublasDiagType_t>();
    int m  = in->Get<int>();
    int n  = in->Get<int>();
    
    const cuDoubleComplex * alpha = in->Assign<cuDoubleComplex>();
    cuDoubleComplex * A = in->GetFromMarshal<cuDoubleComplex*>();
    int lda = in->Get<int>();
    
    cuDoubleComplex * B = in->GetFromMarshal<cuDoubleComplex*>();
    int ldb = in->Get<int>();
    
    cuDoubleComplex * C = in->GetFromMarshal<cuDoubleComplex*>();
    int ldc = in->Get<int>();
    
    cublasStatus_t cs = cublasZtrmm_v2(handle,side,uplo,trans,diag,m,n,alpha,A,lda,B,ldb,C,ldc);
    return std::make_shared<Result>(cs);
}

CUBLAS_ROUTINE_HANDLER(SgemmStridedBatched) {
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("SgemmStridedBatched"));
    
    cublasHandle_t handle = in->Get<cublasHandle_t>();
    cublasOperation_t transa = in->Get<cublasOperation_t>();
    cublasOperation_t transb = in->Get<cublasOperation_t>();
    int m = in->Get<int>();
    int n = in->Get<int>();
    int k = in->Get<int>();
    const float *alpha = in->Assign<float>();
    const float *A = in->GetFromMarshal<float*>();
    int lda = in->Get<int>();
    long long int strideA = in->Get<long long int>();
    const float *B = in->GetFromMarshal<float*>();
    int ldb = in->Get<int>();
    long long int strideB = in->Get<long long int>();
    const float *beta = in->Assign<float>();
    float *C = in->GetFromMarshal<float*>();
    int ldc = in->Get<int>();
    long long int strideC = in->Get<long long int>();
    int batchCount = in->Get<int>();

    cublasStatus_t cs = cublasSgemmStridedBatched(handle, transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount);
    return std::make_shared<Result>(cs);
}