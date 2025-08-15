//
// Created by Mariano Aponte on 07/12/23.
//

#ifndef RDMACM_RDMACOMMUNICATOR_H
#define RDMACM_RDMACOMMUNICATOR_H

#include "gvirtus/communicators/Communicator.h"
#include "ktmrdma.h"
#include <rdma/rdma_cma.h>
#include <rdma/rdma_verbs.h>
#include <string>
#include <netdb.h>

#define BACKLOG 8
#define BUF_SIZE 1024
#define DEBUG

/**
 * @brief RdmaCommunicator represents a communication interface using RDMA (Remote Direct Memory Access).
 */
namespace gvirtus::communicators {
    class RdmaCommunicator : public Communicator {
    private:
        rdma_cm_id * rdmaCmId;
        rdma_cm_id * rdmaCmListenId;

        char hostname[256];
        char port[6];

        ibv_wc workCompletion;

        ibv_mr * memoryRegion;

        char preregisteredBuffer[1024 * 5];
        ibv_mr * preregisteredMr;
        
        bool isRoce = false;

    public:
        RdmaCommunicator() = default;
        RdmaCommunicator(const std::string& hostname, const std::string& port);
        RdmaCommunicator(const std::string& hostname, const std::string& port, bool isRoce);
        RdmaCommunicator(rdma_cm_id * rdmaCmId);

        ~RdmaCommunicator();

        void Serve();
        const Communicator *const Accept() const;

        void Connect();

        size_t Read(char * buffer, size_t size);
        size_t Write(const char * buffer, size_t size);

        void Sync();

        void Close();

        std::string to_string() {return "rdmacommunicator";};
    };

}

#endif //RDMACM_RDMACOMMUNICATOR_H
