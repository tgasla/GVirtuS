//
// Created by Mariano Aponte on 30/11/23.
//

#include "ktmrdma.h"

void ktm_rdma_getaddrinfo(char *node, char *service, struct rdma_addrinfo *hints,
                          struct rdma_addrinfo **res) {
#ifdef DEBUG
    std::cout << "addrinfo called" << std::endl;
#endif

    int addrinfoResult = rdma_getaddrinfo(node, service, hints, res);
    if (addrinfoResult == -1) {
        throw "rdma_getaddrinfo(): error: " + std::string(strerror(errno));
    } else if (addrinfoResult != 0) {
        throw "rdma_getaddrinfo(): error (non-zero return value): " +
            std::string(gai_strerror(addrinfoResult));
    }
}

void Testlib() { std::cout << "ciao" << std::endl; }

void ktm_rdma_create_ep(struct rdma_cm_id **id, struct rdma_addrinfo *res, struct ibv_pd *pd,
                        struct ibv_qp_init_attr *qp_init_attr) {
    if (rdma_create_ep(id, res, pd, qp_init_attr) == -1) {
        throw "rdma_create_ep(): error: " + std::string(strerror(errno));
    }
}

ibv_mr *ktm_rdma_reg_msgs(struct rdma_cm_id *id, void *addr, size_t length) {
    auto registered = rdma_reg_msgs(id, addr, length);

    if (not registered) {
        throw "rdma_reg_msgs(): error: " + std::string(strerror(errno));
    }

    return registered;
}

ibv_mr *ktm_rdma_reg_read(struct rdma_cm_id *id, void *addr, size_t length) {
    auto registered = rdma_reg_read(id, addr, length);

    if (not registered) {
        throw "rdma_reg_read(): error: " + std::string(strerror(errno));
    }

    return registered;
}

ibv_mr *ktm_rdma_reg_write(struct rdma_cm_id *id, void *addr, size_t length) {
    auto registered = rdma_reg_write(id, addr, length);

    if (not registered) {
        throw "rdma_reg_write(): error: " + std::string(strerror(errno));
    }

    return registered;
}

void ktm_rdma_connect(struct rdma_cm_id *id, struct rdma_conn_param *conn_param) {
    if (rdma_connect(id, conn_param) == -1) {
        throw "rdma_connect(): error: " + std::string(strerror(errno));
    }
}

void ktm_rdma_listen(struct rdma_cm_id *id, int backlog) {
    if (rdma_listen(id, backlog) == -1) {
        throw "rdma_listen(): error: " + std::string(strerror(errno));
    }
}

void ktm_rdma_get_request(struct rdma_cm_id *listen, struct rdma_cm_id **id) {
    if (rdma_get_request(listen, id) == -1) {
        throw "rdma_get_request(): error: " + std::string(strerror(errno));
    }
}

void ktm_rdma_accept(struct rdma_cm_id *id, struct rdma_conn_param *conn_param) {
    if (rdma_accept(id, conn_param) == -1) {
        throw "rdma_accept(): error: " + std::string(strerror(errno));
    }
}

void ktm_rdma_post_recv(struct rdma_cm_id *id, void *context, void *addr, size_t length,
                        struct ibv_mr *mr) {
    if (rdma_post_recv(id, context, addr, length, mr) == -1) {
        throw "rdma_post_recv(): error: " + std::string(strerror(errno));
    }
}

void ktm_rdma_post_send(struct rdma_cm_id *id, void *context, void *addr, size_t length,
                        struct ibv_mr *mr, int flags) {
    if (rdma_post_send(id, context, addr, length, mr, flags) == -1) {
        throw "rdma_post_send(): error: " + std::string(strerror(errno));
    }
}

int ktm_rdma_get_send_comp(struct rdma_cm_id *id, struct ibv_wc *wc) {
    int returned = rdma_get_send_comp(id, wc);

    if (returned < 0) {
        throw "rdma_get_send_comp(): error: " + std::string(strerror(errno));
    }

    if (wc->status != IBV_WC_SUCCESS) {
        throw "rdma_get_send_comp(): error: (completion with error) failed status " +
            std::string(ibv_wc_status_str(wc->status));
    }

    return returned;
}

int ktm_rdma_get_recv_comp(struct rdma_cm_id *id, struct ibv_wc *wc) {
    int returned = rdma_get_recv_comp(id, wc);

    if (returned < 0) {
        throw "rdma_get_recv_comp(): error: " + std::string(strerror(errno));
    }

    if (wc->status != IBV_WC_SUCCESS) {
        throw "rdma_get_send_comp(): error: (completion with error) failed status " +
            std::string(ibv_wc_status_str(wc->status));
    }

    return returned;
}

void ktm_rdma_post_read(struct rdma_cm_id *id, void *context, void *addr, size_t length,
                        struct ibv_mr *mr, int flags, uint64_t remote_addr, uint32_t rkey) {
    if (rdma_post_read(id, context, addr, length, mr, flags, remote_addr, rkey) == -1) {
        throw "rdma_post_read(): error: " + std::string(strerror(errno));
    }
}

void ktm_rdma_post_write(struct rdma_cm_id *id, void *context, void *addr, size_t length,
                         struct ibv_mr *mr, int flags, uint64_t remote_addr, uint32_t rkey) {
    if (rdma_post_write(id, context, addr, length, mr, flags, remote_addr, rkey) == -1) {
        throw "rdma_post_write(): error: " + std::string(strerror(errno));
    }
}

void ktm_rdma_send_address(struct rdma_cm_id *id, void *addr, size_t length, struct ibv_mr *mr) {
    auto local_int_address = (uintptr_t)mr->addr;
    memcpy(addr, &local_int_address, sizeof(local_int_address));

    // printf("Sending address: %p\n", (void *) local_int_address);

    ktm_rdma_post_send(id, nullptr, addr, length, mr, 0);

    ibv_wc wc;
    memset(&wc, 0, sizeof(wc));
    int num_wc = ktm_rdma_get_send_comp(id, &wc);
}

void ktm_rdma_send_address(struct rdma_cm_id *id, void *addr, size_t length, struct ibv_mr *mr,
                           struct ibv_mr *rdma_mr) {
    auto local_int_address = (uintptr_t)rdma_mr->addr;
    memcpy(addr, &local_int_address, sizeof(local_int_address));

    // printf("Sending address: %p\n", (void *) local_int_address);

    ktm_rdma_post_send(id, nullptr, addr, length, mr, 0);

    ibv_wc wc;
    memset(&wc, 0, sizeof(wc));
    int num_wc = ktm_rdma_get_send_comp(id, &wc);
}

uintptr_t ktm_rdma_get_address(struct rdma_cm_id *id, void *addr, size_t length,
                               struct ibv_mr *mr) {
    ktm_rdma_post_recv(id, nullptr, addr, length, mr);

    ibv_wc wc;
    memset(&wc, 0, sizeof(wc));
    int num_wc = ktm_rdma_get_recv_comp(id, &wc);

    uintptr_t peer_int_address = 0;
    memcpy(&peer_int_address, addr, sizeof(peer_int_address));

    // printf("Got address: %p\n", (void *) peer_int_address);

    return peer_int_address;
}

void ktm_rdma_send_rkey(struct rdma_cm_id *id, void *addr, size_t length, struct ibv_mr *mr) {
    auto local_rkey = mr->rkey;
    memcpy(addr, &local_rkey, sizeof(local_rkey));

    // printf("Sending rkey: %u\n", local_rkey);

    ktm_rdma_post_send(id, nullptr, addr, length, mr, 0);

    ibv_wc wc;
    memset(&wc, 0, sizeof(wc));
    int num_wc = ktm_rdma_get_send_comp(id, &wc);
}

void ktm_rdma_send_rkey(struct rdma_cm_id *id, void *addr, size_t length, struct ibv_mr *mr,
                        struct ibv_mr *rdma_mr) {
    auto local_rkey = rdma_mr->rkey;
    memcpy(addr, &local_rkey, sizeof(local_rkey));

    // printf("Sending rkey: %u\n", local_rkey);

    ktm_rdma_post_send(id, nullptr, addr, length, mr, 0);

    ibv_wc wc;
    memset(&wc, 0, sizeof(wc));
    int num_wc = ktm_rdma_get_send_comp(id, &wc);
}

uint32_t ktm_rdma_get_rkey(struct rdma_cm_id *id, void *addr, size_t length, struct ibv_mr *mr) {
    ktm_rdma_post_recv(id, nullptr, addr, length, mr);

    ibv_wc wc;
    memset(&wc, 0, sizeof(wc));
    int num_wc = ktm_rdma_get_recv_comp(id, &wc);

    uint32_t peer_rkey = 0;
    memcpy(&peer_rkey, addr, sizeof(peer_rkey));

    // printf("Got rkey: %u\n", peer_rkey);

    return peer_rkey;
}

void ktm_send_rdma_info(struct rdma_cm_id *id, void *addr, size_t length, struct ibv_mr *mr,
                        uintptr_t *remote_addr, uint32_t *remote_rkey) {
    ktm_rdma_send_address(id, addr, length, mr);
    ktm_rdma_send_rkey(id, addr, length, mr);
}

void ktm_recv_rdma_info(struct rdma_cm_id *id, void *addr, size_t length, struct ibv_mr *mr,
                        uintptr_t *remote_addr, uint32_t *remote_rkey) {
    uintptr_t received_addr = ktm_rdma_get_address(id, addr, length, mr);
    memcpy(remote_addr, &received_addr, sizeof(uintptr_t));

    uint32_t received_rkey = ktm_rdma_get_rkey(id, addr, length, mr);
    memcpy(remote_rkey, &received_rkey, sizeof(uint32_t));
}

void ktm_client_exchange_rdma_info(struct rdma_cm_id *id, void *addr, size_t length,
                                   struct ibv_mr *mr, uintptr_t *remote_addr,
                                   uint32_t *remote_rkey) {
    ktm_rdma_send_address(id, addr, length, mr);

    uintptr_t received_addr = ktm_rdma_get_address(id, addr, length, mr);
    memcpy(remote_addr, &received_addr, sizeof(uintptr_t));

    ktm_rdma_send_rkey(id, addr, length, mr);

    uint32_t received_rkey = ktm_rdma_get_rkey(id, addr, length, mr);
    memcpy(remote_rkey, &received_rkey, sizeof(uint32_t));
}

void ktm_server_exchange_rdma_info(struct rdma_cm_id *id, void *addr, size_t length,
                                   struct ibv_mr *mr, uintptr_t *remote_addr,
                                   uint32_t *remote_rkey) {
    uintptr_t received_addr = ktm_rdma_get_address(id, addr, length, mr);
    memcpy(remote_addr, &received_addr, sizeof(uintptr_t));

    ktm_rdma_send_address(id, addr, length, mr);

    uint32_t received_rkey = ktm_rdma_get_rkey(id, addr, length, mr);
    memcpy(remote_rkey, &received_rkey, sizeof(uint32_t));

    ktm_rdma_send_rkey(id, addr, length, mr);
}

void ktm_client_exchange_rdma_info(struct rdma_cm_id *id, void *addr, size_t length,
                                   struct ibv_mr *mr, uintptr_t *remote_addr, uint32_t *remote_rkey,
                                   struct ibv_mr *rdma_mr) {
    ktm_rdma_send_address(id, addr, length, mr, rdma_mr);

    uintptr_t received_addr = ktm_rdma_get_address(id, addr, length, mr);
    memcpy(remote_addr, &received_addr, sizeof(uintptr_t));

    ktm_rdma_send_rkey(id, addr, length, mr, rdma_mr);

    uint32_t received_rkey = ktm_rdma_get_rkey(id, addr, length, mr);
    memcpy(remote_rkey, &received_rkey, sizeof(uint32_t));
}

void ktm_server_exchange_rdma_info(struct rdma_cm_id *id, void *addr, size_t length,
                                   struct ibv_mr *mr, uintptr_t *remote_addr, uint32_t *remote_rkey,
                                   struct ibv_mr *rdma_mr) {
    uintptr_t received_addr = ktm_rdma_get_address(id, addr, length, mr);
    memcpy(remote_addr, &received_addr, sizeof(uintptr_t));

    ktm_rdma_send_address(id, addr, length, mr, rdma_mr);

    uint32_t received_rkey = ktm_rdma_get_rkey(id, addr, length, mr);
    memcpy(remote_rkey, &received_rkey, sizeof(uint32_t));

    ktm_rdma_send_rkey(id, addr, length, mr, rdma_mr);
}