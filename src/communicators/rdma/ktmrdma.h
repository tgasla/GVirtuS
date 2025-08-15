//
// Created by Mariano Aponte on 30/11/23.
//

#ifndef RDMACM_KTMRDMA_H
#define RDMACM_KTMRDMA_H

#include <errno.h>
#include <string>
#include <iostream>
#include <sys/types.h>
#include <sys/socket.h>
#include <netdb.h>
#include <rdma/rdma_cma.h>
#include <rdma/rdma_verbs.h>

#define DEBUG

    void Testlib();
/**
 * @brief Wrapper function for rdma_getaddrinfo with error handling.
 *
 * @param node A pointer to the node name or IP address.
 * @param service A pointer to the service name or port number.
 * @param hints A pointer to the hints for resolving the address.
 * @param res A pointer to the result of rdma_getaddrinfo.
 * @throws std::string If rdma_getaddrinfo encounters an error, an exception is thrown with an error message.
 */
void ktm_rdma_getaddrinfo(char *node, char *service, struct rdma_addrinfo *hints, struct rdma_addrinfo **res);

/**
 * @brief Wrapper function for rdma_create_ep with error handling.
 *
 * @param id A pointer to the RDMA communication identifier.
 * @param res A pointer to the result of rdma_getaddrinfo.
 * @param pd A pointer to the protection domain.
 * @param qp_init_attr A pointer to the queue pair initialization attributes.
 * @throws std::string If rdma_create_ep encounters an error, an exception is thrown with an error message.
 */
void ktm_rdma_create_ep(struct rdma_cm_id **id, struct rdma_addrinfo *res, struct ibv_pd *pd, struct ibv_qp_init_attr *qp_init_attr);

/**
 * @brief Wrapper function for rdma_reg_msgs with error handling.
 *
 * @param id A pointer to the RDMA communication identifier.
 * @param addr A pointer to the memory region to be registered.
 * @param length The length of the memory region.
 * @return A pointer to the registered memory region.
 * @throws std::string If rdma_reg_msgs encounters an error, an exception is thrown with an error message.
 */
ibv_mr * ktm_rdma_reg_msgs(struct rdma_cm_id *id, void *addr, size_t length);

/**
 * @brief Wrapper function for rdma_reg_read with error handling.
 *
 * @param id A pointer to the RDMA communication identifier.
 * @param addr A pointer to the local buffer to be registered for read operations.
 * @param length The length of the local buffer.
 * @return A pointer to the memory region associated with the registered buffer.
 * @throws std::string If rdma_reg_read encounters an error, an exception is thrown with an error message.
 */
ibv_mr * ktm_rdma_reg_read(struct rdma_cm_id *id, void *addr, size_t length);

/**
 * @brief Wrapper function for rdma_reg_write with error handling.
 *
 * @param id A pointer to the RDMA communication identifier.
 * @param addr A pointer to the local buffer to be registered for write operations.
 * @param length The length of the local buffer.
 * @return A pointer to the memory region associated with the registered buffer.
 * @throws std::string If rdma_reg_write encounters an error, an exception is thrown with an error message.
 */
ibv_mr * ktm_rdma_reg_write(struct rdma_cm_id *id, void *addr, size_t length);

/**
 * @brief Wrapper function for rdma_connect with error handling.
 *
 * @param id A pointer to the RDMA communication identifier.
 * @param conn_param A pointer to the connection parameters.
 * @throws std::string If rdma_connect encounters an error, an exception is thrown with an error message.
 */
void ktm_rdma_connect(struct rdma_cm_id *id, struct rdma_conn_param *conn_param);

/**
 * @brief Wrapper function for rdma_listen with error handling.
 *
 * @param id A pointer to the RDMA communication identifier.
 * @param backlog The maximum number of pending connections in the listen queue.
 * @throws std::string If rdma_listen encounters an error, an exception is thrown with an error message.
 */
void ktm_rdma_listen(struct rdma_cm_id *id, int backlog);

/**
 * @brief Wrapper function for rdma_get_request with error handling.
 *
 * @param listen A pointer to the RDMA communication identifier representing the listening endpoint.
 * @param id A pointer to the RDMA communication identifier for the incoming connection request.
 * @throws std::string If rdma_get_request encounters an error, an exception is thrown with an error message.
 */
void ktm_rdma_get_request(struct rdma_cm_id *listen, struct rdma_cm_id **id);

/**
 * @brief Wrapper function for rdma_accept with error handling.
 *
 * @param id A pointer to the RDMA communication identifier.
 * @param conn_param A pointer to the connection parameters.
 * @throws std::string If rdma_accept encounters an error, an exception is thrown with an error message.
 */
void ktm_rdma_accept(struct rdma_cm_id *id, struct rdma_conn_param *conn_param);

/**
 * @brief Wrapper function for rdma_post_recv with error handling.
 *
 * @param id A pointer to the RDMA communication identifier.
 * @param context A user-defined context associated with the receive operation.
 * @param addr A pointer to the receive buffer.
 * @param length The length of the receive buffer.
 * @param mr A pointer to the memory region associated with the receive buffer.
 * @throws std::string If rdma_post_recv encounters an error, an exception is thrown with an error message.
 */
void ktm_rdma_post_recv(struct rdma_cm_id *id, void *context, void *addr, size_t length, struct ibv_mr *mr);

/**
 * @brief Wrapper function for rdma_post_send with error handling.
 *
 * @param id A pointer to the RDMA communication identifier.
 * @param context A user-defined context associated with the send operation.
 * @param addr A pointer to the send buffer.
 * @param length The length of the send buffer.
 * @param mr A pointer to the memory region associated with the send buffer.
 * @param flags Flags to control the behavior of the send operation.
 * @throws std::string If rdma_post_send encounters an error, an exception is thrown with an error message.
 */
void ktm_rdma_post_send (struct rdma_cm_id *id, void *context, void *addr, size_t length, struct ibv_mr *mr, int flags);

/**
 * @brief Wrapper function for rdma_get_send_comp with error handling.
 *
 * @param id A pointer to the RDMA communication identifier.
 * @param wc A pointer to the work completion structure for send completion details.
 * @return The result of rdma_get_send_comp.
 * @throws std::string If rdma_get_send_comp encounters an error, an exception is thrown with an error message.
 */
int ktm_rdma_get_send_comp(struct rdma_cm_id *id, struct ibv_wc *wc);

/**
 * @brief Wrapper function for rdma_get_recv_comp with error handling.
 *
 * @param id A pointer to the RDMA communication identifier.
 * @param wc A pointer to the work completion structure for receive completion details.
 * @return The result of rdma_get_recv_comp.
 * @throws std::string If rdma_get_recv_comp encounters an error, an exception is thrown with an error message.
 */
int ktm_rdma_get_recv_comp(struct rdma_cm_id *id, struct ibv_wc *wc);

/**
 * @brief Wrapper function for rdma_post_read with error handling.
 *
 * @param id A pointer to the RDMA communication identifier.
 * @param context A user-defined context associated with the read operation.
 * @param addr A pointer to the local buffer for the read operation.
 * @param length The length of the local buffer.
 * @param mr A pointer to the local memory region associated with the buffer.
 * @param flags Flags to control the behavior of the read operation.
 * @param remote_addr The remote address to read from.
 * @param rkey The remote key associated with the remote memory region.
 * @throws std::string If rdma_post_read encounters an error, an exception is thrown with an error message.
 */
void ktm_rdma_post_read(struct rdma_cm_id *id, void *context, void *addr, size_t length, struct ibv_mr *mr, int flags, uint64_t remote_addr, uint32_t rkey);

/**
 * @brief Wrapper function for rdma_post_write with error handling.
 *
 * @param id A pointer to the RDMA communication identifier.
 * @param context A user-defined context associated with the write operation.
 * @param addr A pointer to the local buffer for the write operation.
 * @param length The length of the local buffer.
 * @param mr A pointer to the local memory region associated with the buffer.
 * @param flags Flags to control the behavior of the write operation.
 * @param remote_addr The remote address to write to.
 * @param rkey The remote key associated with the remote memory region.
 * @throws std::string If rdma_post_write encounters an error, an exception is thrown with an error message.
 */
void ktm_rdma_post_write (struct rdma_cm_id *id, void *context, void *addr, size_t length, struct ibv_mr *mr, int flags, uint64_t remote_addr, uint32_t rkey);

void ktm_send_rdma_info(struct rdma_cm_id *id, void *addr, size_t length, struct ibv_mr *mr, uintptr_t * remote_addr, uint32_t * remote_rkey);

void ktm_recv_rdma_info(struct rdma_cm_id *id, void *addr, size_t length, struct ibv_mr *mr, uintptr_t * remote_addr, uint32_t * remote_rkey);

/**
 * @brief Exchange RDMA information from the client's perspective.
 *
 * This function performs the exchange of RDMA-related information such as the remote address and remote rkey
 * between the client and the server using RDMA operations.
 *
 * @note This function must be called AFTER calling `ktm_server_exchange_rdma_info` function on the server side.
 * The two functions must always be used together to ensure a proper exchange of RDMA information.
 *
 * @param id A pointer to the RDMA communication identifier.
 * @param addr A pointer to the local buffer for RDMA operations.
 * @param length The length of the local buffer.
 * @param mr A pointer to the local memory region associated with the buffer.
 * @param remote_addr A pointer to store the received remote address.
 * @param remote_rkey A pointer to store the received remote rkey.
 */
void ktm_client_exchange_rdma_info(struct rdma_cm_id *id, void *addr, size_t length, struct ibv_mr *mr, uintptr_t * remote_addr, uint32_t * remote_rkey);

void ktm_client_exchange_rdma_info(struct rdma_cm_id *id, void *addr, size_t length, struct ibv_mr *mr, uintptr_t * remote_addr, uint32_t * remote_rkey, struct ibv_mr *rdma_mr);

/**
 * @brief Exchange RDMA information from the server's perspective.
 *
 * This function performs the exchange of RDMA-related information such as the remote address and remote rkey
 * between the server and the client using RDMA operations.
 *
 * @note This function must be called BEFORE calling `ktm_client_exchange_rdma_info` function on the client side.
 * The two functions must always be used together to ensure a proper exchange of RDMA information.
 *
 * @param id A pointer to the RDMA communication identifier.
 * @param addr A pointer to the local buffer for RDMA operations.
 * @param length The length of the local buffer.
 * @param mr A pointer to the local memory region associated with the buffer.
 * @param remote_addr A pointer to store the received remote address.
 * @param remote_rkey A pointer to store the received remote rkey.
 */
void ktm_server_exchange_rdma_info(struct rdma_cm_id *id, void *addr, size_t length, struct ibv_mr *mr, uintptr_t * remote_addr, uint32_t * remote_rkey);

void ktm_server_exchange_rdma_info(struct rdma_cm_id *id, void *addr, size_t length, struct ibv_mr *mr, uintptr_t * remote_addr, uint32_t * remote_rkey, struct ibv_mr *rdma_mr);


#endif //RDMACM_KTMRDMA_H
