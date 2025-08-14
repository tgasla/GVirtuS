//
// Created by Mariano Aponte on 18/12/23.
//

#ifndef GVIRTUS_ENDPOINT_RDMA_H
#define GVIRTUS_ENDPOINT_RDMA_H

#pragma once

#include <nlohmann/json.hpp>

#include "Endpoint.h"

namespace gvirtus::communicators {

class Endpoint_Rdma : public Endpoint {
   private:
    std::string _address;
    std::uint16_t _port;

   public:
    Endpoint_Rdma() = default;

    explicit Endpoint_Rdma(const std::string &endp_suite, const std::string &endp_protocol,
                           const std::string &endp_address, const std::string &endp_port);

    explicit Endpoint_Rdma(const std::string &endp_suite)
        : Endpoint_Rdma(endp_suite, "ib", "127.0.0.1", "9999") {}

    Endpoint &suite(const std::string &suite) override;
    Endpoint &protocol(const std::string &protocol) override;

    Endpoint_Rdma &address(const std::string &address);
    inline const std::string &address() const { return _address; }

    Endpoint_Rdma &port(const std::string &port);
    inline const std::uint16_t &port() const { return _port; }

    virtual inline const std::string to_string() const {
        return _suite + _protocol + _address + std::to_string(_port);
    }

    std::string to_string() { return "EndpointRdma"; }
};
void from_json(const nlohmann::json &j, Endpoint_Rdma &end);
}  // namespace gvirtus::communicators
#endif  // GVIRTUS_ENDPOINT_RDMA_H
