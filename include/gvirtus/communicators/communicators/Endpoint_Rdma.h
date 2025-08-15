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
    std::string _suite;       // Added: suite type (e.g. infiniband-rdma or rdma-roce)
    std::string _protocol;    // e.g. "ib", "tcp"
    std::string _address;
    std::uint16_t _port;

public:
    Endpoint_Rdma() = default;

    // Full constructor
    explicit Endpoint_Rdma(const std::string &endp_suite,
                           const std::string &endp_protocol,
                           const std::string &endp_address,
                           const std::string &endp_port);

    // Default constructor fallback (for quick testing)
    explicit Endpoint_Rdma(const std::string &endp_suite)
        : Endpoint_Rdma(endp_suite, "ib", "127.0.0.1", "9999") {}

    // Getters
    inline const std::string &suite() const { return _suite; }     // Added: allow reading suite value
    inline const std::string &protocol() const { return _protocol; }
    inline const std::string &address() const { return _address; }
    inline const std::uint16_t &port() const { return _port; }

    // Setters (for factory compatibility)
    Endpoint &suite(const std::string &suite) override;
    Endpoint &protocol(const std::string &protocol) override;
    Endpoint_Rdma &address(const std::string &address);
    Endpoint_Rdma &port(const std::string &port);

    virtual inline const std::string to_string() const override {
        return _suite + ":" + _protocol + "://" + _address + ":" + std::to_string(_port);
    }

    friend void from_json(const nlohmann::json &j, Endpoint_Rdma &end);
};

}  // namespace gvirtus::communicators

#endif //GVIRTUS_ENDPOINT_RDMA_H
