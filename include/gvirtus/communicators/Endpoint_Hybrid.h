#pragma once

#include <nlohmann/json.hpp>
#include "Endpoint.h"

namespace gvirtus::communicators {

class Endpoint_Hybrid : public Endpoint {
private:
    std::string _suite;
    std::string _protocol;
    std::string _address;

    std::string _rdma_suite;
    std::uint16_t _rdma_port;

    std::string _tcp_suite;
    std::uint16_t _tcp_port;

public:
    Endpoint_Hybrid() = default;

    Endpoint_Hybrid(const std::string &suite,
                    const std::string &protocol,
                    const std::string &address,
                    const std::string &rdma_suite,
                    const std::string &rdma_port,
                    const std::string &tcp_suite,
                    const std::string &tcp_port);

    // Fluent setters
    Endpoint &suite(const std::string &suite) override;
    Endpoint &protocol(const std::string &protocol) override;
    Endpoint_Hybrid &address(const std::string &address);
    Endpoint_Hybrid &rdma_suite(const std::string &s);
    Endpoint_Hybrid &tcp_suite(const std::string &s);
    Endpoint_Hybrid &rdma_port(const std::string &port);
    Endpoint_Hybrid &tcp_port(const std::string &port);

    // Getters
    // Getters
    inline const std::string &suite() const { return _suite; }     // Added: allow reading suite value
    inline const std::string &protocol() const { return _protocol; }
    inline const std::string &rdma_suite() const { return _rdma_suite; }
    inline const std::string &tcp_suite() const { return _tcp_suite; }
    inline const std::string &address() const { return _address; }
    inline const std::uint16_t &rdma_port() const { return _rdma_port; }
    inline const std::uint16_t &tcp_port() const { return _tcp_port; }

    virtual inline const std::string to_string() const override {
        return _suite + ":" + _protocol + "://" + _address +
               " (rdma=" + _rdma_suite + ":" + std::to_string(_rdma_port) +
               ", tcp=" + _tcp_suite + ":" + std::to_string(_tcp_port) + ")";
    }

    // JSON loader
    friend void from_json(const nlohmann::json &j, Endpoint_Hybrid &end);
};

}  // namespace gvirtus::communicators
