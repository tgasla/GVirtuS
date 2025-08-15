#include "gvirtus/communicators/Endpoint_Hybrid.h"
#include "gvirtus/communicators/EndpointFactory.h"

#include <regex>
#include <iostream>

using namespace gvirtus::communicators;

Endpoint_Hybrid::Endpoint_Hybrid(const std::string &suite,
                                 const std::string &protocol,
                                 const std::string &address,
                                 const std::string &rdma_suite,
                                 const std::string &rdma_port,
                                 const std::string &tcp_suite,
                                 const std::string &tcp_port) {
    this->suite(suite);
    this->protocol(protocol);
    this->address(address);
    this->rdma_suite(rdma_suite);
    this->tcp_suite(tcp_suite);
    this->rdma_port(rdma_port);
    this->tcp_port(tcp_port);
}

Endpoint &Endpoint_Hybrid::suite(const std::string &suite) {
    _suite = suite;
    return *this;
}

Endpoint &Endpoint_Hybrid::protocol(const std::string &protocol) {
    _protocol = protocol;
    return *this;
}

Endpoint_Hybrid &Endpoint_Hybrid::address(const std::string &address) {
    _address = address;
    return *this;
}

Endpoint_Hybrid &Endpoint_Hybrid::rdma_suite(const std::string &s) {
    _rdma_suite = s;
    return *this;
}

Endpoint_Hybrid &Endpoint_Hybrid::tcp_suite(const std::string &s) {
    _tcp_suite = s;
    return *this;
}

Endpoint_Hybrid &Endpoint_Hybrid::rdma_port(const std::string &port) {
    _rdma_port = static_cast<uint16_t>(std::stoi(port));
    return *this;
}

Endpoint_Hybrid &Endpoint_Hybrid::tcp_port(const std::string &port) {
    _tcp_port = static_cast<uint16_t>(std::stoi(port));
    return *this;
}

void gvirtus::communicators::from_json(const nlohmann::json &j, Endpoint_Hybrid &end) {
    auto el = j["communicator"][EndpointFactory::index()]["endpoint"];

    end.suite(el.at("suite"));
    end.protocol(el.at("protocol"));
    end.address(el.at("server_address"));

    // Hybrid specific
    end.rdma_suite(el.at("rdma_suite"));
    end.rdma_port(el.at("rdma_port"));
    end.tcp_suite(el.at("tcp_suite"));
    end.tcp_port(el.at("tcp_port"));
}
